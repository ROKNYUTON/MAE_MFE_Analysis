from pathlib import Path
import os
import sys
import warnings
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from io import StringIO
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[WARN] requests not installed. Install with 'pip install requests' to fetch VIX from CBOE if needed.")
# Optional libs
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
try:
    from polygon import RESTClient
    POLYGON_AVAILABLE = True
except Exception:
    POLYGON_AVAILABLE = False
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("[WARN] yfinance not installed. Install with 'pip install yfinance' to fetch VIX/BTC data.")

# ---------------- CONFIG ----------------
BASE = Path(".")
REPORT_PATH = BASE / "data" / "reports" / "multi_asset_report.csv"
DATASETS_FOLDER = BASE / "data" / "datasets"
NEWS_FOLDER = BASE / "data" / "news"
OUTPUT_DIR = BASE / "data" / "reports" / "ml_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INITIAL_BALANCE = 100000.0
CONTRACT_SIZE = {
    "GOLD.pro": 100, "USDJPY.pro": 1000, "US100.pro": 20,
    "US500.pro": 50, "DE30.pro": 25,
}
TOP_N_DRAWDOWNS = 20
TOP_N_GOOD = 20
RANDOM_STATE = 42
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# ---------------- Helpers: logging ----------------
def log_info(msg):
    print("[INFO]", msg)

def log_debug(msg):
    print("[DEBUG]", msg)

def log_warn(msg):
    print("[WARN]", msg)

# ---------------- Load price data ----------------
def load_price_data(symbol, timeframe_preference=('M5','D1')):
    """Carica file {symbol}_{TF}_2020_2025.csv."""
    for tf in timeframe_preference:
        path = DATASETS_FOLDER / f"{symbol}_{tf}_2020_2025.csv"
        if not path.exists():
            continue
        try:
            try:
                df = pd.read_csv(path, sep='\t')
            except Exception:
                df = pd.read_csv(path)
            df.columns = [c.replace('<','').replace('>','').strip() for c in df.columns]
            if 'DATE' in df.columns and 'TIME' in df.columns:
                df['time'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str),
                                            format='%Y.%m.%d %H:%M:%S', errors='coerce')
                df.set_index('time', inplace=True)
            elif 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
                df.set_index('time', inplace=True)
            else:
                try:
                    df.index = pd.to_datetime(df.iloc[:,0], errors='coerce')
                except Exception:
                    pass
            cols_map = {}
            for c in df.columns:
                if c.lower() in ['open','high','low','close']:
                    cols_map[c] = c.upper()
            if cols_map:
                df.rename(columns=cols_map, inplace=True)
            for col in ['OPEN','HIGH','LOW','CLOSE']:
                if col not in df.columns:
                    raise ValueError(f"Missing {col} in {path}")
            df = df[['OPEN','HIGH','LOW','CLOSE']].astype(float)
            df = df[~df.index.isna()].sort_index()
            log_info(f"Loaded {symbol} {tf} rows={len(df)}")
            return df
        except Exception as e:
            log_warn(f"Failed to load {path}: {e}")
            continue
    
    yf_symbols = {'VIX': '^VIX', 'BTC': 'BTC-USD'}
    if symbol in yf_symbols and YFINANCE_AVAILABLE:
        for tf in timeframe_preference:
            if tf == 'M5':
                interval = '5m'
            elif tf == 'D1':
                interval = '1d'
            else:
                continue
            try:
                df = yf.download(yf_symbols[symbol], start='2020-01-01', end='2025-12-31', interval=interval, progress=False)
                if df.empty or len(df) < 100:
                    continue
                df = df[['Open', 'High', 'Low', 'Close']].rename(columns={'Open': 'OPEN', 'High': 'HIGH', 'Low': 'LOW', 'Close': 'CLOSE'})
                df.index.name = 'time'
                df = df.astype(float)
                log_info(f"Fetched {symbol} {tf} rows={len(df)} from yfinance")
                path = DATASETS_FOLDER / f"{symbol}_{tf}_2020_2025.csv"
                df.to_csv(path)
                return df
            except Exception as e:
                log_warn(f"Failed to fetch {symbol} {tf} from yfinance: {e}")
                continue
    
    if symbol == 'VIX' and REQUESTS_AVAILABLE:
        for tf in timeframe_preference:
            if tf != 'D1':
                continue
            try:
                url = 'https://cdn.cboe.com/api/global/us_indices/daily_values/VIX_History.csv'
                response = requests.get(url)
                response.raise_for_status()
                df = pd.read_csv(StringIO(response.text))
                df['time'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y', errors='coerce')
                df = df.set_index('time')[['OPEN', 'HIGH', 'LOW', 'CLOSE']]
                df = df.loc['2020-01-01':'2025-12-31']
                df = df.dropna()
                log_info(f"Fetched VIX {tf} rows={len(df)} from CBOE")
                path = DATASETS_FOLDER / f"{symbol}_{tf}_2020_2025.csv"
                df.to_csv(path)
                return df
            except Exception as e:
                log_warn(f"Failed to fetch VIX {tf} from CBOE: {e}")
                continue
    
    log_warn(f"No dataset found for {symbol} (checked {timeframe_preference})")
    return None

# ---------------- Equity high-res (OPTIMIZED) ----------------
def compute_equity_highres(df_trades, m5_db, initial_balance=INITIAL_BALANCE, freq='5min'):
    """Ricostruisce equity ad alta risoluzione - VERSIONE OTTIMIZZATA."""
    log_info("Computing high-resolution equity (optimized)...")
    df = df_trades.copy()
    if 'Open time' in df.columns and 'Close time' in df.columns:
        df['Open time'] = pd.to_datetime(df['Open time'], dayfirst=True, errors='coerce')
        df['Close time'] = pd.to_datetime(df['Close time'], dayfirst=True, errors='coerce')
    else:
        raise ValueError("Trades file missing 'Open time'/'Close time' columns")
    
    start = df['Open time'].min()
    end = df['Close time'].max()
    timeline = pd.date_range(start=start, end=end, freq=freq)
    high_res = pd.DataFrame(index=timeline)
    high_res.index.name = 'time'
    
    bc = df[['Close time','Profit/Loss (Global)']].copy().sort_values('Close time')
    bc['Cumulative'] = initial_balance + bc['Profit/Loss (Global)'].cumsum()
    merged = pd.merge_asof(high_res, bc.rename(columns={'Close time':'time'}), on='time', direction='backward')
    merged.set_index('time', inplace=True)
    merged['balance'] = merged['Cumulative'].fillna(initial_balance)
    merged['equity_close'] = merged['balance'].copy()
    merged['equity_low'] = merged['balance'].copy()
    
    # OTTIMIZZAZIONE: Pre-process price data per symbol e usa merge_asof invece di reindex
    log_info(f"Processing {len(df)} trades for unrealized P/L...")
    for sym in df['Symbol'].unique():
        if sym not in m5_db:
            continue
        
        # Filtra trade per questo symbol
        sym_trades = df[df['Symbol'] == sym].copy()
        
        # Pre-process price data: prepara solo i dati necessari
        price_data = m5_db[sym][['CLOSE', 'LOW', 'HIGH']].copy()
        
        # Per ogni trade di questo symbol
        for idx, tr in sym_trades.iterrows():
            mask = (merged.index >= tr['Open time']) & (merged.index < tr['Close time'])
            if not mask.any():
                continue
            
            # Usa merge_asof invece di reindex per performance
            subset_times = merged.index[mask].to_frame(index=False, name='time')
            subset_with_prices = pd.merge_asof(
                subset_times, 
                price_data.reset_index(), 
                left_on='time', 
                right_on='time', 
                direction='backward'
            ).set_index('time')
            
            if subset_with_prices.empty:
                continue
            
            mult = float(tr.get('Size',1)) * CONTRACT_SIZE.get(sym, 100)
            entry = float(tr.get('Open price', np.nan))
            is_buy = 'Buy' in str(tr.get('Type',''))
            
            if is_buy:
                pl_close = (subset_with_prices['CLOSE'] - entry) * mult
                pl_low = (subset_with_prices['LOW'] - entry) * mult
            else:
                pl_close = (entry - subset_with_prices['CLOSE']) * mult
                pl_low = (entry - subset_with_prices['HIGH']) * mult
            
            merged.loc[mask, 'equity_close'] += pl_close.values
            merged.loc[mask, 'equity_low'] += pl_low.values
    
    log_info("Equity computed (optimized).")
    return merged

# ---------------- Drawdown episodes ----------------
def detect_all_drawdown_episodes(equity_series, debug=False):
    """Detect all peak->trough->recovery episodes."""
    s = equity_series.dropna().copy()
    if s.empty:
        return []
    peak = s.cummax()
    dd = s - peak
    in_dd = dd < 0
    diff = in_dd.astype(int).diff().fillna(0)
    starts = s.index[diff == 1].tolist()
    ends = s.index[diff == -1].tolist()
    
    if len(starts) > 0 and (len(ends) == 0 or starts[0] > ends[0]):
        ends = ends[1:]
    n = min(len(starts), len(ends))
    episodes = []
    for i in range(n):
        start = starts[i]
        end = ends[i]
        window = s.loc[start:end]
        trough_time = window.idxmin()
        trough_value = window.min()
        peak_value = s.loc[start]
        drawdown_amt = trough_value - peak_value
        drawdown_pct = drawdown_amt / peak_value if peak_value != 0 else np.nan
        duration = (end - start)
        episodes.append({
            'peak_time': start,
            'trough_time': trough_time,
            'recovery_time': end,
            'drawdown_amount': drawdown_amt,
            'drawdown_pct': drawdown_pct,
            'duration': duration
        })
        if debug and drawdown_amt <= -2000:
            log_debug(f"Large DD: start={start}, trough={trough_time}, recovery={end}, amt={drawdown_amt}")
    
    if not episodes:
        trough_time = s.idxmin()
        peak_time = s[:trough_time].idxmax() if (s.index[0] < trough_time) else s.index[0]
        episodes.append({
            'peak_time': peak_time,
            'trough_time': trough_time,
            'recovery_time': pd.NaT,
            'drawdown_amount': s.loc[trough_time] - s.loc[peak_time],
            'drawdown_pct': (s.loc[trough_time] - s.loc[peak_time]) / s.loc[peak_time],
            'duration': trough_time - peak_time
        })
    return episodes

# ---------------- VIX regime helper ----------------
def vix_regime(v):
    try:
        v = float(v)
    except Exception:
        return 'UNKNOWN'
    if math.isnan(v):
        return 'UNKNOWN'
    if v < 15:
        return 'LOW'
    elif v < 20:
        return 'MEDIUM'
    elif v < 30:
        return 'HIGH'
    else:
        return 'CRISIS'

# ---------------- Analyze drawdowns vs VIX & assets ----------------
def analyze_drawdowns_and_vix(merged_df, assets_db, thresholds=(2000,1000)):
    log_info("Analyzing drawdown episodes vs VIX/assets...")
    equity = merged_df['equity_close']
    episodes = detect_all_drawdown_episodes(equity, debug=True)
    rows = []
    for e in episodes:
        start = e['peak_time']
        trough = e['trough_time']
        recovery = e['recovery_time']
        dd_amt = -float(e['drawdown_amount'])
        duration_sec = e['duration'].total_seconds() if pd.notna(e['duration']) else np.nan
        
        vix_mean = np.nan; vix_max = np.nan; vix_reg = 'UNKNOWN'
        if 'VIX' in assets_db:
            try:
                vix_series = assets_db['VIX']['CLOSE'].reindex(pd.date_range(start=start, end=recovery, freq='D'), method='ffill').dropna()
                if not vix_series.empty:
                    vix_mean = float(vix_series.mean()); vix_max = float(vix_series.max()); vix_reg = vix_regime(vix_mean)
            except Exception:
                pass
        
        us500_corr = np.nan; us100_corr = np.nan
        try:
            if 'US500.pro' in assets_db:
                a = assets_db['US500.pro']['CLOSE'].reindex(pd.date_range(start=start, end=recovery, freq='D'), method='ffill').dropna().pct_change()
                eqr = equity.loc[start:recovery].pct_change()
                aligned = pd.concat([a, eqr], axis=1).dropna()
                if not aligned.empty:
                    us500_corr = float(aligned.iloc[:,0].corr(aligned.iloc[:,1]))
        except Exception:
            pass
        try:
            if 'US100.pro' in assets_db:
                a = assets_db['US100.pro']['CLOSE'].reindex(pd.date_range(start=start, end=recovery, freq='D'), method='ffill').dropna().pct_change()
                eqr = equity.loc[start:recovery].pct_change()
                aligned = pd.concat([a, eqr], axis=1).dropna()
                if not aligned.empty:
                    us100_corr = float(aligned.iloc[:,0].corr(aligned.iloc[:,1]))
        except Exception:
            pass
        
        rows.append({
            'start': start, 'trough': trough, 'recovery': recovery,
            'drawdown_amount': dd_amt, 'duration_sec': duration_sec,
            'vix_mean': vix_mean, 'vix_max': vix_max, 'vix_regime': vix_reg,
            'US500_corr': us500_corr, 'US100_corr': us100_corr
        })
    
    events_df = pd.DataFrame(rows)
    events_df.to_csv(OUTPUT_DIR / 'drawdown_events_all.csv', index=False)
    log_info(f"Detected {len(events_df)} drawdown episodes. Saved to drawdown_events_all.csv")
    
    suggested_rules = []
    for thr in sorted(thresholds, reverse=True):
        df_thr = events_df[events_df['drawdown_amount'] >= thr]
        df_thr.to_csv(OUTPUT_DIR / f'drawdown_events_ge_{int(thr)}.csv', index=False)
        cnt = len(df_thr)
        pct = cnt / len(events_df) * 100 if len(events_df) > 0 else 0
        log_info(f"Episodes >= {thr}: {cnt} ({pct:.1f}%)")
        if 'vix_regime' in df_thr.columns and not df_thr['vix_regime'].isna().all():
            regime_counts = df_thr['vix_regime'].value_counts(normalize=True) * 100
            log_info(f"VIX regimes for >= {thr}: {regime_counts.to_dict()}")
            crisis_pct = regime_counts.get('CRISIS', 0) + regime_counts.get('HIGH', 0)
            if crisis_pct >= 50:
                suggested_rules.append(f"If VIX >= 30 or VIX regime HIGH/CRISIS -> reduce size by 50% or pause new trades (empirical {crisis_pct:.1f}% for drawdowns >= {thr}).")
            elif crisis_pct >= 25:
                suggested_rules.append(f"If VIX >= 25 -> consider reducing size by 30% (empirical {crisis_pct:.1f}% for drawdowns >= {thr}).")
        
        mean_us100_corr = df_thr['US100_corr'].dropna().mean() if 'US100_corr' in df_thr.columns else np.nan
        if not np.isnan(mean_us100_corr) and mean_us100_corr > 0.25:
            suggested_rules.append(f"High mean correlation with US100 ({mean_us100_corr:.2f}) for drawdowns >= {thr}: consider hedging Nasdaq exposure when correlation > 0.25.")
    
    summary = {
        'total_episodes': len(events_df),
        'events_ge_1000': int((events_df['drawdown_amount'] >= 1000).sum()) if not events_df.empty else 0,
        'events_ge_2000': int((events_df['drawdown_amount'] >= 2000).sum()) if not events_df.empty else 0
    }
    pd.Series(summary).to_csv(OUTPUT_DIR / 'drawdown_summary.csv')
    log_info("Drawdown summary saved to drawdown_summary.csv")
    log_info("Suggested rules:")
    for r in suggested_rules:
        log_info(" - " + r)
    return suggested_rules

# ---------------- Top best periods ----------------
def top_k_best_periods(equity, freq='W', top_k=20):
    rets = equity.pct_change().dropna()
    try:
        agg = rets.resample(freq).apply(lambda x: (1 + x).prod() - 1)
    except Exception:
        agg = rets.resample('D').apply(lambda x: (1 + x).prod() - 1)
    agg = agg.dropna()
    if agg.empty:
        return []
    top = agg.sort_values(ascending=False).head(top_k)
    periods = []
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except Exception:
        offset = pd.tseries.frequencies.to_offset('D')
    for idx, r in top.items():
        start = (pd.to_datetime(idx) - offset + pd.Timedelta(seconds=1))
        periods.append({'start': start, 'end': pd.to_datetime(idx), 'return': float(r)})
    return periods

# ---------------- Compute window features ----------------
def compute_window_features(start, end, equity_series, assets_db, news_df=None):
    features = {}
    start = pd.to_datetime(start); end = pd.to_datetime(end)
    features['window_start'] = start; features['window_end'] = end
    window_eq = equity_series.loc[start:end]
    if window_eq.empty:
        features.update({
            'eq_return': np.nan, 'eq_max': np.nan, 'eq_min': np.nan, 'eq_vol': np.nan,
            'eq_window_maxdd': np.nan, 'eq_mean_ret': np.nan, 'eq_skew': np.nan, 'eq_kurt': np.nan
        })
    else:
        features['eq_return'] = window_eq.iloc[-1] / window_eq.iloc[0] - 1 if len(window_eq) > 1 else 0.0
        features['eq_max'] = window_eq.max(); features['eq_min'] = window_eq.min()
        features['eq_vol'] = window_eq.pct_change().std()
        peak = window_eq.cummax(); features['eq_window_maxdd'] = (window_eq - peak).min()
        features['eq_mean_ret'] = window_eq.pct_change().mean(); features['eq_skew'] = window_eq.pct_change().skew()
        features['eq_kurt'] = window_eq.pct_change().kurt()
    
    for sym, df in assets_db.items():
        try:
            s = df['CLOSE'].reindex(pd.date_range(start=start, end=end, freq='D'), method='ffill').dropna()
        except Exception:
            s = df['CLOSE'].loc[start:end].ffill()
        if s.empty:
            features[f'{sym}_ret'] = np.nan; features[f'{sym}_vol'] = np.nan; features[f'{sym}_maxdd'] = np.nan; features[f'{sym}_corr_with_eq'] = np.nan
            continue
        sret = s.iloc[-1] / s.iloc[0] - 1 if len(s) > 1 else 0.0
        features[f'{sym}_ret'] = sret
        features[f'{sym}_vol'] = s.pct_change().std()
        peak_s = s.cummax(); features[f'{sym}_maxdd'] = (s - peak_s).min()
        try:
            aligned = pd.concat([s.pct_change(), equity_series.loc[start:end].pct_change()], axis=1).dropna()
            aligned.columns = ['asset','eq']
            features[f'{sym}_corr_with_eq'] = aligned['asset'].corr(aligned['eq'])
        except Exception:
            features[f'{sym}_corr_with_eq'] = np.nan
    
    if 'VIX' in assets_db:
        try:
            v = assets_db['VIX']['CLOSE'].reindex(pd.date_range(start=start, end=end, freq='D'), method='ffill').dropna()
            features['vix_mean'] = v.mean() if not v.empty else np.nan
            features['vix_max'] = v.max() if not v.empty else np.nan
            features['vix_regime'] = vix_regime(features['vix_mean']) if not math.isnan(features.get('vix_mean', np.nan)) else 'UNKNOWN'
        except Exception:
            features['vix_mean'] = np.nan; features['vix_max'] = np.nan; features['vix_regime'] = 'UNKNOWN'
    
    if news_df is not None and not news_df.empty:
        news_window = news_df[(news_df['time'] >= start) & (news_df['time'] <= end)]
        features['news_count'] = len(news_window)
        if 'sentiment' in news_window.columns:
            features['news_sent_mean'] = news_window['sentiment'].mean(); features['news_sent_std'] = news_window['sentiment'].std()
        else:
            features['news_sent_mean'] = np.nan; features['news_sent_std'] = np.nan
    else:
        features['news_count'] = np.nan; features['news_sent_mean'] = np.nan; features['news_sent_std'] = np.nan
    return features

# ---------------- Build ML dataset ----------------
def build_ml_dataset(merged, df_trades, assets_to_load=('US500.pro','US100.pro','BTC','VIX','GOLD.pro'), top_n_draw=TOP_N_DRAWDOWNS, top_n_good=TOP_N_GOOD, news_path=None):
    assets_db = {}
    for s in assets_to_load:
        try:
            dfp = load_price_data(s, timeframe_preference=('D1','M5'))
            if dfp is not None:
                assets_db[s] = dfp
        except Exception:
            continue
    equity = merged['equity_close']
    
    dd_eps = detect_all_drawdown_episodes(equity)
    dd_windows = []
    for e in dd_eps:
        start = e.get('peak_time'); end = e.get('recovery_time') if pd.notna(e.get('recovery_time')) else e.get('trough_time')
        dd_windows.append((start,end))
    
    best_periods = top_k_best_periods(equity, freq='W', top_k=top_n_good)
    best_windows = [(p['start'], p['end']) for p in best_periods]
    
    news_df = None
    if news_path and Path(news_path).exists():
        news_df = pd.read_csv(news_path); news_df['time'] = pd.to_datetime(news_df['time'], errors='coerce')
    
    rows = []
    for s,e in dd_windows:
        f = compute_window_features(s,e,equity, assets_db, news_df=news_df); f['label'] = 1; rows.append(f)
    for s,e in best_windows:
        f = compute_window_features(s,e,equity, assets_db, news_df=news_df); f['label'] = 2; rows.append(f)
    
    labeled_intervals = [(r['window_start'], r['window_end']) for r in rows if 'window_start' in r and 'window_end' in r]
    all_times = equity.index
    neutral_count_target = max(1, len(rows)//2)
    neutral_added = 0; idx = 0
    window_seconds = int(np.median([( (r['window_end'] - r['window_start']).total_seconds() ) for r in rows if r.get('window_end') is not None] or [7*24*3600]))
    step = max(1, len(all_times)//1000)
    while neutral_added < neutral_count_target and idx < len(all_times)-1:
        start = all_times[idx]; end = start + pd.Timedelta(seconds=window_seconds)
        if end not in all_times:
            idx += step; continue
        overlap = False
        for a,b in labeled_intervals:
            if not (end < a or start > b):
                overlap = True; break
        if not overlap:
            f = compute_window_features(start, end, equity, assets_db, news_df=news_df)
            f['label'] = 0; rows.append(f); labeled_intervals.append((start,end)); neutral_added += 1
        idx += step
    
    df = pd.DataFrame(rows)
    if 'window_start' in df.columns:
        df.drop(columns=['window_start','window_end'], inplace=True, errors='ignore')
    df = df.loc[:, df.notna().any(axis=0)]
    df.to_csv(OUTPUT_DIR / 'ml_raw_dataset.csv', index=False)
    log_info("ML dataset saved to ml_raw_dataset.csv, rows=" + str(len(df)))
    
    label_counts = df['label'].value_counts()
    log_info(f"Label distribution: {label_counts.to_dict()}")
    return df

# ---------------- Train and report ----------------
def train_and_report(df, target_col='label'):
    df = df.copy().dropna(axis=1, how='all')
    if df.empty or target_col not in df.columns:
        log_warn("Dataset vuoto o mancante label. Skip training.")
        return {}
    y = df[target_col].astype(int); X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number]).fillna(0.0)
    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced')
    lr = LogisticRegression(max_iter=2000)
    models = {'RandomForest': rf, 'LogisticRegression': lr}
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss')
    results = {}
    for name, model in models.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        try:
            cv_res = cross_validate(pipe, X, y, cv=tscv, scoring=['accuracy'], return_train_score=False)
            results[name] = {k: float(np.mean(v)) for k,v in cv_res.items()}
            pipe.fit(X, y)
            ypred = pipe.predict(X)
            report = classification_report(y, ypred, output_dict=True)
            results[name]['train_report'] = report
            
            try:
                clf = pipe.named_steps['clf']
                fi = None
                if hasattr(clf, 'feature_importances_'):
                    fi = clf.feature_importances_
                elif hasattr(clf, 'coef_'):
                    fi = np.mean(np.abs(clf.coef_), axis=0)
                if fi is not None:
                    fi_series = pd.Series(fi, index=X.columns).sort_values(ascending=False)
                    fi_series.to_csv(OUTPUT_DIR / f'feature_importance_{name}.csv')
                    plt.figure(figsize=(10,6)); fi_series.head(30).plot(kind='bar'); plt.title(f'Feature importance {name}'); plt.tight_layout(); plt.savefig(OUTPUT_DIR / f'feature_importance_{name}.png'); plt.close()
            except Exception as e:
                log_warn(f"Feature importance failed for {name}: {e}")
            
            if SHAP_AVAILABLE and name == 'RandomForest':
                try:
                    explainer = shap.TreeExplainer(pipe.named_steps['clf'])
                    shap_values = explainer.shap_values(X)
                    shap.summary_plot(shap_values, X, show=False)
                    plt.tight_layout(); plt.savefig(OUTPUT_DIR / f'shap_summary_{name}.png'); plt.close()
                except Exception as e:
                    log_warn(f"SHAP failed: {e}")
        except Exception as e:
            results[name] = {'error': str(e)}
    
    try:
        best_name = max(results.keys(), key=lambda n: results[n].get('test_accuracy', -1))
    except Exception:
        best_name = None
    if best_name:
        try:
            best_pipe = Pipeline([('scaler', StandardScaler()), ('clf', models[best_name])]); best_pipe.fit(X,y)
            ypred = best_pipe.predict(X); cm = confusion_matrix(y, ypred); plt.figure(figsize=(6,5)); sns.heatmap(cm, annot=True, fmt='d'); plt.title(f'Confusion Matrix - {best_name}'); plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'confusion_matrix_train.png'); plt.close()
            try:
                import joblib; joblib.dump(best_pipe, OUTPUT_DIR / 'best_model.joblib')
            except Exception:
                pass
        except Exception:
            pass
    try:
        pd.DataFrame(results).to_json(OUTPUT_DIR / 'ml_results_summary.json')
    except Exception:
        pass
    log_info("Training completed. Results saved in " + str(OUTPUT_DIR))
    return results

# ---------------- Plots ----------------
def plot_equity(merged, output_dir=OUTPUT_DIR, label='Equity'):
    plt.figure(figsize=(14,7))
    plt.plot(merged.index, merged['balance'], lw=1, label='Balance')
    plt.plot(merged.index, merged['equity_close'], lw=1, label=f'{label} CLOSE')
    plt.plot(merged.index, merged['equity_low'], lw=1, label=f'{label} LOW')
    peak = merged['equity_low'].cummax(); dd = merged['equity_low'] - peak
    try:
        t = dd.idxmin(); v = merged.loc[t,'equity_low']; plt.scatter([t], [v], color='black', s=80, label='Max DD')
    except Exception:
        pass
    plt.legend(); plt.title(f'{label} curve high-res'); plt.tight_layout(); plt.savefig(output_dir / f'global_{label.lower().replace(" ", "_")}_highres.png'); plt.close()
    log_info(f"Saved {label} plot.")

def plot_equity_comparison(merged_original, merged_adjusted, output_dir=OUTPUT_DIR, label_original='Original', label_adjusted='Adjusted'):
    plt.figure(figsize=(14,7))
    plt.plot(merged_original.index, merged_original['equity_close'], label=f'{label_original} Equity')
    plt.plot(merged_adjusted.index, merged_adjusted['equity_close'], label=f'{label_adjusted} Equity')
    plt.legend()
    plt.title(f'{label_original} vs {label_adjusted} Equity Comparison')
    plt.tight_layout()
    plt.savefig(output_dir / f'equity_comparison_{label_adjusted.lower().replace(" ", "_")}.png')
    plt.close()
    log_info(f"Saved equity comparison plot for {label_adjusted}.")

def plot_equity_vs_assets(merged, assets_db, output_dir=OUTPUT_DIR):
    try:
        equity_daily = merged['equity_close'].resample('D').last().dropna()
    except Exception:
        equity_daily = merged['equity_close'].asfreq('D').fillna(method='ffill').dropna()
    if equity_daily.empty:
        log_warn("equity_daily empty, skip equity vs assets plot")
        return
    norm_equity = equity_daily / equity_daily.iloc[0] * 100.0
    plt.figure(figsize=(14,8))
    plt.plot(norm_equity.index, norm_equity, lw=2, label='Equity (base100)')
    for sym, df in assets_db.items():
        try:
            s = df['CLOSE'].resample('D').last().reindex(equity_daily.index, method='ffill').dropna()
            if s.empty: continue
            norm = s / s.iloc[0] * 100.0
            plt.plot(norm.index, norm, lw=1, alpha=0.8, label=sym)
        except Exception:
            continue
    plt.title('Equity vs Assets (base100)'); plt.legend(loc='upper left'); plt.tight_layout(); plt.savefig(output_dir / 'equity_vs_assets_base100.png'); plt.close()
    
    if 'VIX' in assets_db:
        try:
            vix = assets_db['VIX']['CLOSE'].resample('D').last().reindex(equity_daily.index, method='ffill')
            fig, ax1 = plt.subplots(figsize=(14,8))
            ax1.plot(norm_equity.index, norm_equity, lw=2, label='Equity (base100)')
            ax2 = ax1.twinx(); ax2.plot(vix.index, vix, lw=1, linestyle='--', label='VIX'); ax1.set_title('Equity vs VIX'); fig.tight_layout(); fig.savefig(output_dir / 'equity_vs_vix.png'); plt.close()
        except Exception as e:
            log_warn(f"Error plotting VIX: {e}")
    log_info("Saved equity vs assets plots.")

def correlation_heatmap(df_features, output_dir=OUTPUT_DIR):
    try:
        df_features = df_features.select_dtypes(include=[np.number])
        corr = df_features.corr()
        plt.figure(figsize=(12,10)); sns.heatmap(corr, cmap='coolwarm', center=0); plt.title('Correlation Heatmap'); plt.tight_layout(); plt.savefig(output_dir / 'features_correlation_heatmap.png'); plt.close()
        log_info("Saved features correlation heatmap.")
    except Exception as e:
        log_warn(f"Heatmap failed: {e}")

# ---------------- Recommendations (FIXED) ----------------
def generate_recommendations(df_ml, merged, assets_db, extra_recs=None, output_dir=OUTPUT_DIR):
    recs = []
    if df_ml is None or df_ml.empty:
        recs.append("ML dataset empty or missing")
    else:
        means = df_ml.groupby('label').mean(numeric_only=True)
        stds = df_ml.groupby('label').std(numeric_only=True)
        means.to_csv(output_dir / 'feature_means_by_label.csv'); stds.to_csv(output_dir / 'feature_std_by_label.csv')
        recs.append("Feature means/std saved to CSV")
        
        if 0 in means.index and 1 in means.index:
            diff = (means.loc[1] - means.loc[0]).abs().sort_values(ascending=False)
            recs.append("Top features that separate drawdown vs neutral: " + ", ".join(diff.head(12).index.tolist()))
        else:
            recs.append("Top features (general): " + ", ".join(means.abs().sum().sort_values(ascending=False).head(12).index.tolist()))
        
        # FIXED: Handle vix_mean comparison
        if 'vix_mean' in df_ml.columns:
            vix_draw = df_ml[df_ml['label']==1]['vix_mean'].dropna()
            vix_neu = df_ml[df_ml['label']==0]['vix_mean'].dropna()
            if len(vix_draw) > 0 and len(vix_neu) > 0:  # Check length instead of empty
                vix_draw_mean = vix_draw.mean()
                vix_neu_mean = vix_neu.mean()
                if vix_draw_mean > vix_neu_mean:
                    thr = float(np.percentile(df_ml['vix_mean'].dropna(), 75))
                    recs.append(f"Rule suggestion: reduce size when VIX > {thr:.2f} (75th percentile).")
                else:
                    thr = float(np.percentile(df_ml['vix_mean'].dropna(), 25))
                    recs.append(f"Rule suggestion: consider increasing size when VIX < {thr:.2f} (25th percentile).")
    
    try:
        eq_close = pd.to_numeric(merged['equity_close'], errors='coerce').dropna()
        if not eq_close.empty:
            peak = eq_close.cummax()
            dd = eq_close - peak
            max_dd = dd.min()
            time_dd = dd.idxmin()
            recs.append(f"Max drawdown observed (close): {max_dd:.2f} at {time_dd}")
    except Exception as e:
        recs.append(f"Error computing max drawdown: {e}")
    
    if extra_recs:
        for r in extra_recs: recs.append(str(r))
    recs.append("To diversify during high VIX/equity crises, consider adding algorithms on bonds (e.g., US Treasuries), safe-haven currencies (e.g., USDJPY long USD), or increase weight on gold, defensive sectors like utilities, healthcare, consumer staples.")
    
    out_file = output_dir / 'portfolio_recommendations.txt'
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("=== Portfolio Recommendations ===\n\n")
        for r in recs:
            f.write(f"- {r}\n")
    log_info("Recommendations saved to " + str(out_file))
    return recs

# ---------------- News analysis scaffold ----------------
def analyze_news_impact(df_trades, merged, assets_db, news_df, events=['NFP','CPI','FOMC'], windows_minutes=(10,60)):
    results = {}
    if news_df is None or news_df.empty:
        log_warn("No news data provided to analyze_news_impact.")
        return results
    news_df['time'] = pd.to_datetime(news_df['time'], errors='coerce')
    for ev in events:
        ev_df = news_df[news_df['event_type']==ev]
        stats = []
        for _, row in ev_df.iterrows():
            t = row['time']
            for w in windows_minutes:
                start = t - pd.Timedelta(minutes=w); end = t + pd.Timedelta(minutes=w)
                try:
                    eq_start = merged['equity_close'].asof(start)
                    eq_end = merged['equity_close'].asof(end)
                    pnl = float(eq_end - eq_start) if (pd.notna(eq_start) and pd.notna(eq_end)) else np.nan
                except Exception:
                    pnl = np.nan
                stats.append({'event': ev, 'time': t, 'window_min': w, 'pnl': pnl})
        results[ev] = pd.DataFrame(stats)
        if not results[ev].empty:
            results[ev].to_csv(OUTPUT_DIR / f'news_impact_{ev}.csv', index=False)
    log_info("News impact analysis finished (scaffold).")
    return results

# ---------------- Simulate VIX adjusted equity ----------------
def simulate_vix_adjusted_equity(df_trades, m5_db, vix_df, thresholds=[25,30], reductions=[0.7,0.5], initial_balance=INITIAL_BALANCE):
    """Simulate equity with size reduction based on VIX at trade open time."""
    log_info(f"Simulating VIX-adjusted equity with thresholds={thresholds}, reductions={reductions}...")
    df = df_trades.copy()
    df['Open time'] = pd.to_datetime(df['Open time'], dayfirst=True, errors='coerce')
    for idx, row in df.iterrows():
        open_time = row['Open time']
        vix = vix_df['CLOSE'].asof(open_time)
        try:
            vix = float(vix)
        except (ValueError, TypeError):
            continue
        if np.isnan(vix):
            continue
        f = 1.0
        for t, r in sorted(zip(thresholds, reductions), reverse=True):
            if vix >= t:
                f = r
                break
        df.at[idx, 'Size'] = row['Size'] * f
        df.at[idx, 'Profit/Loss (Global)'] = row['Profit/Loss (Global)'] * f
    merged_adjusted = compute_equity_highres(df, m5_db, initial_balance=initial_balance)
    merged_adjusted.to_csv(OUTPUT_DIR / 'equity_adjusted_series.csv')
    log_info("Adjusted equity series saved.")
    return merged_adjusted

# ---------------- Optimize VIX thresholds (SIMPLIFIED) ----------------
def optimize_vix_thresholds(df_trades, m5_db, vix_df, initial_balance=INITIAL_BALANCE):
    """Grid search for best VIX thresholds - SIMPLIFIED VERSION."""
    log_info("Optimizing VIX thresholds (simplified grid search)...")
    # Reduced grid: only test a few meaningful combinations
    test_configs = [
        ([25], [0.7]),       # Single threshold
        ([30], [0.5]),       # Single threshold
        ([25, 30], [0.7, 0.5])  # Double threshold
    ]
    
    best_dd = -np.inf
    best_params = None
    
    for i, (thresholds, reductions) in enumerate(test_configs):
        log_info(f"Testing config {i+1}/{len(test_configs)}: thresholds={thresholds}, reductions={reductions}")
        try:
            merged = simulate_vix_adjusted_equity(df_trades, m5_db, vix_df, thresholds=thresholds, reductions=reductions, initial_balance=initial_balance)
            peak = merged['equity_close'].cummax()
            dd = (merged['equity_close'] - peak).min()
            log_info(f"  -> Max DD: {dd:.2f}")
            if dd > best_dd:
                best_dd = dd
                best_params = (thresholds, reductions)
        except Exception as e:
            log_warn(f"Error testing config {i+1}: {e}")
            continue
    
    if best_params:
        log_info(f"Best params: thresholds={best_params[0]}, reductions={best_params[1]} with max DD={best_dd:.2f}")
    else:
        log_warn("No valid configuration found during optimization")
    return best_params

# ---------------- Main pipeline ----------------
def run_full_pipeline(report_path=REPORT_PATH, news_path=None):
    log_info(f"Starting pipeline. Report: {report_path}")
    if not Path(report_path).exists():
        log_warn("Report file not found: " + str(report_path))
        return
    df_trades = pd.read_csv(report_path)
    df_trades.columns = [c.strip() for c in df_trades.columns]
    log_info(f"Trades loaded: {len(df_trades)} rows")
    symbols = df_trades['Symbol'].unique()
    log_info(f"Unique symbols in trades: {len(symbols)}")
    
    assets_db = {}
    loaded = 0
    for s in symbols:
        try:
            d = load_price_data(s, timeframe_preference=('M5','D1'))
            if d is not None:
                assets_db[s] = d; loaded += 1
        except Exception as e:
            log_warn(f"Error loading {s}: {e}")
    vix_df = None
    if 'VIX' not in assets_db:
        vix_df = load_price_data('VIX', timeframe_preference=('D1','M5'))
        if vix_df is not None:
            assets_db['VIX'] = vix_df; loaded += 1
    log_info(f"Loaded asset datasets: {loaded}")
    
    merged = compute_equity_highres(df_trades, assets_db)
    merged['equity_close'].to_csv(OUTPUT_DIR / 'equity_close_series.csv')
    log_info("Equity series saved.")
    
    try:
        extra_recs = analyze_drawdowns_and_vix(merged, assets_db, thresholds=(2000,1000))
    except Exception as e:
        log_warn("analyze_drawdowns_and_vix failed: " + str(e))
        extra_recs = []
    
    df_ml = build_ml_dataset(merged, df_trades, assets_to_load=('US500.pro','US100.pro','BTC','VIX','GOLD.pro'), news_path=news_path)
    if df_ml is None or df_ml.empty:
        log_warn("ML dataset empty.")
    else:
        log_info(f"ML dataset rows: {len(df_ml)}, cols: {len(df_ml.columns)}")
        try:
            correlation_heatmap(df_ml.drop(columns=['label'], errors='ignore'))
        except Exception as e:
            log_warn("correlation heatmap error: " + str(e))
    
    results = train_and_report(df_ml, target_col='label')
    
    try:
        plot_equity(merged)
        plot_equity_vs_assets(merged, assets_db)
    except Exception as e:
        log_warn("plotting error: " + str(e))
    
    news_df = None
    if news_path and Path(news_path).exists():
        try:
            news_df = pd.read_csv(news_path); news_df['time'] = pd.to_datetime(news_df['time'], errors='coerce')
            analyze_news_impact(df_trades, merged, assets_db, news_df, events=['NFP','CPI','FOMC'], windows_minutes=(10,60))
        except Exception as e:
            log_warn("news analysis failed: " + str(e))
    
    try:
        recs = generate_recommendations(df_ml, merged, assets_db, extra_recs=extra_recs)
        log_info("Recommendations generated. Count: " + str(len(recs)))
    except Exception as e:
        log_warn("generate_recommendations failed: " + str(e))
    
    if vix_df is not None:
        log_info("Running VIX-based simulations...")
        merged_adjusted = simulate_vix_adjusted_equity(df_trades, assets_db, vix_df)
        plot_equity(merged_adjusted, label='Adjusted Equity')
        plot_equity_comparison(merged, merged_adjusted)
        
        # Simplified optimization
        best_params = optimize_vix_thresholds(df_trades, assets_db, vix_df)
        if best_params:
            merged_best = simulate_vix_adjusted_equity(df_trades, assets_db, vix_df, thresholds=list(best_params[0]), reductions=list(best_params[1]))
            plot_equity(merged_best, label='Optimized Adjusted Equity')
            plot_equity_comparison(merged, merged_best, label_adjusted='Optimized Adjusted')
    else:
        log_warn("VIX data not available for simulation.")
    
    log_info("Pipeline finished. Check outputs in " + str(OUTPUT_DIR))
    return results

if __name__ == "__main__":
    run_full_pipeline()