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

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[WARN] requests not installed. Install with 'pip install requests' to fetch VIX from CBOE if needed.")

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
    """Carica file {symbol}_{TF}_2020_2025.csv.
    Supporta file con colonne DATE, TIME oppure time, oppure primo colonna datetime.
    Ritorna DataFrame con colonne OPEN,HIGH,LOW,CLOSE e DatetimeIndex.
    """
    for tf in timeframe_preference:
        path = DATASETS_FOLDER / f"{symbol}_{tf}_2020_2025.csv"
        if not path.exists():
            continue
        try:
            # try common separators: tab or comma
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
                # assume first column is datetime-like
                try:
                    df.index = pd.to_datetime(df.iloc[:,0], errors='coerce')
                except Exception:
                    pass
            
            # rename open/high/low/close
            cols_map = {}
            for c in df.columns:
                cl = c.lower()
                if cl in ['open', 'high', 'low', 'close']:
                    cols_map[c] = c.upper()
                elif cl == 'price' and 'OPEN' not in cols_map:
                    cols_map[c] = 'CLOSE'  # fallback
            if cols_map:
                df.rename(columns=cols_map, inplace=True)
            
            # Check for required columns
            required = ['OPEN','HIGH','LOW','CLOSE']
            missing = [col for col in required if col not in df.columns]
            if missing:
                # Try to find alternative column names
                for col in required:
                    for df_col in df.columns:
                        if col.lower() in df_col.lower():
                            df.rename(columns={df_col: col}, inplace=True)
                            break
                missing = [col for col in required if col not in df.columns]
                if missing:
                    log_warn(f"Missing columns {missing} in {path}")
                    continue
            
            df = df[['OPEN','HIGH','LOW','CLOSE']].astype(float)
            df = df[~df.index.isna()].sort_index()
            log_info(f"Loaded {symbol} {tf} rows={len(df)}")
            return df
        except Exception as e:
            log_warn(f"Failed to load {path}: {e}")
            continue
    
    # If not found locally and symbol is VIX or BTC, fetch from yfinance if available
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
                # Save to local file
                path = DATASETS_FOLDER / f"{symbol}_{tf}_2020_2025.csv"
                df.to_csv(path)
                return df
            except Exception as e:
                log_warn(f"Failed to fetch {symbol} {tf} from yfinance: {e}")
                continue
    
    log_warn(f"No dataset found for {symbol} (checked {timeframe_preference})")
    return None

# ---------------- Equity high-res ----------------
def compute_equity_highres(df_trades, m5_db, initial_balance=INITIAL_BALANCE, freq='5min'):
    """Ricostruisce equity ad alta risoluzione a partire da trades e database m5_db dict(symbol->df)."""
    log_info("Computing high-resolution equity...")
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
    
    # Build balance changes from realized P/L
    bc = df[['Close time','Profit/Loss (Global)']].copy().sort_values('Close time')
    bc['Cumulative'] = initial_balance + bc['Profit/Loss (Global)'].cumsum()
    merged = pd.merge_asof(high_res, bc.rename(columns={'Close time':'time'}), on='time', direction='backward')
    merged.set_index('time', inplace=True)
    merged['balance'] = merged['Cumulative'].fillna(initial_balance)
    merged['equity_close'] = merged['balance'].copy()
    merged['equity_low'] = merged['balance'].copy()
    
    # simulate unrealized P/L during each trade using m5_db if available
    for idx, tr in df.iterrows():
        sym = tr['Symbol']
        if sym not in m5_db:
            continue
        mask = (merged.index >= tr['Open time']) & (merged.index < tr['Close time'])
        if not mask.any():
            continue
        p = m5_db[sym].reindex(merged.index[mask], method='ffill')
        if p.empty:
            continue
        # multiplier
        mult = float(tr.get('Size',1)) * CONTRACT_SIZE.get(sym, 100)
        entry = float(tr.get('Open price', np.nan))
        is_buy = 'Buy' in str(tr.get('Type',''))
        if is_buy:
            pl_close = (p['CLOSE'] - entry) * mult
            pl_low = (p['LOW'] - entry) * mult
        else:
            pl_close = (entry - p['CLOSE']) * mult
            pl_low = (entry - p['HIGH']) * mult
        merged.loc[mask, 'equity_close'] += pl_close.values
        merged.loc[mask, 'equity_low'] += pl_low.values
    
    log_info("Equity computed.")
    return merged

# ---------------- Drawdown episodes ----------------
def detect_all_drawdown_episodes(equity_series):
    """Detect all peak->trough->recovery episodes. Returns list of dicts."""
    s = equity_series.dropna().copy()
    if s.empty:
        return []
    peak = s.cummax()
    dd = s - peak
    in_dd = dd < 0
    diff = in_dd.astype(int).diff().fillna(0)
    starts = s.index[diff == 1].tolist()
    ends = s.index[diff == -1].tolist()
    
    # align
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
    
    # fallback to single trough if none detected
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
    episodes = detect_all_drawdown_episodes(equity)
    rows = []
    for e in episodes:
        start = e['peak_time']
        trough = e['trough_time']
        recovery = e['recovery_time']
        dd_amt = -float(e['drawdown_amount']) # positive magnitude
        duration_sec = e['duration'].total_seconds() if pd.notna(e['duration']) else np.nan
        
        # VIX features
        vix_mean = np.nan; vix_max = np.nan; vix_reg = 'UNKNOWN'
        if 'VIX' in assets_db:
            try:
                vix_series = assets_db['VIX']['CLOSE'].reindex(pd.date_range(start=start, end=recovery, freq='D'), method='ffill').dropna()
                if not vix_series.empty:
                    vix_mean = float(vix_series.mean()); vix_max = float(vix_series.max()); vix_reg = vix_regime(vix_mean)
            except Exception:
                pass
        
        # correlations
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

# ---------------- Top best periods (e.g. top weeks) ----------------
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
    
    # assets
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
    
    # VIX summary
    if 'VIX' in assets_db:
        try:
            v = assets_db['VIX']['CLOSE'].reindex(pd.date_range(start=start, end=end, freq='D'), method='ffill').dropna()
            features['vix_mean'] = v.mean() if not v.empty else np.nan
            features['vix_max'] = v.max() if not v.empty else np.nan
            features['vix_regime'] = vix_regime(features['vix_mean']) if not math.isnan(features.get('vix_mean', np.nan)) else 'UNKNOWN'
        except Exception:
            features['vix_mean'] = np.nan; features['vix_max'] = np.nan; features['vix_regime'] = 'UNKNOWN'
    
    # news
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
    
    # drawdown windows
    dd_eps = detect_all_drawdown_episodes(equity)
    dd_windows = []
    for e in dd_eps:
        start = e.get('peak_time'); end = e.get('recovery_time') if pd.notna(e.get('recovery_time')) else e.get('trough_time')
        dd_windows.append((start,end))
    
    # best periods
    best_periods = top_k_best_periods(equity, freq='W', top_k=top_n_good)
    best_windows = [(p['start'], p['end']) for p in best_periods]
    
    # news
    news_df = None
    if news_path and Path(news_path).exists():
        news_df = pd.read_csv(news_path); news_df['time'] = pd.to_datetime(news_df['time'], errors='coerce')
    
    # assemble rows
    rows = []
    for s,e in dd_windows:
        f = compute_window_features(s,e,equity, assets_db, news_df=news_df); f['label'] = 1; rows.append(f)
    for s,e in best_windows:
        f = compute_window_features(s,e,equity, assets_db, news_df=news_df); f['label'] = 2; rows.append(f)
    
    # neutral windows: generate similar count of neutral windows by scanning timeline
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
    return df

# ========== FUNZIONI DI VERIFICA ==========

def verify_drawdown_analysis(df_trades, assets_db, merged):
    """Verifica indipendente dei drawdown e delle condizioni di mercato"""
    
    log_info("=== VERIFICA DRAWDOWN ANALYSIS ===")
    
    # 1. Analisi dei periodi di crisi importanti
    crisis_periods = {
        'COVID-19': ('2020-02-15', '2020-04-01'),
        'Inflation 2022': ('2022-01-01', '2022-12-31'),
        'Ukraine War': ('2022-02-15', '2022-03-31')
    }
    
    crisis_results = {}
    for crisis_name, (start, end) in crisis_periods.items():
        # Estrai trades nel periodo
        mask = (df_trades['Close time'] >= start) & (df_trades['Open time'] <= end)
        if 'Close time' in df_trades.columns and 'Open time' in df_trades.columns:
            crisis_trades = df_trades[mask]
        else:
            crisis_trades = df_trades
        
        # Calcola PnL durante la crisi
        if 'Profit/Loss (Global)' in crisis_trades.columns:
            crisis_pnl = crisis_trades['Profit/Loss (Global)'].sum()
        else:
            crisis_pnl = 0
            
        # VIX medio nel periodo
        avg_vix = None
        if 'VIX' in assets_db:
            try:
                vix_series = assets_db['VIX']['CLOSE'].loc[start:end]
                avg_vix = float(vix_series.mean()) if not vix_series.empty else None
            except:
                pass
                
        # Performance S&P500
        spx_return = None
        if 'US500.pro' in assets_db:
            try:
                spx = assets_db['US500.pro']['CLOSE'].loc[start:end]
                if len(spx) > 1:
                    spx_return = float((spx.iloc[-1] / spx.iloc[0] - 1) * 100)
            except:
                pass
        
        crisis_results[crisis_name] = {
            'pnl_total': crisis_pnl,
            'avg_vix': avg_vix,
            'spx_return_pct': spx_return,
            'n_trades': len(crisis_trades),
            'win_rate': float((crisis_trades['Profit/Loss (Global)'] > 0).mean()) if len(crisis_trades) > 0 else None
        }
    
    # 2. Verifica distribuzione posizioni (long/short)
    if 'Type' in df_trades.columns:
        position_types = df_trades['Type'].value_counts()
        log_info(f"Distribuzione posizioni: {position_types.to_dict()}")
        
        # Analisi performance per tipo di posizione
        buy_trades = df_trades[df_trades['Type'].str.contains('Buy', na=False, case=False)]
        sell_trades = df_trades[df_trades['Type'].str.contains('Sell', na=False, case=False)]
        
        buy_pnl = buy_trades['Profit/Loss (Global)'].sum() if 'Profit/Loss (Global)' in buy_trades.columns else 0
        sell_pnl = sell_trades['Profit/Loss (Global)'].sum() if 'Profit/Loss (Global)' in sell_trades.columns else 0
        
        log_info(f"PNL totale Buy: {buy_pnl:.2f}")
        log_info(f"PNL totale Sell: {sell_pnl:.2f}")
        
        if len(buy_trades) > 0:
            buy_win_rate = (buy_trades['Profit/Loss (Global)'] > 0).mean()
            log_info(f"Win rate Buy: {buy_win_rate:.2%}")
        if len(sell_trades) > 0:
            sell_win_rate = (sell_trades['Profit/Loss (Global)'] > 0).mean()
            log_info(f"Win rate Sell: {sell_win_rate:.2%}")
    
    # 3. Verifica correlazione direzionale
    try:
        equity_daily = merged['equity_close'].resample('D').last().dropna()
        equity_returns = equity_daily.pct_change().dropna()
        
        if 'US500.pro' in assets_db:
            spx_daily = assets_db['US500.pro']['CLOSE'].resample('D').last().reindex(equity_daily.index, method='ffill').pct_change().dropna()
            
            # Allinea le date
            aligned = pd.concat([equity_returns, spx_daily], axis=1).dropna()
            aligned.columns = ['equity_returns', 'spx_returns']
            
            # Calcola correlazione
            correlation = aligned['equity_returns'].corr(aligned['spx_returns'])
            log_info(f"Correlazione totale equity vs S&P500: {correlation:.3f}")
            
            # Correlazione condizionale (quando VIX è alto)
            if 'VIX' in assets_db:
                vix_daily = assets_db['VIX']['CLOSE'].resample('D').last().reindex(aligned.index, method='ffill').dropna()
                aligned_vix = pd.concat([aligned, vix_daily], axis=1).dropna()
                aligned_vix.columns = ['equity_returns', 'spx_returns', 'vix']
                
                # Separa per regime VIX
                high_vix = aligned_vix[aligned_vix['vix'] > 25]
                low_vix = aligned_vix[aligned_vix['vix'] < 15]
                
                if len(high_vix) > 5:
                    corr_high = high_vix['equity_returns'].corr(high_vix['spx_returns'])
                    log_info(f"Correlazione quando VIX > 25: {corr_high:.3f} (n={len(high_vix)})")
                
                if len(low_vix) > 5:
                    corr_low = low_vix['equity_returns'].corr(low_vix['spx_returns'])
                    log_info(f"Correlazione quando VIX < 15: {corr_low:.3f} (n={len(low_vix)})")
    except Exception as e:
        log_warn(f"Errore nella verifica correlazione: {e}")
    
    return crisis_results

def check_drawdown_veracity(merged, thresholds=[2000, 1000]):
    """Verifica manuale dei drawdown per confermare l'analisi"""
    
    log_info("=== VERIFICA MANUALE DRAWDOWNS ===")
    
    equity = merged['equity_close']
    peak = equity.cummax()
    drawdown_pct = (equity - peak) / peak * 100  # in percentuale
    drawdown_abs = equity - peak  # in valore assoluto
    
    for threshold in thresholds:
        # Trova periodi con drawdown > threshold in valore assoluto
        dd_periods = (drawdown_abs <= -threshold)
        
        if dd_periods.any():
            # Trova inizio e fine di ogni periodo di drawdown
            diff = dd_periods.astype(int).diff()
            starts = drawdown_abs[diff == 1].index
            ends = drawdown_abs[diff == -1].index
            
            log_info(f"DRAWDOWN >= {threshold} EUR:")
            log_info(f"  Numero di periodi: {len(starts)}")
            
            for i, (start, end) in enumerate(zip(starts[:3], ends[:3])):
                if start in drawdown_pct.index and end in drawdown_pct.index:
                    dd_min = drawdown_pct.loc[start:end].min()
                    log_info(f"  Periodo {i+1}: {start} a {end}, DD min: {dd_min:.2f}%")
                else:
                    log_info(f"  Periodo {i+1}: {start} a {end}")
        else:
            log_info(f"Nessun drawdown >= {threshold} EUR trovato")

# ========== RISK ENGINE CON VIX-BASED POSITION SIZING ==========

class RiskEngineVIX:
    """Engine di gestione rischio basato su VIX"""
    
    def __init__(self, assets_db, initial_size=1.0):
        self.assets_db = assets_db
        self.initial_size = initial_size
        self.current_size = initial_size
        self.vix_history = []
        self.position_adjustments = []
        
        # Soglie configurabili
        self.thresholds = {
            'reduce_30pct': 25,  # VIX >= 25: riduci 30%
            'reduce_50pct': 30,  # VIX >= 30: riduci 50%
            'resume_75pct': 20,  # VIX < 20: ripristina 75%
            'resume_full': 15    # VIX < 15: ripristina 100%
        }
    
    def get_vix_at_time(self, timestamp):
        """Ottiene valore VIX al timestamp più vicino"""
        if 'VIX' not in self.assets_db:
            return None
        
        vix_df = self.assets_db['VIX']
        
        # Controlla se abbiamo dati VIX
        if 'CLOSE' not in vix_df.columns:
            return None
        
        try:
            vix_series = vix_df['CLOSE']
            
            # Se il timestamp è nell'indice
            if timestamp in vix_series.index:
                return float(vix_series.loc[timestamp])
            
            # Trova l'ultimo timestamp disponibile prima del tempo dato
            available_times = vix_series.index[vix_series.index <= timestamp]
            if len(available_times) > 0:
                last_time = available_times[-1]
                return float(vix_series.loc[last_time])
        except Exception as e:
            log_warn(f"Errore ottenendo VIX al tempo {timestamp}: {e}")
        
        return None
    
    def calculate_position_multiplier(self, timestamp):
        """Calcola moltiplicatore di posizione basato su VIX"""
        vix = self.get_vix_at_time(timestamp)
        
        if vix is None:
            return self.current_size  # Mantieni dimensione corrente
        
        # Aggiorna storia VIX
        self.vix_history.append({'timestamp': timestamp, 'vix': vix, 'size_before': self.current_size})
        
        # Applica regole
        new_size = self.current_size
        
        if vix >= self.thresholds['reduce_50pct']:
            new_size = self.initial_size * 0.5  # Riduci 50%
        elif vix >= self.thresholds['reduce_30pct']:
            new_size = self.initial_size * 0.7  # Riduci 30%
        elif vix <= self.thresholds['resume_full']:
            new_size = self.initial_size  # Ripristina 100%
        elif vix <= self.thresholds['resume_75pct']:
            new_size = self.initial_size * 0.75  # Ripristina 75%
        
        # Registra cambiamento se diverso
        if abs(new_size - self.current_size) > 0.01:
            self.position_adjustments.append({
                'timestamp': timestamp,
                'vix': vix,
                'old_size': self.current_size,
                'new_size': new_size
            })
            self.current_size = new_size
        
        return self.current_size
    
    def simulate_equity_with_risk_management(self, df_trades, initial_balance=INITIAL_BALANCE):
        """Simula equity curve con gestione rischio VIX"""
        
        log_info("Simulazione equity con Risk Engine VIX...")
        
        # Crea copia dei trades
        trades = df_trades.copy()
        
        # Assicurati che le colonne timestamp siano nel formato corretto
        required_cols = ['Open time', 'Close time', 'Profit/Loss (Global)']
        for col in required_cols:
            if col not in trades.columns:
                log_warn(f"Colonna {col} mancante nei trades")
                return None, None, []
        
        # Converti timestamp
        trades['Open time'] = pd.to_datetime(trades['Open time'], dayfirst=True, errors='coerce')
        trades['Close time'] = pd.to_datetime(trades['Close time'], dayfirst=True, errors='coerce')
        
        # Rimuovi righe con timestamp non validi
        trades = trades.dropna(subset=['Open time', 'Close time'])
        
        # Ordina cronologicamente
        trades = trades.sort_values('Open time').reset_index(drop=True)
        
        # Simula trade by trade con size adjustment
        balance = initial_balance
        equity_series = []
        
        for idx, trade in trades.iterrows():
            open_time = trade['Open time']
            close_time = trade['Close time']
            
            # Determina size multiplier al momento dell'apertura
            size_multiplier = self.calculate_position_multiplier(open_time)
            
            # Aggiusta il PnL in base al moltiplicatore
            original_pnl = float(trade['Profit/Loss (Global)'])
            adjusted_pnl = original_pnl * size_multiplier
            
            # Aggiorna bilancio
            balance += adjusted_pnl
            
            # Registra equity al close time
            equity_series.append({
                'timestamp': close_time,
                'balance': balance,
                'equity': balance,
                'vix_at_open': self.get_vix_at_time(open_time),
                'size_multiplier': size_multiplier,
                'original_pnl': original_pnl,
                'adjusted_pnl': adjusted_pnl
            })
        
        # Crea DataFrame dell'equity
        if not equity_series:
            log_warn("Nessun trade processato nella simulazione")
            return None, None, []
            
        equity_df = pd.DataFrame(equity_series)
        equity_df.set_index('timestamp', inplace=True)
        
        # Rimuovi duplicati nell'indice (se due trade chiudono allo stesso timestamp)
        equity_df = equity_df[~equity_df.index.duplicated(keep='last')]
        equity_df = equity_df.sort_index()
        
        # Crea serie temporale continua
        if len(equity_df) > 1:
            full_index = pd.date_range(start=equity_df.index.min(), end=equity_df.index.max(), freq='5min')
            # Usa asof invece di reindex per evitare duplicati
            full_equity = pd.Series(index=full_index, dtype=float)
            for ts in full_index:
                # Trova l'ultimo valore di equity prima o a questo timestamp
                mask = equity_df.index <= ts
                if mask.any():
                    last_idx = equity_df.index[mask][-1]
                    full_equity.loc[ts] = equity_df.loc[last_idx, 'equity']
        else:
            full_equity = equity_df['equity']
        
        log_info(f"Simulazione completata. Adjustments applicati: {len(self.position_adjustments)}")
        
        return full_equity, equity_df, self.position_adjustments

# ========== FUNZIONI AUSILIARIE PER METRICHE ==========

def calculate_max_drawdown(equity_series):
    """Calcola max drawdown"""
    if equity_series is None or len(equity_series) == 0:
        return 0
    
    equity_series = equity_series.dropna()
    if len(equity_series) == 0:
        return 0
    
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak
    return float(drawdown.min() * 100)  # in percentuale

def calculate_sharpe_ratio(equity_series, risk_free_rate=0.02):
    """Calcola Sharpe Ratio annualizzato"""
    if equity_series is None or len(equity_series) < 2:
        return 0
    
    equity_series = equity_series.dropna()
    if len(equity_series) < 2:
        return 0
    
    returns = equity_series.pct_change().dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0
    
    excess_returns = returns - risk_free_rate/252  # Daily risk-free
    sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
    return float(sharpe)

# ========== BUILD BALANCED DATASET ==========

def build_balanced_ml_dataset(merged, df_trades, assets_to_load=('US500.pro','US100.pro','BTC','VIX','GOLD.pro'), news_path=None):
    """Costruisce dataset ML bilanciato"""
    
    log_info("Costruendo dataset ML bilanciato...")
    
    # Carica asset data
    assets_db = {}
    for s in assets_to_load:
        try:
            dfp = load_price_data(s, timeframe_preference=('D1','M5'))
            if dfp is not None:
                assets_db[s] = dfp
        except Exception:
            continue
    
    equity = merged['equity_close']
    
    # 1. Drawdown periods (classe 1) - usa tutti
    dd_eps = detect_all_drawdown_episodes(equity)
    dd_windows = []
    for e in dd_eps[:100]:  # Limita a 100 drawdown più grandi
        start = e.get('peak_time')
        end = e.get('recovery_time') if pd.notna(e.get('recovery_time')) else e.get('trough_time')
        dd_windows.append((start, end))
    
    # 2. Best periods (classe 2)
    best_periods = top_k_best_periods(equity, freq='W', top_k=50)
    best_windows = [(p['start'], p['end']) for p in best_periods]
    
    # 3. Neutral periods (classe 0) - campiona strategicamente
    all_times = equity.index
    neutral_windows = []
    
    # Campiona periodi casuali che non siano drawdown o best
    n_neutral = min(100, len(dd_windows) + len(best_windows))
    for _ in range(n_neutral):
        # Scegli un punto casuale
        idx = np.random.randint(0, len(all_times) - 100)
        start = all_times[idx]
        end = start + pd.Timedelta(days=7)  # Finestra di 7 giorni
        
        # Verifica che non si sovrapponga con altri periodi
        overlap = False
        for s, e in dd_windows + best_windows:
            if not (end < s or start > e):
                overlap = True
                break
        
        if not overlap:
            neutral_windows.append((start, end))
    
    # Carica news se disponibili
    news_df = None
    if news_path and Path(news_path).exists():
        news_df = pd.read_csv(news_path)
        news_df['time'] = pd.to_datetime(news_df['time'], errors='coerce')
    
    # Assemble rows
    rows = []
    
    # Classe 1: Drawdown
    for s, e in dd_windows:
        f = compute_window_features(s, e, equity, assets_db, news_df)
        f['label'] = 1
        rows.append(f)
    
    # Classe 2: Best periods
    for s, e in best_windows:
        f = compute_window_features(s, e, equity, assets_db, news_df)
        f['label'] = 2
        rows.append(f)
    
    # Classe 0: Neutral
    for s, e in neutral_windows:
        f = compute_window_features(s, e, equity, assets_db, news_df)
        f['label'] = 0
        rows.append(f)
    
    # Crea DataFrame
    df = pd.DataFrame(rows)
    
    # Rimuovi colonne non numeriche
    if 'window_start' in df.columns:
        df = df.drop(columns=['window_start', 'window_end'])
    
    # Gestisci valori NaN
    df = df.fillna(0)
    
    # Assicurati che ci siano tutte e tre le classi
    class_counts = df['label'].value_counts()
    log_info(f"Distribuzione classi nel dataset bilanciato: {class_counts.to_dict()}")
    
    df.to_csv(OUTPUT_DIR / 'balanced_ml_dataset.csv', index=False)
    log_info(f"Dataset bilanciato salvato: {len(df)} righe")
    
    return df

# ========== TRAIN BALANCED MODELS ==========

def train_balanced_models(df, target_col='label'):
    """Addestra modelli con dataset bilanciato"""
    
    if df is None or df.empty:
        log_warn("Dataset vuoto per training")
        return {}
    
    # Verifica che ci siano almeno 2 classi
    unique_classes = df[target_col].unique()
    if len(unique_classes) < 2:
        log_warn(f"Solo {len(unique_classes)} classe(i) nel dataset. Skip training.")
        return {}
    
    df = df.copy()
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    
    # Seleziona solo colonne numeriche
    X = X.select_dtypes(include=[np.number])
    if X.empty:
        log_warn("Nessuna feature numerica disponibile")
        return {}
    
    # Gestisci valori NaN
    X = X.fillna(0)
    
    # Configura TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Modelli
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        'LogisticRegression': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            eval_metric='logloss'
        )
    
    results = {}
    
    for name, model in models.items():
        try:
            pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
            
            # Cross-validation
            cv_scores = []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Verifica che ci siano almeno 2 classi nel training set
                if len(np.unique(y_train)) < 2:
                    continue
                
                pipe.fit(X_train, y_train)
                score = pipe.score(X_test, y_test)
                cv_scores.append(score)
            
            if cv_scores:
                results[name] = {
                    'cv_mean_accuracy': float(np.mean(cv_scores)),
                    'cv_std_accuracy': float(np.std(cv_scores)),
                    'cv_scores': [float(s) for s in cv_scores]
                }
                
                # Feature importance per RandomForest
                if name == 'RandomForest':
                    try:
                        pipe.fit(X, y)
                        importances = pipe.named_steps['clf'].feature_importances_
                        feature_imp = pd.Series(importances, index=X.columns)
                        feature_imp.sort_values(ascending=False).head(20).to_csv(
                            OUTPUT_DIR / f'balanced_feature_importance_{name}.csv'
                        )
                    except Exception as e:
                        log_warn(f"Errore feature importance {name}: {e}")
            else:
                results[name] = {'error': 'Nessuna fold valida'}
            
        except Exception as e:
            results[name] = {'error': str(e)}
            log_warn(f"Errore training {name}: {e}")
    
    # Salva risultati
    pd.DataFrame(results).T.to_csv(OUTPUT_DIR / 'balanced_ml_results.csv')
    return results

# ========== ANALISI DIVERSIFICAZIONE ==========

def analyze_diversification_opportunities(df_trades, assets_db, merged):
    """Analizza opportunità di diversificazione"""
    
    log_info("=== ANALISI DIVERSIFICAZIONE ===")
    
    suggestions = []
    
    # 1. Analisi performance per asset
    asset_performance = {}
    for asset_name, asset_df in assets_db.items():
        if asset_name in df_trades['Symbol'].unique():
            # Calcola performance dell'asset
            try:
                asset_returns = asset_df['CLOSE'].pct_change().dropna()
                asset_vol = asset_returns.std() * np.sqrt(252)
                asset_performance[asset_name] = {
                    'volatility': asset_vol,
                    'avg_daily_return': asset_returns.mean()
                }
            except:
                pass
    
    # 2. Identifica asset con bassa correlazione
    equity_daily = merged['equity_close'].resample('D').last().dropna()
    equity_returns = equity_daily.pct_change().dropna()
    
    correlations = {}
    for asset_name, asset_df in assets_db.items():
        try:
            asset_daily = asset_df['CLOSE'].resample('D').last().reindex(equity_daily.index, method='ffill')
            asset_returns_asset = asset_daily.pct_change().dropna()
            
            # Allinea
            aligned = pd.concat([equity_returns, asset_returns_asset], axis=1).dropna()
            if len(aligned) > 10:
                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                correlations[asset_name] = corr
        except:
            continue
    
    # 3. Suggerimenti basati su correlazione
    low_corr_assets = [asset for asset, corr in correlations.items() if abs(corr) < 0.3]
    if low_corr_assets:
        suggestions.append(f"Asset a bassa correlazione con il portafoglio: {', '.join(low_corr_assets[:3])}")
        suggestions.append("Considera di aumentare l'esposizione a questi asset per diversificazione")
    
    # 4. Strategie di hedge suggerite
    hedge_strategies = [
        {
            'name': 'Gold Hedge',
            'description': 'Aumenta posizioni long su oro durante periodi di alta volatilità',
            'asset': 'GOLD.pro',
            'when': 'VIX > 25'
        },
        {
            'name': 'USDJPY Safe Haven',
            'description': 'USD/JPY tende ad apprezzarsi in crisi di mercato',
            'asset': 'USDJPY.pro',
            'when': 'Equity drawdown > 2%'
        },
        {
            'name': 'VIX Mean Reversion',
            'description': 'Trading sul ritorno alla media del VIX',
            'asset': 'VIX',
            'when': 'VIX > 30 (short) o VIX < 15 (long)'
        }
    ]
    
    # Salva risultati
    diversification_results = {
        'correlations': correlations,
        'asset_performance': asset_performance,
        'suggestions': suggestions,
        'hedge_strategies': hedge_strategies
    }
    
    with open(OUTPUT_DIR / 'diversification_analysis.txt', 'w') as f:
        f.write("=== ANALISI DIVERSIFICAZIONE ===\n\n")
        f.write("Correlazioni con portafoglio:\n")
        for asset, corr in sorted(correlations.items(), key=lambda x: abs(x[1])):
            f.write(f"  {asset}: {corr:.3f}\n")
        
        f.write("\nSuggerimenti:\n")
        for suggestion in suggestions:
            f.write(f"  • {suggestion}\n")
        
        f.write("\nStrategie di Hedge Suggerite:\n")
        for strategy in hedge_strategies:
            f.write(f"\n  {strategy['name']}:\n")
            f.write(f"    {strategy['description']}\n")
            f.write(f"    Asset: {strategy['asset']}\n")
            f.write(f"    Quando: {strategy['when']}\n")
    
    return diversification_results

# ========== PIPELINE MIGLIORATA ==========

def run_enhanced_pipeline(report_path=REPORT_PATH, news_path=None, run_vix_simulation=True):
    """Pipeline principale con tutte le nuove funzionalità"""
    
    log_info("=== PIPELINE MIGLIORATA AVVIATA ===")
    
    # Carica dati originali
    if not Path(report_path).exists():
        log_warn(f"Report file non trovato: {report_path}")
        return None
    
    df_trades = pd.read_csv(report_path)
    df_trades.columns = [c.strip() for c in df_trades.columns]
    log_info(f"Trades caricati: {len(df_trades)} righe")
    
    # Carica asset data
    symbols = list(df_trades['Symbol'].unique()) + ['VIX', 'US500.pro', 'US100.pro', 'GOLD.pro', 'BTC']
    assets_db = {}
    
    for s in symbols:
        try:
            d = load_price_data(s, timeframe_preference=('D1', 'M5'))
            if d is not None:
                assets_db[s] = d
        except Exception as e:
            log_warn(f"Errore caricamento {s}: {e}")
            continue
    
    log_info(f"Asset caricati: {len(assets_db)}")
    
    # Calcola equity originale
    merged = compute_equity_highres(df_trades, assets_db)
    if merged is None:
        log_warn("Impossibile calcolare equity")
        return None
    
    merged['equity_close'].to_csv(OUTPUT_DIR / 'equity_close_series.csv')
    
    # 1. VERIFICA ANALISI DRAWDOWN
    log_info("\n1. VERIFICA ANALISI DRAWDOWN")
    crisis_results = verify_drawdown_analysis(df_trades, assets_db, merged)
    check_drawdown_veracity(merged)
    
    # Salva risultati verifica
    if crisis_results:
        pd.DataFrame(crisis_results).T.to_csv(OUTPUT_DIR / 'crisis_performance_verification.csv')
    
    # 2. ANALISI ORIGINALE DRAWDOWNS E VIX
    log_info("\n2. ANALISI ORIGINALE DRAWDOWNS E VIX")
    suggested_rules = analyze_drawdowns_and_vix(merged, assets_db, thresholds=(2000,1000))
    
    # 3. BUILD BALANCED ML DATASET
    log_info("\n3. COSTRUZIONE DATASET ML BILANCIATO")
    df_ml = build_balanced_ml_dataset(merged, df_trades)
    
    # 4. TRAINING MODELLI BILANCIATI
    log_info("\n4. TRAINING MODELLI BILANCIATI")
    if df_ml is not None and not df_ml.empty:
        ml_results = train_balanced_models(df_ml)
    else:
        ml_results = {}
        log_warn("Dataset ML vuoto, skip training")
    
    # 5. SIMULAZIONE RISK ENGINE VIX
    scenario_results = {}
    if run_vix_simulation and 'VIX' in assets_db:
        log_info("\n5. SIMULAZIONE RISK ENGINE VIX")
        
        # Scenari da testare
        scenarios = [
            {'name': 'Baseline', 'initial_size': 1.0},
            {'name': 'Conservative_VIX25', 'thresholds': {'reduce_30pct': 25, 'reduce_50pct': 30}},
            {'name': 'Aggressive_VIX20', 'thresholds': {'reduce_30pct': 20, 'reduce_50pct': 25}},
        ]
        
        # Equity originale per confronto
        original_equity = merged['equity_close']
        scenario_results['Original'] = {
            'final_equity': float(original_equity.iloc[-1]) if len(original_equity) > 0 else INITIAL_BALANCE,
            'max_drawdown': calculate_max_drawdown(original_equity),
            'sharpe_ratio': calculate_sharpe_ratio(original_equity)
        }
        
        for scenario in scenarios:
            log_info(f"  Simulando scenario: {scenario['name']}")
            
            # Configura engine
            engine = RiskEngineVIX(assets_db)
            if 'thresholds' in scenario:
                engine.thresholds.update(scenario['thresholds'])
            
            # Esegui simulazione
            equity_curve, details, adjustments = engine.simulate_equity_with_risk_management(df_trades)
            
            if equity_curve is not None and len(equity_curve) > 0:
                # Salva risultati
                scenario_results[scenario['name']] = {
                    'final_equity': float(equity_curve.iloc[-1]),
                    'max_drawdown': calculate_max_drawdown(equity_curve),
                    'sharpe_ratio': calculate_sharpe_ratio(equity_curve),
                    'n_adjustments': len(adjustments)
                }
                
                # Salva equity curve
                pd.Series(equity_curve, name='equity').to_csv(OUTPUT_DIR / f'equity_{scenario["name"]}.csv')
                
                # Salva adjustments
                if adjustments:
                    pd.DataFrame(adjustments).to_csv(OUTPUT_DIR / f'vix_adjustments_{scenario["name"]}.csv', index=False)
            else:
                log_warn(f"Simulazione {scenario['name']} fallita")
        
        # Confronto scenari
        if len(scenario_results) > 1:
            comparison_df = pd.DataFrame(scenario_results).T
            comparison_df.to_csv(OUTPUT_DIR / 'vix_scenarios_comparison.csv')
            
            log_info("\nConfronto scenari VIX:")
            print(comparison_df[['final_equity', 'max_drawdown', 'sharpe_ratio', 'n_adjustments']])
    
    # 6. ANALISI DIVERSIFICAZIONE
    log_info("\n6. ANALISI DIVERSIFICAZIONE")
    diversification = analyze_diversification_opportunities(df_trades, assets_db, merged)
    
    # 7. REPORT FINALE
    log_info("\n7. GENERAZIONE REPORT FINALE")
    generate_final_report(merged, scenario_results, diversification, crisis_results, suggested_rules)
    
    log_info("\n=== PIPELINE COMPLETATA ===")
    log_info(f"Risultati salvati in: {OUTPUT_DIR}")
    
    return {
        'crisis_results': crisis_results,
        'ml_results': ml_results,
        'scenario_results': scenario_results,
        'diversification': diversification
    }

def generate_final_report(merged, scenarios, diversification, crisis_results, suggested_rules):
    """Genera report finale"""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ANALISI QUANTITATIVA COMPLETA - RISK MANAGEMENT REPORT")
    report_lines.append("=" * 80)
    
    # Performance originale
    original_equity = merged['equity_close']
    initial = original_equity.iloc[0] if len(original_equity) > 0 else INITIAL_BALANCE
    final = original_equity.iloc[-1] if len(original_equity) > 0 else initial
    total_return = (final - initial) / initial * 100 if initial != 0 else 0
    
    report_lines.append(f"\n1. PERFORMANCE ORIGINALE:")
    report_lines.append(f"   Capitale iniziale: €{initial:,.2f}")
    report_lines.append(f"   Capitale finale: €{final:,.2f}")
    report_lines.append(f"   Ritorno totale: {total_return:.2f}%")
    report_lines.append(f"   Max Drawdown: {calculate_max_drawdown(original_equity):.2f}%")
    report_lines.append(f"   Sharpe Ratio: {calculate_sharpe_ratio(original_equity):.2f}")
    
    # Verifica crisi
    if crisis_results:
        report_lines.append("\n2. PERFORMANCE IN CRISI:")
        for crisis, results in crisis_results.items():
            if results.get('pnl_total') is not None:
                report_lines.append(f"   {crisis}:")
                report_lines.append(f"     PnL: €{results['pnl_total']:,.2f}")
                if results.get('avg_vix'):
                    report_lines.append(f"     VIX medio: {results['avg_vix']:.1f}")
                if results.get('spx_return_pct'):
                    report_lines.append(f"     S&P500: {results['spx_return_pct']:.1f}%")
                report_lines.append(f"     Trades: {results.get('n_trades', 0)}")
    
    # Scenari VIX
    if scenarios:
        report_lines.append("\n3. SCENARI RISK MANAGEMENT VIX:")
        for scenario, stats in scenarios.items():
            report_lines.append(f"   {scenario}:")
            report_lines.append(f"     Equity finale: €{stats.get('final_equity', 0):,.2f}")
            report_lines.append(f"     Max DD: {stats.get('max_drawdown', 0):.2f}%")
            report_lines.append(f"     Sharpe: {stats.get('sharpe_ratio', 0):.2f}")
            if 'n_adjustments' in stats:
                report_lines.append(f"     Adjustments: {stats['n_adjustments']}")
    
    # Regole suggerite
    if suggested_rules:
        report_lines.append("\n4. REGOLE SUGGERITE DALL'ANALISI:")
        for rule in suggested_rules:
            report_lines.append(f"   • {rule}")
    
    # Diversificazione
    report_lines.append("\n5. RACCOMANDAZIONI DIVERSIFICAZIONE:")
    if diversification and diversification.get('suggestions'):
        for suggestion in diversification['suggestions'][:5]:
            report_lines.append(f"   {suggestion}")
    
    # Regole operative
    report_lines.append("\n6. REGOLE OPERATIVE IMMEDIATE:")
    report_lines.append("   • IF VIX >= 30 THEN reduce position size by 50%")
    report_lines.append("   • IF VIX >= 25 THEN reduce position size by 30%")
    report_lines.append("   • IF correlation(portfolio, NASDAQ) > 0.25 THEN hedge with put options")
    report_lines.append("   • DURING equity crisis: consider long positions on GOLD.pro or USDJPY.pro")
    
    # Salva report
    report_path = OUTPUT_DIR / 'comprehensive_analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    log_info(f"Report completo salvato in: {report_path}")
    return report_lines

# ========== MAIN ==========

if __name__ == "__main__":
    # Esegui la pipeline migliorata
    results = run_enhanced_pipeline(
        report_path=REPORT_PATH,
        news_path=None,
        run_vix_simulation=True
    )