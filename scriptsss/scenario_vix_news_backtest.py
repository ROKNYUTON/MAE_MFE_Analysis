"""
Backtest "what-if" scenarios combining NEWS windows (remove trades overlapping events)
and VIX-based sizing (scale per-trade P/L when trade opened under high-VIX).
Outputs:
 - data/reports/scenario_analysis/scenario_comparison.csv
 - equity comparison PNGs
 - per-scenario equity series CSVs (euros)
 - prints summary in console
"""

from pathlib import Path
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import json
import math

# ---------------- CONFIG ----------------
BASE = Path('.')
TRADES_CSV = BASE / 'data' / 'reports' / 'multi_asset_report.csv'
# CORRECTED PATH: Using the file you found
NEWS_CSV = BASE / 'data' / 'reports' / 'news_analysis' / 'ml_dataset_news_events.csv'
VIX_LOCAL = BASE / 'data' / 'datasets' / 'VIX_D1_2020_2025.csv'

OUTPUT_DIR = BASE / 'data' / 'reports' / 'scenario_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_BALANCE = 100000.0

# VIX thresholds -> scale factor to apply to trade P/L when VIX >= threshold
VIX_RULES = [
    (30, 0.5),
    (25, 0.7),
    (20, 0.8),
]
VIX_RULES = sorted(VIX_RULES, key=lambda x: -x[0])

# News windows to test (minutes)
NEWS_WINDOWS = [15, 60]
NEWS_TYPES = ['NFP', 'CPI', 'FOMC']

# Plot settings
MAX_SCENARIOS_TO_PLOT = 6

# ---------------- logging ----------------
def info(msg): print("[INFO]", msg)
def warn(msg): print("[WARN]", msg)

# ---------------- helpers ----------------
def parse_trades(path=TRADES_CSV):
    if not path.exists():
        raise FileNotFoundError(f"Trades CSV not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # FIXED THE SYNTAX ERROR HERE:
    if 'Open time' in df.columns and 'Close time' in df.columns:
        df['Open time'] = pd.to_datetime(df['Open time'], errors='coerce', dayfirst=True)
        df['Close time'] = pd.to_datetime(df['Close time'], errors='coerce', dayfirst=True)
    else:
        raise ValueError("Trades CSV must contain 'Open time' and 'Close time' columns")
    pl_cols = [c for c in df.columns if ('Profit' in c or 'profit' in c or 'P/L' in c or 'PL' in c)]
    if 'Profit/Loss (Global)' in df.columns:
        pl_col = 'Profit/Loss (Global)'
    elif pl_cols:
        pl_col = pl_cols[0]
    else:
        raise ValueError("Could not find Profit/Loss column in trades CSV")
    df['PL'] = df[pl_col].astype(float)
    if 'Size' not in df.columns:
        df['Size'] = 1.0
    df = df.dropna(subset=['Open time','Close time']).sort_values('Close time').reset_index(drop=True)
    info(f"Loaded {len(df)} trades.")
    return df

def load_news(equity_index):
    """
    FIXED VERSION: Handles your 'event_time' column name.
    Your CSV has duplicate events (15min and 60min windows) - we deduplicate.
    """
    p = NEWS_CSV
    info(f"Loading news from: {p}")
    if not p.exists():
        warn(f"News CSV not found at specified path: {p}")
        warn("-> Generating approximate NFP/CPI/FOMC events as fallback.")
        return approximate_generate_events(equity_index.min(), equity_index.max())

    try:
        df = pd.read_csv(p)
        info(f"Raw CSV loaded. Shape: {df.shape}, Columns: {df.columns.tolist()}")

        # FIX 1: Your CSV uses 'event_time', not 'time'
        if 'event_time' in df.columns:
            df['time'] = pd.to_datetime(df['event_time'], errors='coerce')
            info("Using 'event_time' column as event time.")
        elif 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
        else:
            # Try other common names
            time_col_candidates = ['datetime', 'date', 'timestamp']
            for col in time_col_candidates:
                if col in df.columns:
                    df['time'] = pd.to_datetime(df[col], errors='coerce')
                    info(f"Using '{col}' column as event time.")
                    break
            else:
                raise ValueError("No recognizable time column found in news CSV.")

        # FIX 2: Ensure event_type is properly formatted
        if 'event_type' in df.columns:
            df['event_type'] = df['event_type'].astype(str).str.upper().str.strip()
            info(f"Unique event types found: {df['event_type'].unique()}")
        else:
            warn("No 'event_type' column found. Attempting to use 'type' column if available.")
            if 'type' in df.columns:
                df['event_type'] = df['type'].astype(str).str.upper().str.strip()
            else:
                warn("No event type column found. All rows will be treated as generic events.")
                df['event_type'] = 'GENERIC'

        df = df.dropna(subset=['time']).reset_index(drop=True)
        
        # IMPORTANT: Your CSV has duplicate events (15min and 60min windows)
        # We need to deduplicate by time and event_type
        info(f"Before deduplication: {len(df)} rows")
        df = df.drop_duplicates(subset=['time', 'event_type']).reset_index(drop=True)
        info(f"After deduplication: {len(df)} unique events")

        # Filter for relevant event types if column exists
        if 'event_type' in df.columns:
            filtered_df = df[df['event_type'].str.upper().isin([et.upper() for et in NEWS_TYPES])].copy()
            info(f"Filtered to {len(filtered_df)} relevant events (NFP/CPI/FOMC) out of {len(df)} total.")
            if filtered_df.empty:
                warn("Filter resulted in NO events. Check 'event_type' values in your CSV.")
                warn("Available event types: " + str(df['event_type'].unique()))
            return filtered_df
        else:
            return df

    except Exception as e:
        warn(f"Failed to load news CSV: {e}")
        warn("-> Falling back to approximate event generation.")
        return approximate_generate_events(equity_index.min(), equity_index.max())

def approximate_generate_events(start_dt, end_dt):
    events = []
    cur = date(start_dt.year, start_dt.month, 1)
    while cur <= date(end_dt.year, end_dt.month, 1):
        y, m = cur.year, cur.month
        def first_weekday_of_month(yr, mo, wd):
            d = date(yr, mo, 1)
            days_ahead = (wd - d.weekday() + 7) % 7
            return d + timedelta(days=days_ahead)
        def nth_weekday(yr, mo, wd, n=1):
            return first_weekday_of_month(yr, mo, wd) + timedelta(weeks=n-1)
        try:
            fd = nth_weekday(y, m, 4, 1)
            dt_nfp = datetime(fd.year, fd.month, fd.day, 13, 30)
            if start_dt <= dt_nfp <= end_dt:
                events.append({'time': dt_nfp, 'event_type': 'NFP', 'description': 'Approx NFP (1st Fri)'})
        except Exception:
            pass
        try:
            cd = nth_weekday(y, m, 1, 2)
            dt_cpi = datetime(cd.year, cd.month, cd.day, 13,30)
            if start_dt <= dt_cpi <= end_dt:
                events.append({'time': dt_cpi, 'event_type': 'CPI', 'description': 'Approx CPI (2nd Tue)'})
        except Exception:
            pass
        fomc_months = {1,3,4,6,7,9,11,12}
        if m in fomc_months:
            try:
                fw = nth_weekday(y, m, 2, 2)
                dt_fomc = datetime(fw.year, fw.month, fw.day, 18, 0)
                if start_dt <= dt_fomc <= end_dt:
                    events.append({'time': dt_fomc, 'event_type': 'FOMC', 'description': 'Approx FOMC (2nd Wed)'})
            except Exception:
                pass
        if cur.month == 12:
            cur = date(cur.year+1, 1, 1)
        else:
            cur = date(cur.year, cur.month+1, 1)
    if events:
        events = sorted(events, key=lambda x: x['time'])
    info(f"Generated {len(events)} approximate events.")
    return pd.DataFrame(events)

def load_vix(equity_index):
    """
    IMPROVED: Better VIX data loading with Yahoo Finance as primary source.
    """
    # Try Yahoo Finance first for reliability
    try:
        start_date = equity_index.min().date()
        end_date = (equity_index.max() + timedelta(days=1)).date()
        info(f"Downloading VIX from Yahoo Finance ({start_date} to {end_date})...")
        
        # Use 1-hour data for better time matching with trades
        v = yf.download("^VIX", 
                       start=start_date, 
                       end=end_date, 
                       interval='1h', 
                       progress=False,
                       timeout=30)
        
        if v is None or v.empty:
            warn("Yahoo VIX download returned empty. Trying daily data...")
            v = yf.download("^VIX", 
                           start=start_date, 
                           end=end_date, 
                           interval='1d', 
                           progress=False,
                           timeout=30)
        
        if v is not None and not v.empty:
            v['VIX'] = v['Close']
            v.index = pd.to_datetime(v.index)
            vix_df = v[['VIX']].dropna()
            info(f"Loaded {len(vix_df)} VIX records from Yahoo Finance (min: {vix_df['VIX'].min():.1f}, max: {vix_df['VIX'].max():.1f})")
            return vix_df
    except Exception as e:
        warn(f"Yahoo Finance VIX download failed: {e}")

    # Fallback to local file if Yahoo fails
    p = VIX_LOCAL
    if p.exists():
        try:
            df = pd.read_csv(p)
            cols = [c.replace('<','').replace('>','').strip() for c in df.columns]
            df.columns = cols
            
            if 'DATE' in df.columns and 'TIME' in df.columns:
                df['time'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str), errors='coerce')
                df.set_index('time', inplace=True)
            elif 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
                df.set_index('time', inplace=True)
            elif 'Date' in df.columns:
                df['time'] = pd.to_datetime(df['Date'], errors='coerce')
                df.set_index('time', inplace=True)
                
            if 'CLOSE' in df.columns:
                df['VIX'] = df['CLOSE']
            elif 'Close' in df.columns:
                df['VIX'] = df['Close']
            elif 'Adj Close' in df.columns:
                df['VIX'] = df['Adj Close']
            else:
                # Try to find any numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df['VIX'] = df[numeric_cols[0]]
                else:
                    raise ValueError("No numeric VIX data column found")
            
            df.index = pd.to_datetime(df.index)
            vix_df = df[['VIX']].dropna()
            info(f"Loaded {len(vix_df)} VIX records from local file (min: {vix_df['VIX'].min():.1f}, max: {vix_df['VIX'].max():.1f})")
            return vix_df
        except Exception as e:
            warn(f"Failed to load local VIX: {e}")
    
    warn("No VIX data available — VIX-based scenarios will apply no scaling")
    return None

def trades_overlap_event(trade_row, event_time, window_min):
    start = event_time - timedelta(minutes=window_min)
    end = event_time + timedelta(minutes=window_min)
    return (trade_row['Open time'] < end) and (trade_row['Close time'] > start)

def remove_trades_overlapping_events(trades_df, events_df, event_type, window_min):
    if events_df is None or events_df.empty:
        return trades_df.copy()
    evs = events_df[events_df['event_type'].str.upper() == event_type.upper()]
    if evs.empty:
        info(f"No events of type '{event_type}' to filter.")
        return trades_df.copy()
    mask = np.ones(len(trades_df), dtype=bool)
    for i, tr in trades_df.iterrows():
        keep = True
        for _, ev in evs.iterrows():
            if trades_overlap_event(tr, ev['time'], window_min):
                keep = False
                break
        mask[i] = keep
    filtered_trades = trades_df.loc[mask].reset_index(drop=True)
    info(f"Filter '{event_type} ±{window_min}min': removed {len(trades_df)-len(filtered_trades)} trades.")
    return filtered_trades

def remove_trades_overlapping_any(events_df, trades_df, window_min):
    if events_df is None or events_df.empty:
        return trades_df.copy()
    evs = events_df
    mask = np.ones(len(trades_df), dtype=bool)
    for i, tr in trades_df.iterrows():
        keep = True
        for _, ev in evs.iterrows():
            if trades_overlap_event(tr, ev['time'], window_min):
                keep = False
                break
        mask[i] = keep
    filtered_trades = trades_df.loc[mask].reset_index(drop=True)
    info(f"Filter 'ALL NEWS ±{window_min}min': removed {len(trades_df)-len(filtered_trades)} trades.")
    return filtered_trades

def apply_vix_scaling(trades_df, vix_df, vix_rules):
    """
    FIXED: Proper VIX scaling with debug output.
    Only scales trades opened when VIX >= threshold.
    """
    if vix_df is None or vix_df.empty:
        info("No VIX data -> no scaling applied (scale=1.0 for all).")
        trades_df['PL_adj'] = trades_df['PL'].astype(float)
        return trades_df
    
    vix_series = vix_df['VIX'].sort_index()
    pl_adj = []
    scaled_count = 0
    vix_values = []
    
    # DEBUG: Set to True to see VIX values for first 20 trades
    DEBUG_VIX = True
    
    for idx, tr in trades_df.iterrows():
        ot = tr['Open time']
        try:
            # Get the last VIX value available at or before the trade open time
            v = vix_series.asof(ot)
            if pd.isna(v):
                scale = 1.0
                v = None
            else:
                v = float(v)
                vix_values.append(v)
                scale = 1.0
                # Apply the first (highest) threshold that is met
                for thr, sc in vix_rules:
                    if v >= thr:
                        scale = sc
                        scaled_count += 1
                        break
            
            if DEBUG_VIX and idx < 20:
                print(f"  DEBUG Trade {idx}: Open={ot}, VIX={v}, Scale={scale}")
                
        except Exception as e:
            if DEBUG_VIX and idx < 20:
                print(f"  DEBUG Trade {idx}: Error {e}")
            scale = 1.0
            
        pl_adj.append(tr['PL'] * scale)
    
    # Only show debug for first batch
    if DEBUG_VIX:
        DEBUG_VIX = False
    
    trades_df = trades_df.copy()
    trades_df['PL_adj'] = pl_adj
    
    if vix_values:
        info(f"VIX stats: min={min(vix_values):.1f}, max={max(vix_values):.1f}, mean={np.mean(vix_values):.1f}")
    info(f"VIX scaling applied to {scaled_count} of {len(trades_df)} trades (thresholds: {VIX_RULES}).")
    
    return trades_df

def build_equity_series_from_trades(trades_df, initial_balance=INITIAL_BALANCE):
    df = trades_df.copy()
    if 'PL_adj' not in df.columns:
        df['PL_adj'] = df['PL'].astype(float)
    df = df.sort_values('Close time').reset_index(drop=True)
    balances = []
    bal = float(initial_balance)
    if len(df) == 0:
        now = pd.Timestamp.now().floor('1h')
        return pd.Series([initial_balance], index=pd.DatetimeIndex([now]), name='equity')
    first_close = df['Close time'].iloc[0]
    balances.append((first_close - timedelta(seconds=1), initial_balance))
    for idx, row in df.iterrows():
        bal += float(row['PL_adj'])
        balances.append((row['Close time'], bal))
    df_bal = pd.DataFrame(balances, columns=['time', 'balance'])
    df_bal['time'] = pd.to_datetime(df_bal['time'])
    df_bal = df_bal.groupby('time', as_index=True).last().sort_index()
    s = df_bal['balance']
    start = s.index.min()
    end = s.index.max()
    full_idx = pd.date_range(start=start, end=end, freq='1h')
    s_full = s.reindex(full_idx, method='ffill')
    if pd.isna(s_full.iloc[0]):
        s_full.iloc[0] = initial_balance
    s_full.name = 'equity'
    return s_full

def compute_drawdown_metrics(equity_series):
    s = equity_series.dropna().astype(float)
    if s.empty:
        return {'final_equity': np.nan, 'total_pnl': np.nan, 'max_dd_eur': np.nan, 'max_dd_time': None}
    peak = s.cummax()
    dd = s - peak
    max_dd = dd.min()
    max_dd_time = dd.idxmin()
    final_equity = float(s.iloc[-1])
    total_pnl = final_equity - float(s.iloc[0])
    return {'final_equity': final_equity, 'total_pnl': total_pnl, 'max_dd_eur': float(max_dd), 'max_dd_time': max_dd_time}

def generate_scenarios(trades_df, events_df, vix_df):
    info("Generating scenario combinations...")
    scenarios = []
    original_trades = trades_df.copy()
    original_trades['PL_adj'] = original_trades['PL']
    scenarios.append({'name':'ORIGINAL','trades': original_trades})
    
    for ev in NEWS_TYPES:
        for w in NEWS_WINDOWS:
            name = f'NO_{ev}_{w}min'
            tdf = remove_trades_overlapping_events(trades_df.copy(), events_df, ev, w)
            tdf['PL_adj'] = tdf['PL']
            scenarios.append({'name':name,'trades':tdf})
    
    for w in NEWS_WINDOWS:
        name = f'NO_ALL_NEWS_{w}min'
        tdf = remove_trades_overlapping_any(events_df, trades_df.copy(), w)
        tdf['PL_adj'] = tdf['PL']
        scenarios.append({'name':name,'trades':tdf})
    
    for thr, sc in VIX_RULES:
        name = f'VIX_GE_{thr}'
        tdf = apply_vix_scaling(trades_df.copy(), vix_df, VIX_RULES)
        scenarios.append({'name':name,'trades':tdf})
    
    for ev in NEWS_TYPES:
        for w in NEWS_WINDOWS:
            base_trades = remove_trades_overlapping_events(trades_df.copy(), events_df, ev, w)
            for thr, sc in VIX_RULES:
                name = f'NO_{ev}_{w}min_AND_VIX_GE_{thr}'
                tdf = apply_vix_scaling(base_trades.copy(), vix_df, VIX_RULES)
                scenarios.append({'name':name,'trades':tdf})
    
    for w in NEWS_WINDOWS:
        base_trades = remove_trades_overlapping_any(events_df, trades_df.copy(), w)
        for thr, sc in VIX_RULES:
            name = f'NO_ALL_NEWS_{w}min_AND_VIX_GE_{thr}'
            tdf = apply_vix_scaling(base_trades.copy(), vix_df, VIX_RULES)
            scenarios.append({'name':name,'trades':tdf})
    
    info(f"Generated {len(scenarios)} total scenarios.")
    return scenarios

def run_all():
    info("="*60)
    info("STARTING ENHANCED SCENARIO BACKTEST ANALYSIS")
    info("="*60)

    info("1. Loading trades...")
    trades = parse_trades(TRADES_CSV)
    if trades.empty:
        warn("No trades loaded -> aborting.")
        return
    info(f"   Time range: {trades['Open time'].min()} to {trades['Close time'].max()}")

    info("2. Loading news/events...")
    events = load_news(trades['Close time'])
    info(f"   Events loaded for backtest: {len(events)}")
    if len(events) > 0:
        info(f"   Event distribution: {events['event_type'].value_counts().to_dict()}")

    info("3. Loading VIX data...")
    vix = load_vix(trades['Close time'])

    info("4. Generating scenarios...")
    scenarios = generate_scenarios(trades, events, vix)

    info("5. Evaluating scenarios...")
    results = []
    equity_series_map = {}

    for sc in scenarios:
        name = sc['name']
        tdf = sc['trades'].copy()
        s = build_equity_series_from_trades(tdf, initial_balance=INITIAL_BALANCE)
        metrics = compute_drawdown_metrics(s)
        trades_cnt = len(tdf)
        res = {
            'scenario': name,
            'final_equity_eur': metrics['final_equity'],
            'total_pnl_eur': metrics['total_pnl'],
            'max_dd_eur': metrics['max_dd_eur'],
            'max_dd_time': metrics['max_dd_time'],
            'trades': trades_cnt
        }
        results.append(res)
        equity_series_map[name] = s
        try:
            outcsv = OUTPUT_DIR / f'eq_series_{name}.csv'
            s.rename('equity').to_csv(outcsv, index=True)
        except Exception as e:
            warn(f"Failed to save equity series for {name}: {e}")

    dfres = pd.DataFrame(results)
    orig_row = dfres[dfres['scenario']=='ORIGINAL']
    if not orig_row.empty:
        orig_final = float(orig_row.iloc[0]['final_equity_eur'])
        dfres['delta_vs_original_eur'] = dfres['final_equity_eur'] - orig_final
    else:
        dfres['delta_vs_original_eur'] = np.nan
    
    dfres = dfres.sort_values('final_equity_eur', ascending=False).reset_index(drop=True)
    summary_csv = OUTPUT_DIR / 'scenario_comparison.csv'
    dfres.to_csv(summary_csv, index=False)
    info(f"\n[MAIN RESULT] Scenario summary saved to: {summary_csv}")

    plot_count = min(MAX_SCENARIOS_TO_PLOT, len(dfres))
    top_scenarios = ['ORIGINAL'] + [s for s in dfres['scenario'].tolist() if s!='ORIGINAL'][:plot_count-1]
    plt.figure(figsize=(14,8))
    for name in top_scenarios:
        s = equity_series_map.get(name)
        if s is None or s.empty: 
            continue
        plt.plot(s.index, s.values, lw=2 if name=='ORIGINAL' else 1.2, 
                label=f'{name} (Final €{s.iloc[-1]:.0f})')
    plt.title('Equity Curve Comparison: ORIGINAL vs Top Scenarios (Euro)')
    plt.xlabel('Date')
    plt.ylabel('Equity (€)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    outpng = OUTPUT_DIR / 'equity_comparison_top.png'
    plt.tight_layout()
    plt.savefig(outpng, dpi=150, bbox_inches='tight')
    info(f"[PLOT] Equity comparison plot saved to: {outpng}")

    dd_summary = dfres[['scenario','final_equity_eur','total_pnl_eur',
                       'max_dd_eur','trades','delta_vs_original_eur']].copy()
    dd_summary.to_csv(OUTPUT_DIR / 'scenario_drawdown_summary.csv', index=False)

    dfres['score'] = dfres['final_equity_eur'] - 0.5 * (dfres['max_dd_eur'].abs())
    dfres_scored = dfres.sort_values('score', ascending=False).reset_index(drop=True)
    
    info("\n" + "="*60)
    info("TOP 5 SCENARIOS (Score = Final Equity - 0.5*|MaxDD|):")
    info("="*60)
    for i, r in dfres_scored.head(5).iterrows():
        info(f"{i+1}. {r['scenario']:35} | Final: €{r['final_equity_eur']:12.2f} | "
             f"MaxDD: €{r['max_dd_eur']:9.2f} | Trades: {int(r['trades']):5} | "
             f"Δ vs Orig: €{r.get('delta_vs_original_eur',0):8.2f}")
    info("="*60)
    
    info(f"\nADDITIONAL METRICS:")
    info(f"  - Original Strategy: €{orig_final:.2f} final, {len(trades)} trades")
    info(f"  - Best Scenario: {dfres.iloc[0]['scenario']} (€{dfres.iloc[0]['final_equity_eur']:.2f})")
    info(f"  - Worst Scenario: {dfres.iloc[-1]['scenario']} (€{dfres.iloc[-1]['final_equity_eur']:.2f})")
    
    info(f"\nAll outputs saved in: {OUTPUT_DIR}")
    info("Analysis complete.")

if __name__ == '__main__':
    run_all()

# Aggiungi questa funzione al tuo script per analizzare i trade per condizione
def analyze_trade_characteristics(trades_df, events_df, vix_df):
    """Analizza performance dei trade divisi per condizioni di news e VIX."""
    
    # 1. Identifica trade durante news (finestra 60 min)
    during_news = []
    for idx, trade in trades_df.iterrows():
        is_during_news = False
        for _, event in events_df.iterrows():
            if trades_overlap_event(trade, event['time'], 60):
                is_during_news = True
                break
        during_news.append(is_during_news)
    
    trades_df['during_news'] = during_news
    
    # 2. Identifica trade con VIX alto (>=20 all'apertura)
    vix_at_open = []
    vix_series = vix_df['VIX'].sort_index() if vix_df is not None else None
    
    for idx, trade in trades_df.iterrows():
        if vix_series is not None:
            vix = vix_series.asof(trade['Open time'])
            vix_at_open.append(vix if not pd.isna(vix) else None)
        else:
            vix_at_open.append(None)
    
    trades_df['vix_at_open'] = vix_at_open
    trades_df['high_vix'] = trades_df['vix_at_open'] >= 20
    
    # 3. Calcola metriche per ogni gruppo
    groups = {
        'All Trades': trades_df,
        'During News': trades_df[trades_df['during_news']],
        'Outside News': trades_df[~trades_df['during_news']],
        'High VIX (>=20)': trades_df[trades_df['high_vix'] == True],
        'Low VIX (<20)': trades_df[trades_df['high_vix'] == False]
    }
    
    results = {}
    for name, group in groups.items():
        if len(group) > 0:
            results[name] = {
                'Trade Count': len(group),
                'Total PnL': group['PL'].sum(),
                'Avg PnL per Trade': group['PL'].mean(),
                'Win Rate %': (group['PL'] > 0).sum() / len(group) * 100,
                'Avg Win': group[group['PL'] > 0]['PL'].mean(),
                'Avg Loss': group[group['PL'] < 0]['PL'].mean()
            }
    
    return pd.DataFrame(results).T