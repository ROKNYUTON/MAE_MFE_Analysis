#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
news_impact_analysis.py

Analisi avanzata dell'impatto di eventi ricorrenti (NFP, CPI, FOMC) sulla equity del portafoglio,
con regressione per expected PnL e quantile regression per worst-case (5% quantile).
Genera raccomandazioni operative: STOP / REDUCE / OK.

Outputs (data/reports/news_analysis/):
 - ml_dataset_news_events.csv
 - aggregate_stats_by_event.csv
 - equity_{year}_with_events.png (per anno)
 - event_model_mean.joblib, event_model_q05.joblib
 - news_recommendations.txt
 - news_analysis_summary.json
"""

from pathlib import Path
from datetime import datetime, timedelta, date
import json
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------- CONFIG ----------------
BASE = Path('.')
REPORT_PATH = BASE / 'data' / 'reports' / 'multi_asset_report.csv'
EQUITY_SERIES_PATH = BASE / 'data' / 'reports' / 'ml_analysis' / 'equity_close_series.csv'
NEWS_CSV_PATH = BASE / 'data' / 'news' / 'news_2020_2025.csv'
VIX_LOCAL_PATH = BASE / 'data' / 'datasets' / 'VIX_D1_2020_2025.csv'

OUTPUT_DIR = BASE / 'data' / 'reports' / 'news_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS_MINUTES = [15, 60]  # windows to test (±)
EVENT_COLORS = {'FOMC': 'red', 'NFP': 'green', 'CPI': 'orange', 'OTHER': 'purple'}

RANDOM_STATE = 42

# Decision thresholds (EUR). Tune as needed.
STOP_MEAN_PNL = -500.0       # if expected mean pnl < STOP_MEAN_PNL -> STOP
STOP_WORST_Q05 = -1000.0     # if predicted 5% quantile < STOP_WORST_Q05 -> STOP
REDUCE_MEAN_PNL = -200.0     # reduce size threshold
REDUCE_WORST_Q05 = -500.0

# ---------------- logging ----------------
def info(msg): print("[INFO]", msg)
def warn(msg): print("[WARN]", msg)

# ---------------- helpers: calendar generation ----------------
def first_weekday_of_month(year, month, weekday):
    d = date(year, month, 1)
    days_ahead = (weekday - d.weekday() + 7) % 7
    return d + timedelta(days=days_ahead)

def nth_weekday_of_month(year, month, weekday, n=1):
    first = first_weekday_of_month(year, month, weekday)
    return first + timedelta(weeks=n-1)

def approximate_generate_events(start_dt, end_dt):
    events = []
    cur = date(start_dt.year, start_dt.month, 1)
    while cur <= date(end_dt.year, end_dt.month, 1):
        y, m = cur.year, cur.month
        # NFP: first Friday at 13:30 (approx)
        try:
            fd = nth_weekday_of_month(y, m, weekday=4, n=1)
            dt_nfp = datetime(fd.year, fd.month, fd.day, 13, 30)
            if start_dt <= dt_nfp <= end_dt:
                events.append({'time': dt_nfp, 'event_type': 'NFP', 'description': 'Approx NFP (1st Fri)'})
        except Exception:
            pass
        # CPI: second Tuesday at 13:30 (approx)
        try:
            cd = nth_weekday_of_month(y, m, weekday=1, n=2)
            dt_cpi = datetime(cd.year, cd.month, cd.day, 13, 30)
            if start_dt <= dt_cpi <= end_dt:
                events.append({'time': dt_cpi, 'event_type': 'CPI', 'description': 'Approx CPI (2nd Tue)'})
        except Exception:
            pass
        # FOMC approximate months
        fomc_months = {1,3,4,6,7,9,11,12}
        if m in fomc_months:
            try:
                fw = nth_weekday_of_month(y, m, weekday=2, n=2)  # 2nd Wed
                dt_fomc = datetime(fw.year, fw.month, fw.day, 18, 0)
                if start_dt <= dt_fomc <= end_dt:
                    events.append({'time': dt_fomc, 'event_type': 'FOMC', 'description': 'Approx FOMC (2nd Wed, scheduled months)'} )
            except Exception:
                pass
        # advance month
        if cur.month == 12:
            cur = date(cur.year+1, 1, 1)
        else:
            cur = date(cur.year, cur.month+1, 1)
    df = pd.DataFrame(events)
    if not df.empty:
        df = df.sort_values('time').reset_index(drop=True)
    return df

# ---------------- loaders ----------------
def load_trades():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Trades report not found: {REPORT_PATH}")
    df = pd.read_csv(REPORT_PATH)
    df.columns = [c.strip() for c in df.columns]
    df['Open time'] = pd.to_datetime(df['Open time'], errors='coerce', dayfirst=True)
    df['Close time'] = pd.to_datetime(df['Close time'], errors='coerce', dayfirst=True)
    return df.dropna(subset=['Open time','Close time'])

def load_equity_series():
    if not EQUITY_SERIES_PATH.exists():
        raise FileNotFoundError(f"Equity series file not found: {EQUITY_SERIES_PATH}")
    df = pd.read_csv(EQUITY_SERIES_PATH, index_col=0, parse_dates=True)
    if 'equity_close' in df.columns:
        s = df['equity_close']
    else:
        s = df.select_dtypes(include=[np.number]).iloc[:,0]
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s.name = 'equity'
    return s

def load_news_or_generate(equity_series):
    p = Path(NEWS_CSV_PATH)
    if p.exists():
        df = pd.read_csv(p)
        if 'time' not in df.columns:
            raise ValueError("News CSV must have 'time' column")
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        if 'event_type' not in df.columns:
            df['event_type'] = df.get('type','OTHER')
        df['event_type'] = df['event_type'].astype(str).str.upper().apply(lambda x: 'NFP' if 'NFP' in x or 'PAYROLL' in x else ('CPI' if 'CPI' in x else ('FOMC' if 'FOMC' in x or 'FED' in x else 'OTHER')))
        info(f"Loaded news CSV rows={len(df)}")
        return df[['time','event_type','description']].dropna(subset=['time']).reset_index(drop=True)
    else:
        warn("No news CSV - generating approximate recurring events")
        gen = approximate_generate_events(equity_series.index.min(), equity_series.index.max())
        info(f"Generated {len(gen)} approximate events")
        return gen

def load_vix(equity_series):
    p = Path(VIX_LOCAL_PATH)
    if p.exists():
        try:
            df = pd.read_csv(p)
            cols = [c.replace('<','').replace('>','').strip() for c in df.columns]
            df.columns = cols
            if 'DATE' in df.columns and 'TIME' in df.columns:
                df['time'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str), errors='coerce')
                df.set_index('time', inplace=True)
            elif 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], errors='coerce'); df.set_index('time', inplace=True)
            if 'CLOSE' in df.columns:
                df['VIX'] = df['CLOSE']
            elif 'Close' in df.columns:
                df['VIX'] = df['Close']
            else:
                df['VIX'] = df.select_dtypes(include=[np.number]).iloc[:,0]
            df.index = pd.to_datetime(df.index)
            df = df[['VIX']].dropna()
            info("Loaded local VIX dataset")
            return df
        except Exception as e:
            warn(f"Failed to load local VIX: {e}")
    # fetch from Yahoo hourly
    try:
        info("Downloading VIX from Yahoo Finance (1h)")
        v = yf.download("^VIX", start=equity_series.index.min().date(), end=(equity_series.index.max()+timedelta(days=1)).date(), interval='1h', progress=False)
        if v is None or v.empty:
            warn("Yahoo VIX returned empty")
            return None
        v['VIX'] = v['Close']
        v.index = pd.to_datetime(v.index)
        return v[['VIX']]
    except Exception as e:
        warn(f"VIX download failed: {e}")
        return None

# ---------------- core metrics ----------------
def open_trades_at(trades_df, t):
    return trades_df[(trades_df['Open time'] <= t) & (trades_df['Close time'] > t)]

def pnl_over_window(equity_series, t, minutes):
    t0 = t - pd.Timedelta(minutes=minutes)
    t1 = t + pd.Timedelta(minutes=minutes)
    try:
        v0 = equity_series.asof(t0)
        v1 = equity_series.asof(t1)
        if pd.isna(v0) or pd.isna(v1):
            return np.nan
        return float(v1 - v0)
    except Exception:
        return np.nan

def baseline_pnls(equity_series, event_time, minutes, events_times, n_samples=60, days_back=120):
    pnls = []
    start_search = event_time - pd.Timedelta(days=days_back)
    end_search = event_time + pd.Timedelta(days=days_back)
    all_days = pd.date_range(start_search.date(), end_search.date(), freq='D')
    event_set = set(pd.to_datetime(events_times))
    for d in all_days:
        cand = datetime.combine(d.date(), event_time.time())
        bad = False
        for et in event_set:
            if abs((cand - et).total_seconds()) <= (minutes * 60):
                bad = True; break
        if bad:
            continue
        p = pnl_over_window(equity_series, cand, minutes)
        if not pd.isna(p):
            pnls.append(p)
        if len(pnls) >= n_samples:
            break
    return np.array(pnls)

# ---------------- build dataset ----------------
def build_event_dataset(events_df, equity_series, trades_df, vix_df=None, windows=WINDOWS_MINUTES):
    rows = []
    events_times = list(pd.to_datetime(events_df['time']))
    for _, ev in events_df.iterrows():
        t = pd.to_datetime(ev['time'])
        etype = ev.get('event_type', 'OTHER')
        for w in windows:
            pnl = pnl_over_window(equity_series, t, w)
            open_ct = len(open_trades_at(trades_df, t))
            baseline = baseline_pnls(equity_series, t, w, events_times, n_samples=60, days_back=120)
            baseline_mean = float(np.nanmean(baseline)) if baseline.size>0 else np.nan
            baseline_q05 = float(np.nanpercentile(baseline, 5)) if baseline.size>0 else np.nan
            vix_mean = np.nan
            if vix_df is not None:
                try:
                    vix_slice = vix_df['VIX'].reindex(pd.date_range(t - timedelta(minutes=w), t + timedelta(minutes=w), freq='1min'), method='ffill').dropna()
                    if not vix_slice.empty:
                        vix_mean = float(vix_slice.mean())
                except Exception:
                    vix_mean = np.nan
            rows.append({
                'event_time': t, 'event_type': etype, 'window_min': w,
                'pnl': pnl, 'open_trades': open_ct, 'vix_mean': vix_mean,
                'baseline_mean': baseline_mean, 'baseline_q05': baseline_q05
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / 'ml_dataset_news_events.csv', index=False)
    info(f"Saved ml_dataset_news_events.csv rows={len(df)}")
    return df

# ---------------- modeling ----------------
def train_models(df):
    df_clean = df.dropna(subset=['pnl']).copy()
    if df_clean.empty:
        warn("No rows with pnl -> abort ML")
        return None
    X = df_clean[['open_trades','vix_mean']].fillna(0.0)
    y = df_clean['pnl'].astype(float)
    # model for mean (squared_error) and quantile (alpha=0.05)
    model_mean = GradientBoostingRegressor(loss='squared_error', n_estimators=200, random_state=RANDOM_STATE)
    model_q05 = GradientBoostingRegressor(loss='quantile', alpha=0.05, n_estimators=200, random_state=RANDOM_STATE)
    pipe_mean = Pipeline([('scaler', StandardScaler()), ('gbr', model_mean)])
    pipe_q05 = Pipeline([('scaler', StandardScaler()), ('gbr', model_q05)])
    # CV evaluation (robust)
    tscv = TimeSeriesSplit(n_splits=3)
    try:
        cv_mean = cross_validate(pipe_mean, X, y, cv=tscv, scoring='neg_mean_absolute_error', return_train_score=False, error_score='raise')
        cv_q = cross_validate(pipe_q05, X, y, cv=tscv, scoring='neg_mean_absolute_error', return_train_score=False, error_score='raise')
        info(f"CV MAE mean-model: {-np.mean(cv_mean['test_score']):.2f}")
        info(f"CV MAE q05-model: {-np.mean(cv_q['test_score']):.2f}")
    except Exception as e:
        warn(f"CV failed: {e}")
    # fit full models (in try/except to avoid crashing)
    try:
        pipe_mean.fit(X, y)
        pipe_q05.fit(X, y)
        joblib.dump(pipe_mean, OUTPUT_DIR / 'event_model_mean.joblib')
        joblib.dump(pipe_q05, OUTPUT_DIR / 'event_model_q05.joblib')
        info("Saved models: event_model_mean.joblib, event_model_q05.joblib")
    except Exception as e:
        warn(f"Model fit/save failed: {e}")
        return None
    return {'pipe_mean': pipe_mean, 'pipe_q05': pipe_q05, 'X_cols': X.columns.tolist()}

# ---------------- recommendations ----------------
def make_recommendations(df, models):
    recs = []
    if df.empty:
        warn("Empty df for recommendations")
        return recs
    out_rows = []
    for (etype, w), grp in df.groupby(['event_type','window_min']):
        mean_pnl = float(grp['pnl'].mean(skipna=True)) if not grp['pnl'].isna().all() else np.nan
        q05_emp = float(np.percentile(grp['pnl'].dropna(), 5)) if grp['pnl'].dropna().size>0 else np.nan
        vix_mean = float(grp['vix_mean'].mean(skipna=True)) if 'vix_mean' in grp.columns else np.nan
        baseline_mean = float(grp['baseline_mean'].mean(skipna=True)) if 'baseline_mean' in grp.columns else np.nan
        expected = np.nan; worst_q05 = np.nan
        if models is not None:
            X_pred = grp[['open_trades','vix_mean']].fillna(0.0)
            try:
                expected_vals = models['pipe_mean'].predict(X_pred)
                q05_vals = models['pipe_q05'].predict(X_pred)
                expected = float(np.mean(expected_vals))
                worst_q05 = float(np.mean(q05_vals))
            except Exception as e:
                warn(f"Prediction failed for {etype} w={w}: {e}")
        action = 'OK'; reason = ''
        check_mean = expected if not math.isnan(expected) else mean_pnl
        check_q05 = worst_q05 if not math.isnan(worst_q05) else q05_emp
        if not math.isnan(check_mean) and check_mean <= STOP_MEAN_PNL:
            action = 'STOP'
            reason = f'expected mean pnl {check_mean:.2f} <= {STOP_MEAN_PNL}'
        elif not math.isnan(check_q05) and check_q05 <= STOP_WORST_Q05:
            action = 'STOP'
            reason = f'predicted worst q05 {check_q05:.2f} <= {STOP_WORST_Q05}'
        elif not math.isnan(check_mean) and check_mean <= REDUCE_MEAN_PNL:
            action = 'REDUCE'
            reason = f'expected mean pnl {check_mean:.2f} <= {REDUCE_MEAN_PNL}'
        elif not math.isnan(check_q05) and check_q05 <= REDUCE_WORST_Q05:
            action = 'REDUCE'
            reason = f'predicted worst q05 {check_q05:.2f} <= {REDUCE_WORST_Q05}'
        out_rows.append({
            'event_type': etype, 'window_min': w, 'emp_mean_pnl': mean_pnl, 'emp_q05': q05_emp,
            'pred_mean_pnl': expected, 'pred_q05': check_q05, 'vix_mean': vix_mean,
            'baseline_mean': baseline_mean, 'action': action, 'reason': reason
        })
    rec_df = pd.DataFrame(out_rows)
    rec_df.to_csv(OUTPUT_DIR / 'news_recommendations_summary.csv', index=False)
    with open(OUTPUT_DIR / 'news_recommendations.txt', 'w', encoding='utf-8') as f:
        f.write("=== News Impact Recommendations ===\n\n")
        for _, r in rec_df.iterrows():
            f.write(f"Event: {r['event_type']} | Window ±{int(r['window_min'])}m\n")
            f.write(f"  Empirical mean PnL: {r['emp_mean_pnl']:.2f} | Empirical 5%: {r['emp_q05']:.2f}\n")
            f.write(f"  Predicted mean PnL: {r['pred_mean_pnl'] if not math.isnan(r['pred_mean_pnl']) else 'NA'} | Predicted 5%: {r['pred_q05'] if not math.isnan(r['pred_q05']) else 'NA'}\n")
            f.write(f"  Baseline mean (no-event): {r['baseline_mean']:.2f} | VIX mean: {r['vix_mean']:.2f}\n")
            f.write(f"  ACTION: {r['action']}  REASON: {r['reason']}\n\n")
    info("Saved news_recommendations.txt and news_recommendations_summary.csv")
    return rec_df

# ---------------- plotting ----------------
def plot_equity_with_events_per_year(equity_series, events_df):
    eq = equity_series.dropna()
    if eq.empty:
        warn("Empty equity - skipping plots")
        return
    years = sorted(set(eq.index.year))
    for y in years:
        st = datetime(y,1,1); en = datetime(y,12,31,23,59,59)
        subset = eq.loc[(eq.index>=st) & (eq.index<=en)]
        if subset.empty: continue
        plt.figure(figsize=(14,6))
        plt.plot(subset.index, subset.values, lw=1.5)
        evs = events_df[(events_df['time']>=st) & (events_df['time']<=en)]
        for _, ev in evs.iterrows():
            t = pd.to_datetime(ev['time']); et = ev.get('event_type','OTHER')
            c = EVENT_COLORS.get(et.upper(), 'purple')
            plt.axvline(t, color=c, lw=1.2, alpha=0.9)
            yloc = subset.min() + (subset.max() - subset.min())*0.02
            plt.text(t, yloc, et, rotation=90, fontsize=8, color=c, va='bottom', ha='center')
        plt.title(f'Equity {y} with news events')
        plt.ylabel('Equity (€)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        fname = OUTPUT_DIR / f'equity_{y}_with_events.png'
        plt.savefig(fname); plt.close()
        info(f"Saved {fname}")

# ---------------- main pipeline ----------------
def main():
    info("Starting NEWS impact pipeline (predictive + quantile)")
    trades = load_trades()
    equity = load_equity_series()
    news = load_news_or_generate(equity)
    if news is None or news.empty:
        warn("No news events found -> abort")
        return
    vix = load_vix(equity)
    df_events = build_event_dataset(news, equity, trades, vix, windows=WINDOWS_MINUTES)
    # aggregate simple stats
    try:
        agg = df_events.groupby(['event_type','window_min']).agg({'pnl':['mean','median','count'],'vix_mean':'mean','baseline_mean':'mean'}).reset_index()
        agg.columns = ['event_type','window_min','pnl_mean','pnl_median','count','vix_mean','baseline_mean']
        agg.to_csv(OUTPUT_DIR / 'aggregate_stats_by_event.csv', index=False)
        info("Saved aggregate_stats_by_event.csv")
    except Exception as e:
        warn(f"Failed to aggregate/save stats: {e}")
    # plot equity per year with events overlay
    plot_equity_with_events_per_year(equity, news)
    # ML models
    models = train_models(df_events)
    # generate recommendations
    recs = make_recommendations(df_events, models)
    # save summary JSON
    summary = {
        'n_events': int(len(news)),
        'n_ml_rows': int(len(df_events)),
        'recs_count': int(len(recs))
    }
    with open(OUTPUT_DIR / 'news_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    info("News impact pipeline finished. Outputs in: " + str(OUTPUT_DIR))

if __name__ == '__main__':
    main()
