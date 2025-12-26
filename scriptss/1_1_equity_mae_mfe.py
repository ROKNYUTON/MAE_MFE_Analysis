import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# --- CONFIGURAZIONE ---
REPORT_PATH = 'data/reports/multi_asset_report.csv'
DATASETS_FOLDER = 'data/datasets'
INITIAL_BALANCE = 100000.0

CONTRACT_SIZE = {
    'GOLD.pro': 100, 'USDJPY.pro': 1000, 'US100.pro': 20,
    'US500.pro': 50, 'DE30.pro': 25,
}

def load_price_data(symbol, resample=None):
    """Carica dati M5. Se resample='1H', ottimizza per il grafico generale."""
    path = os.path.join(DATASETS_FOLDER, f"{symbol}_M5_2020_2025.csv")
    if not os.path.exists(path): return None
    
    df = pd.read_csv(path, sep='\t')
    df.columns = [c.replace('<', '').replace('>', '') for c in df.columns]
    df['dt'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y.%m.%d %H:%M:%S')
    df.set_index('dt', inplace=True)
    df = df[['OPEN', 'HIGH', 'LOW', 'CLOSE']].astype(float)
    
    if resample:
        return df.resample(resample).agg({
            'OPEN': 'first', 'HIGH': 'max', 'LOW': 'min', 'CLOSE': 'last'
        }).ffill()
    return df

def generate_crash_report(crash_time, df_trades, portfolio, output_dir, is_real_min=False):
    """Genera grafico zoomato, CSV posizione, grafici per trade, e verifica equity reale M5."""
    label = "REAL_MIN" if is_real_min else "CRASH"
    print(f"\n🕵️‍♂️ AVVIO DEEP DIVE SUL {label}: {crash_time}")
    
    # 1. Grafico Zoomato (2 settimane prima e dopo) - invariato
    start_zoom = crash_time - pd.Timedelta(weeks=2)
    end_zoom = crash_time + pd.Timedelta(weeks=2)
    
    zoom_df = portfolio[(portfolio.index >= start_zoom) & (portfolio.index <= end_zoom)]
    
    if not zoom_df.empty:
        plt.figure(figsize=(14, 7))
        plt.plot(zoom_df.index, zoom_df['balance'], color='blue', lw=2, label='Balance (Chiuso)')
        plt.plot(zoom_df.index, zoom_df['Equity_Low'], color='red', lw=1.5, label='Equity Minima (Approx)')
        plt.fill_between(zoom_df.index, zoom_df['Equity_Low'], zoom_df['Equity_High'], color='gray', alpha=0.2)
        
        crash_val = portfolio.loc[crash_time]['Equity_Low'] if crash_time in portfolio.index else zoom_df['Equity_Low'].min()
        plt.scatter([crash_time], [crash_val], color='black', zorder=5, s=100, label=f'{label} EVENT')
        
        plt.title(f'Deep Dive {label}: {crash_time.date()} (+/- 2 Settimane)', fontsize=14, fontweight='bold')
        plt.ylabel('Equity (€)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        
        zoom_path = f"{output_dir}/{label}_ZOOM_{crash_time.strftime('%Y-%m-%d')}.png"
        plt.savefig(zoom_path)
        print(f"✅ Grafico Zoom salvato: {zoom_path}")
        plt.close()

    # 2. Analisi Posizioni Aperte con MAE/MFE - invariato, ma con label
    open_trades = df_trades[
        (df_trades['Open time'] <= crash_time) & 
        (df_trades['Close time'] > crash_time)
    ].copy()
    
    details = []
    print(f"   Analisi di {len(open_trades)} trade aperti nel momento del {label}...")
    
    for _, trade in open_trades.iterrows():
        s = trade['Symbol']
        m5_data = load_price_data(s, resample=None) 
        if m5_data is None: continue
        
        # Prezzo al crash (nearest CLOSE)
        idx = m5_data.index.get_indexer([crash_time], method='nearest')[0]
        price_at_crash = m5_data.iloc[idx]['CLOSE']

        entry = trade['Open price']
        size = trade['Size'] * CONTRACT_SIZE.get(s, 100)
        
        if 'Buy' in trade['Type']:
            floating_pl = (price_at_crash - entry) * size
        else:
            floating_pl = (entry - price_at_crash) * size
        
        # Calcolo MAE/MFE full lifetime del trade
        trade_mask = (m5_data.index >= trade['Open time']) & (m5_data.index <= trade['Close time'])
        if trade_mask.any():
            prices = m5_data.loc[trade_mask]
            if 'Buy' in trade['Type']:
                mae_price = prices['LOW'].min()
                mfe_price = prices['HIGH'].max()
                mae_pl = (mae_price - entry) * size
                mfe_pl = (mfe_price - entry) * size
            else:
                mae_price = prices['HIGH'].max()
                mfe_price = prices['LOW'].min()
                mae_pl = (entry - mae_price) * size
                mfe_pl = (entry - mfe_price) * size
        else:
            mae_pl = mfe_pl = 0.0
            mae_price = mfe_price = entry
        
        details.append({
            'Ticket': trade.get('Ticket', 'N/A'),
            'Symbol': s,
            'Type': trade['Type'],
            'Open Time': trade['Open time'],
            'Entry Price': entry,
            'Price @ Crash': price_at_crash,
            'Floating P&L (€)': round(floating_pl, 2),
            'Size': trade['Size'],
            'MAE (€)': round(mae_pl, 2),
            'MFE (€)': round(mfe_pl, 2)
        })
        
        # Grafico per posizione: +/-1 settimana, limitato al trade
        start_plot = max(trade['Open time'], crash_time - pd.Timedelta(weeks=1))
        end_plot = min(trade['Close time'], crash_time + pd.Timedelta(weeks=1))
        plot_mask = (m5_data.index >= start_plot) & (m5_data.index <= end_plot)
        if plot_mask.any():
            prices_plot = m5_data.loc[plot_mask]
            plt.figure(figsize=(10, 5))
            plt.plot(prices_plot.index, prices_plot['CLOSE'], label='Close Price')
            plt.axhline(entry, color='green', ls='--', label='Entry')
            plt.axhline(mae_price, color='red', ls='--', label='MAE Price')
            plt.axhline(mfe_price, color='blue', ls='--', label='MFE Price')
            plt.axvline(crash_time, color='black', ls='-', label=f'{label} Time')
            if trade['Open time'] >= start_plot: plt.axvline(trade['Open time'], color='lime', ls=':', label='Open')
            if trade['Close time'] <= end_plot: plt.axvline(trade['Close time'], color='orange', ls=':', label='Close')
            plt.title(f'{s} - Trade {trade.get("Ticket", "N/A")} +/-1 Settimana dal {label}')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            trade_path = f"{output_dir}/{label}_TRADE_{trade.get('Ticket', 'N/A')}_{crash_time.strftime('%Y-%m-%d')}.png"
            plt.savefig(trade_path)
            print(f"✅ Grafico Trade salvato: {trade_path}")
            plt.close()
    
    if details:
        crash_df = pd.DataFrame(details).sort_values('Floating P&L (€)')
        csv_path = f"{output_dir}/{label}_POSITIONS_{crash_time.strftime('%Y-%m-%d')}.csv"
        crash_df.to_csv(csv_path, index=False)
        print(f"✅ CSV Posizioni salvato: {csv_path}")
        print("\nDettagli Posizioni (con MAE/MFE):\n", crash_df.to_string())
        print(f"\nSum Floating P&L al {label}: {crash_df['Floating P&L (€)'].sum():.2f} €")

    # 3. Verifica Equity Reale a M5 (+/-2 Settimane ora, per catturare meglio)
    print(f"\n🔍 Calcolo Equity Reale M5 (+/-2 Settimane dal {label})...")
    start_high = crash_time - pd.Timedelta(weeks=2)
    end_high = crash_time + pd.Timedelta(weeks=2)
    
    open_in_period = df_trades[(df_trades['Open time'] <= end_high) & (df_trades['Close time'] > start_high)]
    symbols_in_period = open_in_period['Symbol'].unique()
    
    m5_db = {s: load_price_data(s, resample=None) for s in symbols_in_period if load_price_data(s, resample=None) is not None}
    
    if not m5_db:
        print("❌ Nessun dato M5 disponibile per il periodo.")
        return
    
    min_time = min(df.index.min() for df in m5_db.values())
    max_time = max(df.index.max() for df in m5_db.values())
    timeline_m5 = pd.date_range(max(start_high, min_time), min(end_high, max_time), freq='5min')
    
    # Balance su M5
    balance_changes = df_trades[['Close time', 'Profit/Loss (Global)']].copy().sort_values('Close time')
    balance_changes['Cumulative_Balance'] = INITIAL_BALANCE + balance_changes['Profit/Loss (Global)'].cumsum()
    
    high_res_port = pd.DataFrame(index=timeline_m5)
    high_res_port.index.name = 'time'
    merged_high = pd.merge_asof(high_res_port, balance_changes.rename(columns={'Close time': 'time'}), on='time', direction='backward')
    merged_high.set_index('time', inplace=True)
    merged_high['balance'] = merged_high['Cumulative_Balance'].fillna(INITIAL_BALANCE)
    merged_high['equity'] = merged_high['balance'].copy()  # Su CLOSE
    merged_high['equity_low'] = merged_high['balance'].copy()  # Su LOW (nuovo, approx min intra-bar)

    # Aggiungi floating P&L reale (CLOSE) e low (LOW)
    for _, trade in df_trades.iterrows():
        s = trade['Symbol']
        if s not in m5_db: continue
        mask = (merged_high.index >= trade['Open time']) & (merged_high.index <= trade['Close time'])
        if not mask.any(): continue
        p = m5_db[s].reindex(merged_high.index[mask], method='ffill')
        mult = trade['Size'] * CONTRACT_SIZE.get(s, 100)
        entry = trade['Open price']
        if 'Buy' in trade['Type']:
            pl_close = (p['CLOSE'] - entry) * mult
            pl_low = (p['LOW'] - entry) * mult
        else:
            pl_close = (entry - p['CLOSE']) * mult
            pl_low = (entry - p['HIGH']) * mult  # Per Sell, worst è HIGH
        merged_high.loc[mask, 'equity'] += pl_close
        merged_high.loc[mask, 'equity_low'] += pl_low
    
    # Risultati
    peak = merged_high['equity'].cummax()
    dd_real = merged_high['equity'] - peak
    min_equity = merged_high['equity'].min()
    time_min = merged_high['equity'].idxmin()
    max_dd_real = dd_real.min()
    
    peak_low = merged_high['equity_low'].cummax()
    dd_low = merged_high['equity_low'] - peak_low
    min_equity_low = merged_high['equity_low'].min()
    time_min_low = merged_high['equity_low'].idxmin()
    max_dd_low = dd_low.min()
    
    print(f"✅ Equity Minima Reale (M5 CLOSE): {min_equity:.2f} € a {time_min}")
    print(f"✅ Max DD Reale (su CLOSE): {max_dd_real:.2f} €")
    
    print(f"✅ Equity Minima Approx (M5 LOW): {min_equity_low:.2f} € a {time_min_low}")
    print(f"✅ Max DD Approx (su LOW M5): {max_dd_low:.2f} €")
    
    # Equity al tempo specifico (fix traceback)
    nearest_idx = merged_high.index.get_indexer([crash_time], method='nearest')[0]
    equity_at_crash = merged_high.iloc[nearest_idx]['equity']
    equity_low_at_crash = merged_high.iloc[nearest_idx]['equity_low']
    print(f"Equity Reale al {label} (nearest): {equity_at_crash:.2f} €")
    print(f"Equity Approx Low al {label} (nearest): {equity_low_at_crash:.2f} €")
    print(f"Confronto con Hourly Approx Equity_Low: {portfolio.loc[crash_time]['Equity_Low'] if crash_time in portfolio.index else 'N/A'}")

def main():
    if not os.path.exists(REPORT_PATH):
        print("❌ Report non trovato")
        return

    # Caricamento - invariato
    print("🚀 Inizio calcolo Equity Cumulativa Reale...")
    df_trades = pd.read_csv(REPORT_PATH)
    df_trades.columns = [c.strip() for c in df_trades.columns]
    df_trades['Open time'] = pd.to_datetime(df_trades['Open time'], dayfirst=True)
    df_trades['Close time'] = pd.to_datetime(df_trades['Close time'], dayfirst=True)
    
    all_symbols = df_trades['Symbol'].unique()
    prices_db = {}
    print("⏳ Caricamento Dataset Prezzi (1H)...")
    for s in all_symbols:
        data = load_price_data(s, resample='1H')
        if data is not None: prices_db[s] = data

    # Timeline e Balance - invariato
    start_date = df_trades['Open time'].min()
    end_date = df_trades['Close time'].max()
    timeline = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    balance_changes = df_trades[['Close time', 'Profit/Loss (Global)']].copy().sort_values('Close time')
    balance_changes['Cumulative_Balance'] = INITIAL_BALANCE + balance_changes['Profit/Loss (Global)'].cumsum()
    
    portfolio_base = pd.DataFrame(index=timeline)
    portfolio_base.index.name = 'Close time'
    merged = pd.merge_asof(portfolio_base, balance_changes, on='Close time', direction='backward')
    merged.set_index('Close time', inplace=True)
    merged['Cumulative_Balance'] = merged['Cumulative_Balance'].fillna(INITIAL_BALANCE)
    
    portfolio = merged[['Cumulative_Balance']].rename(columns={'Cumulative_Balance': 'balance'})
    portfolio['unrealized_low'] = 0.0
    portfolio['unrealized_high'] = 0.0

    # Calcolo Equity Approx - invariato
    print("🔄 Calcolo Equity Fluttuante (Approx)...")
    for _, trade in df_trades.iterrows():
        s = trade['Symbol']
        if s not in prices_db: continue
        mask = (portfolio.index >= trade['Open time']) & (portfolio.index <= trade['Close time'])
        if not mask.any(): continue
        p = prices_db[s].reindex(portfolio.index[mask], method='ffill')
        mult = trade['Size'] * CONTRACT_SIZE.get(s, 100)
        entry = trade['Open price']
        if 'Buy' in trade['Type']:
            portfolio.loc[mask, 'unrealized_low'] += (p['LOW'] - entry) * mult
            portfolio.loc[mask, 'unrealized_high'] += (p['HIGH'] - entry) * mult
        else:
            portfolio.loc[mask, 'unrealized_low'] += (entry - p['HIGH']) * mult
            portfolio.loc[mask, 'unrealized_high'] += (entry - p['LOW']) * mult

    portfolio['Equity_Low'] = portfolio['balance'] + portfolio['unrealized_low']
    portfolio['Equity_High'] = portfolio['balance'] + portfolio['unrealized_high']

    # Max DD Approx
    peak = portfolio['Equity_Low'].cummax()
    drawdown = portfolio['Equity_Low'] - peak
    max_dd_val = drawdown.min()
    max_dd_time = drawdown.idxmin()
    
    print("\n" + "="*40)
    print(f"⚠️ MAX DRAWDOWN (Equity Approx Hourly): {max_dd_val:.2f} €")
    print(f"📅 Data Evento: {max_dd_time}")
    print("="*40)
    if max_dd_time in portfolio.index:
        print(f"Portfolio al Crash:\n{portfolio.loc[max_dd_time]}")
        print(f"Peak fino al Crash: {peak.loc[max_dd_time]:.2f} €")
        print(f"DD Calcolato: {drawdown.loc[max_dd_time]:.2f} €")

    # Salvataggio e Deep Dive su approx time
    out_dir = 'data/reports/comparative_analysis'
    os.makedirs(out_dir, exist_ok=True)
    generate_crash_report(max_dd_time, df_trades, portfolio, out_dir)
    
    # Calcola min reale da deep dive (per evitare rerun full), ma qui assumiamo dal primo run
    # Se vuoi, aggiungi logica per estrarre time_min dal primo deep dive e run su quello
    # Per ora, run deep dive anche su un tempo hardcoded se noto, es. 2024-07-05 15:01:03
    real_min_time = pd.to_datetime('2024-07-05 15:01:03')  # Dal tuo output; cambialo se serve
    if real_min_time != max_dd_time:
        generate_crash_report(real_min_time, df_trades, portfolio, out_dir, is_real_min=True)
    
    # Grafici Annuali con DD Evidenziati - invariato
    years = portfolio.index.year.unique()
    os.makedirs(f"{out_dir}/yearly_equity", exist_ok=True)
    
    print("\n🎨 Generazione Grafici Annuali Cumulativi...")
    for year in years:
        df_y = portfolio[portfolio.index.year == year]
        if df_y.empty: continue
        
        # Calcolo Max DD per anno (su approx)
        peak_y = df_y['Equity_Low'].cummax()
        dd_y = df_y['Equity_Low'] - peak_y
        max_dd_equity = dd_y.min()
        time_dd_equity = dd_y.idxmin()
        
        peak_b = df_y['balance'].cummax()
        dd_b = df_y['balance'] - peak_b
        max_dd_balance = dd_b.min()
        time_dd_balance = dd_b.idxmin()
        
        print(f"Anno {year}: Max DD Equity Approx {max_dd_equity:.2f} € a {time_dd_equity}")
        print(f"Anno {year}: Max DD Balance {max_dd_balance:.2f} € a {time_dd_balance}")
        
        plt.figure(figsize=(14, 7))
        plt.fill_between(df_y.index, df_y['Equity_Low'], df_y['Equity_High'], color='gray', alpha=0.2, label='Intraday Volatility')
        plt.plot(df_y.index, df_y['balance'], color='blue', lw=2, label='Closed Balance')
        plt.plot(df_y.index, df_y['Equity_Low'], color='red', lw=1, alpha=0.9, label='Worst Equity (Approx)')
        
        # Evidenzia Max DD
        plt.scatter([time_dd_equity], [df_y.loc[time_dd_equity]['Equity_Low']], color='darkred', s=100, label='Max DD Equity', zorder=5)
        plt.scatter([time_dd_balance], [df_y.loc[time_dd_balance]['balance']], color='darkblue', s=100, label='Max DD Balance', zorder=5)
        
        # Annotazioni
        plt.annotate(f'Max DD Equity: {max_dd_equity:.2f} €\n{time_dd_equity.date()}', 
                     xy=(time_dd_equity, df_y.loc[time_dd_equity]['Equity_Low']), xytext=(10, 10), 
                     textcoords='offset points', arrowprops=dict(arrowstyle='->', color='red'))
        plt.annotate(f'Max DD Balance: {max_dd_balance:.2f} €\n{time_dd_balance.date()}', 
                     xy=(time_dd_balance, df_y.loc[time_dd_balance]['balance']), xytext=(10, 10), 
                     textcoords='offset points', arrowprops=dict(arrowstyle='->', color='blue'))
        
        plt.title(f'Performance {year} (Cumulativa Reale)', fontsize=14)
        plt.ylabel('Euro')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        plt.tight_layout()
        plt.savefig(f"{out_dir}/yearly_equity/Equity_{year}.png")
        plt.close()
        print(f"✅ Salvato Anno {year}")

if __name__ == "__main__":
    main()