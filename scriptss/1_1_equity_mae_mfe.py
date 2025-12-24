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

def generate_crash_report(crash_time, df_trades, portfolio, output_dir):
    """Genera grafico zoomato e CSV delle posizioni durante il crash"""
    print(f"\n🕵️‍♂️ AVVIO DEEP DIVE SUL CRASH: {crash_time}")
    
    # 1. Grafico Zoomato (2 settimane prima e dopo)
    start_zoom = crash_time - pd.Timedelta(weeks=2)
    end_zoom = crash_time + pd.Timedelta(weeks=2)
    
    zoom_df = portfolio[(portfolio.index >= start_zoom) & (portfolio.index <= end_zoom)]
    
    if not zoom_df.empty:
        plt.figure(figsize=(14, 7))
        plt.plot(zoom_df.index, zoom_df['balance'], color='blue', lw=2, label='Balance (Chiuso)')
        plt.plot(zoom_df.index, zoom_df['Equity_Low'], color='red', lw=1.5, label='Equity Minima (Reale)')
        plt.fill_between(zoom_df.index, zoom_df['Equity_Low'], zoom_df['Equity_High'], color='gray', alpha=0.2)
        
        # Evidenzia il punto esatto del crash
        crash_val = portfolio.loc[crash_time]['Equity_Low'] if crash_time in portfolio.index else zoom_df['Equity_Low'].min()
        plt.scatter([crash_time], [crash_val], color='black', zorder=5, s=100, label='CRASH EVENT')
        
        plt.title(f'Deep Dive Crash: {crash_time.date()} (+/- 2 Settimane)', fontsize=14, fontweight='bold')
        plt.ylabel('Equity (€)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        
        zoom_path = f"{output_dir}/CRASH_ZOOM_{crash_time.strftime('%Y-%m-%d')}.png"
        plt.savefig(zoom_path)
        print(f"✅ Grafico Zoom salvato: {zoom_path}")
        plt.close()

    # 2. CSV Analisi Posizioni Aperte
    open_trades = df_trades[
        (df_trades['Open time'] <= crash_time) & 
        (df_trades['Close time'] > crash_time)
    ].copy()
    
    details = []
    print(f"   Analisi di {len(open_trades)} trade aperti nel momento del crash...")
    
    for _, trade in open_trades.iterrows():
        s = trade['Symbol']
        # Carica M5 per precisione chirurgica
        m5_data = load_price_data(s, resample=None) 
        
        if m5_data is None: continue
        
        # Prezzo esatto al minuto del crash
        try:
            # Trova il prezzo più vicino al crash time
            idx = m5_data.index.get_indexer([crash_time], method='nearest')[0]
            price_at_crash = m5_data.iloc[idx]['CLOSE']
        except:
            price_at_crash = trade['Open price']

        entry = trade['Open price']
        size = trade['Size'] * CONTRACT_SIZE.get(s, 100)
        
        # Calcolo P&L Latente al momento del crash
        if 'Buy' in trade['Type']:
            floating_pl = (price_at_crash - entry) * size
        else:
            floating_pl = (entry - price_at_crash) * size
            
        details.append({
            'Ticket': trade.get('Ticket', 'N/A'),
            'Symbol': s,
            'Type': trade['Type'],
            'Open Time': trade['Open time'],
            'Entry Price': entry,
            'Price @ Crash': price_at_crash,
            'Floating P&L (€)': round(floating_pl, 2),
            'Size': trade['Size']
        })
    
    if details:
        crash_csv = pd.DataFrame(details).sort_values('Floating P&L (€)')
        csv_path = f"{output_dir}/CRASH_POSITIONS_{crash_time.strftime('%Y-%m-%d')}.csv"
        crash_csv.to_csv(csv_path, index=False)
        print(f"✅ CSV Posizioni salvato: {csv_path}")

def main():
    if not os.path.exists(REPORT_PATH):
        print("❌ Report non trovato")
        return

    # 1. Caricamento e Preparazione
    print("🚀 Inizio calcolo Equity Cumulativa Reale...")
    df_trades = pd.read_csv(REPORT_PATH)
    df_trades.columns = [c.strip() for c in df_trades.columns]
    df_trades['Open time'] = pd.to_datetime(df_trades['Open time'], dayfirst=True)
    df_trades['Close time'] = pd.to_datetime(df_trades['Close time'], dayfirst=True)
    
    # 2. Carica Prezzi in RAM
    all_symbols = df_trades['Symbol'].unique()
    prices_db = {}
    print("⏳ Caricamento Dataset Prezzi (1H)...")
    for s in all_symbols:
        data = load_price_data(s, resample='1H')
        if data is not None: prices_db[s] = data

    # 3. Creazione Timeline e Balance Cumulativo (CORRETTO)
    start_date = df_trades['Open time'].min()
    end_date = df_trades['Close time'].max()
    timeline = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Crea un dataframe temporaneo con i cambi di balance
    balance_changes = df_trades[['Close time', 'Profit/Loss (Global)']].copy()
    balance_changes = balance_changes.sort_values('Close time')
    
    # Calcola il balance comulativo su ogni chiusura trade
    balance_changes['Cumulative_Balance'] = INITIAL_BALANCE + balance_changes['Profit/Loss (Global)'].cumsum()
    
    # Mappa questi valori sulla timeline oraria usando merge_asof (riempie in avanti)
    # Questo assicura che la linea blu salga e scenda e rimanga stabile tra un trade e l'altro
    portfolio_base = pd.DataFrame(index=timeline)
    portfolio_base.index.name = 'Close time'
    
    # Uniamo i dati: per ogni ora della timeline, prendiamo l'ultimo balance noto
    merged = pd.merge_asof(portfolio_base, balance_changes, on='Close time', direction='backward')
    merged.set_index('Close time', inplace=True)
    merged['Cumulative_Balance'] = merged['Cumulative_Balance'].fillna(INITIAL_BALANCE)
    
    portfolio = merged[['Cumulative_Balance']].rename(columns={'Cumulative_Balance': 'balance'})
    portfolio['unrealized_low'] = 0.0
    portfolio['unrealized_high'] = 0.0

    # 4. Calcolo Fluttuante (Equity Reale)
    print("🔄 Calcolo Equity Fluttuante (MAE/MFE)...")
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

    # 5. Individuazione Max Drawdown Assoluto
    # Il DD si calcola sul picco massimo raggiunto dall'Equity LOW
    peak = portfolio['Equity_Low'].cummax()
    drawdown = portfolio['Equity_Low'] - peak
    max_dd_val = drawdown.min()
    max_dd_time = drawdown.idxmin()
    
    print("\n" + "="*40)
    print(f"⚠️ MAX DRAWDOWN (Equity Based): {max_dd_val:.2f} €")
    print(f"📅 Data Evento: {max_dd_time}")
    print("="*40)

    # 6. Salvataggio Report
    out_dir = 'data/reports/comparative_analysis'
    os.makedirs(out_dir, exist_ok=True)
    
    # Esegui Deep Dive sul Crash
    generate_crash_report(max_dd_time, df_trades, portfolio, out_dir)
    
    # 7. Grafici Annuali
    years = portfolio.index.year.unique()
    os.makedirs(f"{out_dir}/yearly_equity", exist_ok=True)
    
    print("\n🎨 Generazione Grafici Annuali Cumulativi...")
    for year in years:
        df_y = portfolio[portfolio.index.year == year]
        if df_y.empty: continue
        
        plt.figure(figsize=(14, 7))
        
        # Ombra Volatilità
        plt.fill_between(df_y.index, df_y['Equity_Low'], df_y['Equity_High'], 
                         color='gray', alpha=0.2, label='Intraday Volatility')
        
        # Balance (Linea Blu - Ora Cumulativa!)
        plt.plot(df_y.index, df_y['balance'], color='blue', lw=2, label='Closed Balance')
        
        # Equity Low (Linea Rossa - Drawdown Reale)
        plt.plot(df_y.index, df_y['Equity_Low'], color='red', lw=1, alpha=0.9, label='Worst Equity')
        
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