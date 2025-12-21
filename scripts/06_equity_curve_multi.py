import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta

# Config
REPORT_PATH = 'data/reports/multi_asset_report.csv'
DATASETS_FOLDER = 'data/datasets'
INITIAL_BALANCE_EUR = 100000.0
OUTPUT_PLOT = 'data/reports/equity_portafoglio_aggregata.png'

# Tassi mensili EUR/USD (come nel tuo codice)
RATES = {
    '2020-01': 1.1095, '2020-02': 1.0918, '2020-03': 1.1067, '2020-04': 1.0868, '2020-05': 1.0911,
    '2020-06': 1.1256, '2020-07': 1.1466, '2020-08': 1.1831, '2020-09': 1.1786, '2020-10': 1.1772,
    '2020-11': 1.1895, '2020-12': 1.2176,
    '2021-01': 1.2165, '2021-02': 1.2095, '2021-03': 1.1899, '2021-04': 1.1968, '2021-05': 1.2142,
    '2021-06': 1.2037, '2021-07': 1.1825, '2021-08': 1.1766, '2021-09': 1.1768, '2021-10': 1.1597,
    '2021-11': 1.1409, '2021-12': 1.1304,
    '2022-01': 1.1317, '2022-02': 1.1336, '2022-03': 1.1007, '2022-04': 1.0822, '2022-05': 1.0576,
    '2022-06': 1.0564, '2022-07': 1.0190, '2022-08': 1.0112, '2022-09': 0.9912, '2022-10': 0.9835,
    '2022-11': 1.0216, '2022-12': 1.0595,
    '2023-01': 1.0788, '2023-02': 1.0705, '2023-03': 1.0716, '2023-04': 1.0989, '2023-05': 1.0873,
    '2023-06': 1.0843, '2023-07': 1.1064, '2023-08': 1.0909, '2023-09': 1.0679, '2023-10': 1.0565,
    '2023-11': 1.0821, '2023-12': 1.0921,
    '2024-01': 1.0906, '2024-02': 1.0793, '2024-03': 1.0871, '2024-04': 1.0723, '2024-05': 1.0810,
    '2024-06': 1.0762, '2024-07': 1.0848, '2024-08': 1.1022, '2024-09': 1.1105, '2024-10': 1.0896,
    '2024-11': 1.0633, '2024-12': 1.0470,
    '2025-01': 1.0333, '2025-02': 1.0412, '2025-03': 1.0804, '2025-04': 1.1228, '2025-05': 1.1273,
    '2025-06': 1.1530, '2025-07': 1.1682, '2025-08': 1.1653, '2025-09': 1.1733, '2025-10': 1.1638,
    '2025-11': 1.1561, '2025-12': 1.1687,
}

CONTRACT_SIZE = {
    'GOLD.pro': 100,
    'USDJPY.pro': 1000,
    'US100.pro': 20,
    'US500.pro': 50,
    'DE30.pro': 25,
}

prices_cache = {}

def load_price_data(symbol):
    if symbol in prices_cache:
        return prices_cache[symbol]

    filename = f"{symbol}_M5_2020_2025.csv"
    path = os.path.join(DATASETS_FOLDER, filename)

    if not os.path.exists(path):
        print(f"⚠️ Dataset non trovato per {symbol}")
        prices_cache[symbol] = None
        return None

    print(f"🔄 Caricamento {symbol}...")
    df = pd.read_csv(path, sep='\t')
    df.columns = [c.replace('<', '').replace('>', '') for c in df.columns]
    df['Datetime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y.%m.%d %H:%M:%S')
    df.set_index('Datetime', inplace=True)
    df = df[['OPEN', 'HIGH', 'LOW', 'CLOSE']].astype(float)
    prices_cache[symbol] = df
    print(f"✅ {symbol} caricato")
    return df

def load_report():
    df = pd.read_csv(REPORT_PATH, sep=',')
    df.columns = [c.strip() for c in df.columns]
    df['Open time'] = pd.to_datetime(df['Open time'], format='%d.%m.%Y %H:%M:%S')
    df['Close time'] = pd.to_datetime(df['Close time'], format='%d.%m.%Y %H:%M:%S')
    df['Open price'] = df['Open price'].astype(float)
    df['Close price'] = df['Close price'].astype(float)
    df['Size'] = df['Size'].astype(float)
    df['Profit/Loss (Global)'] = df['Profit/Loss (Global)'].astype(float)
    print(f"✅ Report caricato: {len(df)} trade")
    return df

def calculate_drawdowns(series):
    peak = np.maximum.accumulate(series)
    drawdown_pct = (series - peak) / peak * 100
    drawdown_usd = series - peak
    return round(drawdown_pct.min(), 2), round(drawdown_usd.min(), 2)

def get_exchange_rate(date):
    """Ottieni il tasso di cambio per una data specifica"""
    month_key = date.strftime('%Y-%m')
    return RATES.get(month_key, 1.10)

def calculate_portfolio_equity(df_trades):
    """
    Calcola l'equity del portafoglio considerando:
    1. Balance (realized P&L)
    2. Equity con unrealized P&L (favorable/adverse exposure)
    """
    # Carica i dati dei prezzi per tutti i simboli
    symbols = df_trades['Symbol'].unique()
    for symbol in symbols:
        load_price_data(symbol)
    
    # Crea una timeline di riferimento (ogni 4 ore per efficienza)
    all_timestamps = pd.DatetimeIndex([])
    for symbol, df in prices_cache.items():
        if df is not None:
            # Campiona ogni 4 ore (12 candele M5 per ora * 4 = 48)
            sampled_df = df.resample('4H').agg({
                'OPEN': 'first',
                'HIGH': 'max',
                'LOW': 'min',
                'CLOSE': 'last'
            }).dropna()
            all_timestamps = all_timestamps.union(sampled_df.index)
    
    all_timestamps = all_timestamps.sort_values()
    print(f"📅 Timeline creata: {len(all_timestamps)} punti (4 ore)")
    
    # Inizializza le serie
    balance_eur = pd.Series(index=all_timestamps, dtype=float)
    equity_min_eur = pd.Series(index=all_timestamps, dtype=float)
    equity_max_eur = pd.Series(index=all_timestamps, dtype=float)
    balance_usd = pd.Series(index=all_timestamps, dtype=float)
    equity_min_usd = pd.Series(index=all_timestamps, dtype=float)
    equity_max_usd = pd.Series(index=all_timestamps, dtype=float)
    
    # Calcola il cumulative balance in EUR
    df_trades_sorted = df_trades.sort_values('Close time')
    cumulative_balance_eur = INITIAL_BALANCE_EUR
    
    # Per ogni timestamp, calcola il balance fino a quel momento
    print("🔄 Calcolo balance cumulativo...")
    trade_idx = 0
    for i, timestamp in enumerate(all_timestamps):
        # Aggiungi i profit/loss dei trade chiusi fino a questo timestamp
        while (trade_idx < len(df_trades_sorted) and 
               df_trades_sorted.iloc[trade_idx]['Close time'] <= timestamp):
            cumulative_balance_eur += df_trades_sorted.iloc[trade_idx]['Profit/Loss (Global)']
            trade_idx += 1
        
        balance_eur.iloc[i] = cumulative_balance_eur
        
        # Converti in USD
        exchange_rate = get_exchange_rate(timestamp)
        balance_usd.iloc[i] = cumulative_balance_eur * exchange_rate
        
        # Calcola unrealized P&L per i trade aperti
        open_trades = df_trades[
            (df_trades['Open time'] <= timestamp) & 
            (df_trades['Close time'] > timestamp)
        ]
        
        unrealized_min_eur = 0.0
        unrealized_max_eur = 0.0
        
        for _, trade in open_trades.iterrows():
            symbol = trade['Symbol']
            entry_price = trade['Open price']
            size = trade['Size']
            contract_size = CONTRACT_SIZE.get(symbol, 100)
            trade_type = trade['Type'].strip()
            
            # Ottieni i prezzi correnti
            df_prices = prices_cache.get(symbol)
            if df_prices is None:
                continue
            
            # Trova la candela più vicina ma non successiva al timestamp
            available_prices = df_prices[df_prices.index <= timestamp]
            if available_prices.empty:
                continue
            
            current_candle = available_prices.iloc[-1]
            current_low = current_candle['LOW']
            current_high = current_candle['HIGH']
            
            # Calcola unrealized P&L
            if trade_type == 'Buy':
                unrealized_min = (current_low - entry_price) * size * contract_size
                unrealized_max = (current_high - entry_price) * size * contract_size
            else:  # Sell
                unrealized_min = (entry_price - current_high) * size * contract_size
                unrealized_max = (entry_price - current_low) * size * contract_size
            
            unrealized_min_eur += unrealized_min
            unrealized_max_eur += unrealized_max
        
        # Calcola equity
        equity_min_eur.iloc[i] = cumulative_balance_eur + unrealized_min_eur
        equity_max_eur.iloc[i] = cumulative_balance_eur + unrealized_max_eur
        equity_min_usd.iloc[i] = equity_min_eur.iloc[i] * exchange_rate
        equity_max_usd.iloc[i] = equity_max_eur.iloc[i] * exchange_rate
        
        if i % 1000 == 0:
            print(f"   Progresso: {i+1}/{len(all_timestamps)} ({(i+1)/len(all_timestamps)*100:.1f}%)")
    
    return balance_usd, equity_min_usd, equity_max_usd, all_timestamps

def main():
    df_trades = load_report()
    
    # Calcola statistiche di controllo dal report HTML
    print("\n📊 Statistiche dal report aggregato (per controllo):")
    print("   Apri Portfolio_aggregato.html per verificare:")
    print("   - Profitto totale")
    print("   - Drawdown massimo")
    print("   - Valori finali")
    
    # Calcola equity
    balance, eq_min, eq_max, timestamps = calculate_portfolio_equity(df_trades)
    
    # Calcola drawdown
    dd_pct, dd_usd = calculate_drawdowns(eq_min)
    
    print(f"\n📊 Risultati Equity Portafoglio:")
    print(f"   Balance iniziale: ${INITIAL_BALANCE_EUR * 1.10:,.2f} USD")
    print(f"   Balance finale: ${balance.iloc[-1]:,.2f} USD")
    print(f"   Equity Min finale: ${eq_min.iloc[-1]:,.2f} USD")
    print(f"   Equity Max finale: ${eq_max.iloc[-1]:,.2f} USD")
    print(f"   Max Drawdown Equity Min: {dd_pct}% (${dd_usd:,.2f})")
    print(f"   Profitto totale: ${balance.iloc[-1] - INITIAL_BALANCE_EUR * 1.10:,.2f} USD")
    
    # Grafico
    fig, ax = plt.subplots(figsize=(24, 12))
    
    # Plot balance e equity
    ax.plot(timestamps, balance, label='Balance ($)', color='blue', linewidth=2, alpha=0.8)
    ax.fill_between(timestamps, eq_min, eq_max, color='gray', alpha=0.15, label='Range Equity')
    ax.plot(timestamps, eq_min, label='Equity Min (Adverse Exposure)', color='red', linewidth=1, alpha=0.6)
    ax.plot(timestamps, eq_max, label='Equity Max (Favorable Exposure)', color='green', linewidth=1, alpha=0.6)
    
    # Formattazione
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    
    ax.set_title('Equity Curve Portafoglio Aggregata - Balance e Equity Range', fontsize=18, pad=20)
    ax.set_ylabel('Valore ($ USD)', fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Aggiungi annotazioni
    ax.annotate(f'Max DD: {dd_pct}% (${dd_usd:,.0f})', 
                xy=(0.02, 0.05), xycoords='axes fraction',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"\n✅ Grafico salvato: {OUTPUT_PLOT}")
    
    # Salva i dati in CSV per analisi
    equity_df = pd.DataFrame({
        'Timestamp': timestamps,
        'Balance_USD': balance.values,
        'Equity_Min_USD': eq_min.values,
        'Equity_Max_USD': eq_max.values
    })
    equity_df.to_csv('data/reports/equity_curve_detailed.csv', index=False)
    print("✅ Dati equity salvati: data/reports/equity_curve_detailed.csv")
    
    plt.show()

if __name__ == "__main__":
    main()