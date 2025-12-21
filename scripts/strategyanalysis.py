import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats

# Config
REPORT_PATH = 'data/reports/multi_asset_report.csv'
DATASETS_FOLDER = 'data/datasets'
INITIAL_BALANCE_EUR = 100000.0
OUTPUT_FOLDER = 'data/reports/strategies_analysis'

# Tassi mensili EUR/USD
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
    '2024-01': 1.0906, '2024-04': 1.0723, '2024-05': 1.0810,
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

# Crea la cartella di output
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

prices_cache = {}

def load_price_data(symbol):
    """Carica i dati dei prezzi per un simbolo"""
    if symbol in prices_cache:
        return prices_cache[symbol]
    
    filename = f"{symbol}_M5_2020_2025.csv"
    path = os.path.join(DATASETS_FOLDER, filename)
    
    if not os.path.exists(path):
        print(f"⚠️ Dataset non trovato per {symbol}: {filename}")
        prices_cache[symbol] = None
        return None
    
    print(f"🔄 Caricamento dataset {symbol}...")
    df = pd.read_csv(path, sep='\t')
    df.columns = [c.replace('<', '').replace('>', '') for c in df.columns]
    df['Datetime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y.%m.%d %H:%M:%S')
    df.set_index('Datetime', inplace=True)
    df = df[['OPEN', 'HIGH', 'LOW', 'CLOSE']].astype(float)
    prices_cache[symbol] = df
    print(f"✅ {symbol} caricato ({len(df)} candele)")
    return df

def load_report():
    """Carica il report dei trade"""
    df = pd.read_csv(REPORT_PATH, sep=',')
    df.columns = [c.strip() for c in df.columns]
    df['Open time'] = pd.to_datetime(df['Open time'], format='%d.%m.%Y %H:%M:%S')
    df['Close time'] = pd.to_datetime(df['Close time'], format='%d.%m.%Y %H:%M:%S')
    df['Open price'] = df['Open price'].astype(float)
    df['Close price'] = df['Close price'].astype(float)
    df['Size'] = df['Size'].astype(float)
    df['Profit/Loss (Global)'] = df['Profit/Loss (Global)'].astype(float)
    print(f"✅ Report caricato: {len(df)} trade, {df['Strategy name (Global)'].nunique()} strategie")
    return df

def get_exchange_rate(date):
    """Ottieni il tasso di cambio per una data specifica"""
    month_key = date.strftime('%Y-%m')
    return RATES.get(month_key, 1.10)

def calculate_trade_mae_mfe(trade, df_prices):
    """Calcola MAE e MFE per un singolo trade"""
    if df_prices is None:
        return 0, 0, 0, 0, 0, 0
    
    entry_time = trade['Open time']
    exit_time = trade['Close time']
    entry_price = trade['Open price']
    direction = trade['Type'].strip()
    size = trade['Size']
    symbol = trade['Symbol']
    contract = CONTRACT_SIZE.get(symbol, 100)
    
    # Filtra le candele durante la vita del trade
    mask = (df_prices.index >= entry_time) & (df_prices.index <= exit_time)
    period = df_prices.loc[mask]
    
    if period.empty:
        return 0, 0, 0, 0, 0, 0
    
    # Trova MAE e MFE in punti
    if direction == 'Buy':
        mfe_points = period['HIGH'].max() - entry_price
        mae_points = entry_price - period['LOW'].min()
    else:  # Sell
        mfe_points = entry_price - period['LOW'].min()
        mae_points = period['HIGH'].max() - entry_price
    
    # Calcola in EUR
    mae_eur = mae_points * size * contract
    mfe_eur = mfe_points * size * contract
    
    # Calcola P&L finale in EUR
    if direction == 'Buy':
        final_pl_points = trade['Close price'] - entry_price
    else:
        final_pl_points = entry_price - trade['Close price']
    
    final_pl_eur = final_pl_points * size * contract
    
    # Converti in USD
    month_key = exit_time.strftime('%Y-%m')
    rate = RATES.get(month_key, 1.10)
    
    mae_usd = mae_eur * rate
    mfe_usd = mfe_eur * rate
    final_pl_usd = final_pl_eur * rate
    
    return mae_eur, mfe_eur, final_pl_eur, mae_usd, mfe_usd, final_pl_usd

def calculate_strategy_equity(df_strategy_trades):
    """Calcola l'equity curve per una specifica strategia"""
    # Carica i dati dei prezzi per tutti i simboli usati dalla strategia
    symbols = df_strategy_trades['Symbol'].unique()
    for symbol in symbols:
        load_price_data(symbol)
    
    # Crea timeline (ogni 4 ore per efficienza)
    all_timestamps = pd.DatetimeIndex([])
    for symbol in symbols:
        df = prices_cache.get(symbol)
        if df is not None:
            sampled_df = df.resample('4H').agg({
                'OPEN': 'first',
                'HIGH': 'max',
                'LOW': 'min',
                'CLOSE': 'last'
            }).dropna()
            all_timestamps = all_timestamps.union(sampled_df.index)
    
    all_timestamps = all_timestamps.sort_values()
    
    # Inizializza le serie
    balance_eur = pd.Series(index=all_timestamps, dtype=float)
    equity_min_eur = pd.Series(index=all_timestamps, dtype=float)
    equity_max_eur = pd.Series(index=all_timestamps, dtype=float)
    
    # Calcola balance cumulativo
    df_trades_sorted = df_strategy_trades.sort_values('Close time')
    cumulative_balance_eur = INITIAL_BALANCE_EUR
    trade_idx = 0
    
    for i, timestamp in enumerate(all_timestamps):
        # Aggiorna balance con trade chiusi
        while (trade_idx < len(df_trades_sorted) and 
               df_trades_sorted.iloc[trade_idx]['Close time'] <= timestamp):
            cumulative_balance_eur += df_trades_sorted.iloc[trade_idx]['Profit/Loss (Global)']
            trade_idx += 1
        
        balance_eur.iloc[i] = cumulative_balance_eur
        
        # Calcola unrealized P&L per trade aperti
        open_trades = df_strategy_trades[
            (df_strategy_trades['Open time'] <= timestamp) & 
            (df_strategy_trades['Close time'] > timestamp)
        ]
        
        unrealized_min_eur = 0.0
        unrealized_max_eur = 0.0
        
        for _, trade in open_trades.iterrows():
            symbol = trade['Symbol']
            df_prices = prices_cache.get(symbol)
            if df_prices is None:
                continue
            
            available_prices = df_prices[df_prices.index <= timestamp]
            if available_prices.empty:
                continue
            
            current_candle = available_prices.iloc[-1]
            entry_price = trade['Open price']
            size = trade['Size']
            contract_size = CONTRACT_SIZE.get(symbol, 100)
            
            if trade['Type'].strip() == 'Buy':
                unrealized_min_eur += (current_candle['LOW'] - entry_price) * size * contract_size
                unrealized_max_eur += (current_candle['HIGH'] - entry_price) * size * contract_size
            else:
                unrealized_min_eur += (entry_price - current_candle['HIGH']) * size * contract_size
                unrealized_max_eur += (entry_price - current_candle['LOW']) * size * contract_size
        
        equity_min_eur.iloc[i] = cumulative_balance_eur + unrealized_min_eur
        equity_max_eur.iloc[i] = cumulative_balance_eur + unrealized_max_eur
    
    return balance_eur, equity_min_eur, equity_max_eur, all_timestamps

def analyze_strategy_performance(df_strategy_trades, strategy_name):
    """Analizza in dettaglio le performance di una strategia"""
    print(f"\n{'='*80}")
    print(f"ANALISI STRATEGIA: {strategy_name}")
    print(f"{'='*80}")
    
    # Calcola MAE/MFE per tutti i trade
    results = []
    for idx, trade in df_strategy_trades.iterrows():
        symbol = trade['Symbol']
        df_prices = prices_cache.get(symbol)
        
        mae_eur, mfe_eur, final_pl_eur, mae_usd, mfe_usd, final_pl_usd = calculate_trade_mae_mfe(trade, df_prices)
        
        results.append({
            'Symbol': symbol,
            'Type': trade['Type'],
            'Size': trade['Size'],
            'Duration': (trade['Close time'] - trade['Open time']).total_seconds() / 3600,  # in ore
            'MAE_EUR': mae_eur,
            'MFE_EUR': mfe_eur,
            'Final_PL_EUR': final_pl_eur,
            'MAE_USD': mae_usd,
            'MFE_USD': mfe_usd,
            'Final_PL_USD': final_pl_usd,
            'Win': final_pl_eur > 0
        })
    
    df_results = pd.DataFrame(results)
    
    # Statistiche base
    total_trades = len(df_results)
    winning_trades = df_results['Win'].sum()
    win_rate = winning_trades / total_trades * 100
    
    total_profit_eur = df_results['Final_PL_EUR'].sum()
    avg_profit_eur = df_results['Final_PL_EUR'].mean()
    avg_mae_eur = df_results['MAE_EUR'].mean()
    avg_mfe_eur = df_results['MFE_EUR'].mean()
    
    print(f"\n📊 STATISTICHE BASE:")
    print(f"   Trade totali: {total_trades}")
    print(f"   Trade vincenti: {winning_trades} ({win_rate:.1f}%)")
    print(f"   Profitto totale: €{total_profit_eur:,.2f}")
    print(f"   Profitto medio per trade: €{avg_profit_eur:,.2f}")
    print(f"   MAE medio: €{avg_mae_eur:,.2f}")
    print(f"   MFE medio: €{avg_mfe_eur:,.2f}")
    
    # Calcola efficienza di chiusura
    df_results['MFE_Utilization'] = np.where(
        df_results['MFE_EUR'] != 0,
        df_results['Final_PL_EUR'] / df_results['MFE_EUR'] * 100,
        0
    )
    
    df_results['MAE_Avoidance'] = np.where(
        df_results['MAE_EUR'] != 0,
        (df_results['MAE_EUR'] - abs(df_results['Final_PL_EUR'])) / df_results['MAE_EUR'] * 100,
        0
    )
    
    avg_mfe_utilization = df_results[df_results['MFE_EUR'] > 0]['MFE_Utilization'].mean()
    avg_mae_avoidance = df_results[df_results['MAE_EUR'] > 0]['MAE_Avoidance'].mean()
    
    print(f"\n📊 EFFICIENZA DI CHIUSURA:")
    print(f"   Utilizzo medio MFE: {avg_mfe_utilization:.1f}%")
    print(f"   Evitamento medio MAE: {avg_mae_avoidance:.1f}%")
    
    # Analisi per tipo di trade
    print(f"\n📊 ANALISI PER TIPO DI TRADE:")
    for trade_type in df_results['Type'].unique():
        type_trades = df_results[df_results['Type'] == trade_type]
        type_win_rate = type_trades['Win'].sum() / len(type_trades) * 100
        type_avg_profit = type_trades['Final_PL_EUR'].mean()
        print(f"   {trade_type}: {len(type_trades)} trade, Win Rate: {type_win_rate:.1f}%, Profitto medio: €{type_avg_profit:,.2f}")
    
    return df_results

def create_strategy_report(strategy_name, df_strategy_trades, df_results, balance_eur, equity_min_eur, equity_max_eur, timestamps):
    """Crea report completo per una strategia"""
    strategy_folder = os.path.join(OUTPUT_FOLDER, strategy_name.replace('/', '_').replace('\\', '_'))
    os.makedirs(strategy_folder, exist_ok=True)
    
    # 1. Grafico equity curve
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    
    # Equity curve
    ax = axes[0, 0]
    ax.plot(timestamps, balance_eur, label='Balance (€)', color='blue', linewidth=2)
    ax.fill_between(timestamps, equity_min_eur, equity_max_eur, color='gray', alpha=0.2, label='Range Equity')
    ax.plot(timestamps, equity_min_eur, label='Equity Min (€)', color='red', linewidth=1, alpha=0.6)
    ax.plot(timestamps, equity_max_eur, label='Equity Max (€)', color='green', linewidth=1, alpha=0.6)
    ax.set_title(f'Equity Curve - {strategy_name}', fontsize=14)
    ax.set_ylabel('Valore (€)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Distribuzione MFE vs Final P&L
    ax = axes[0, 1]
    scatter = ax.scatter(df_results['MFE_EUR'], df_results['Final_PL_EUR'], 
                        c=df_results['Win'], cmap='coolwarm', alpha=0.6, s=50)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('MFE (€)')
    ax.set_ylabel('Final P&L (€)')
    ax.set_title('MFE vs Final P&L')
    ax.grid(True, alpha=0.3)
    
    # Distribuzione MAE vs Final P&L
    ax = axes[1, 0]
    scatter = ax.scatter(df_results['MAE_EUR'], df_results['Final_PL_EUR'], 
                        c=df_results['Win'], cmap='coolwarm', alpha=0.6, s=50)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('MAE (€)')
    ax.set_ylabel('Final P&L (€)')
    ax.set_title('MAE vs Final P&L')
    ax.grid(True, alpha=0.3)
    
    # Istogramma utilizzo MFE
    ax = axes[1, 1]
    ax.hist(df_results['MFE_Utilization'], bins=30, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(x=df_results['MFE_Utilization'].mean(), color='red', linestyle='--', label=f'Media: {df_results["MFE_Utilization"].mean():.1f}%')
    ax.set_xlabel('Utilizzo MFE (%)')
    ax.set_ylabel('Frequenza')
    ax.set_title('Distribuzione Utilizzo MFE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Box plot per simbolo
    ax = axes[2, 0]
    symbol_data = []
    symbols = []
    for symbol in df_results['Symbol'].unique():
        symbol_data.append(df_results[df_results['Symbol'] == symbol]['Final_PL_EUR'].values)
        symbols.append(symbol)
    
    if symbol_data:
        ax.boxplot(symbol_data, labels=symbols)
        ax.set_ylabel('P&L (€)')
        ax.set_title('Distribuzione P&L per Simbolo')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Grafico timeline dei trade
    ax = axes[2, 1]
    colors = ['green' if win else 'red' for win in df_results['Win']]
    ax.scatter(range(len(df_results)), df_results['Final_PL_EUR'], c=colors, alpha=0.6, s=30)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Trade #')
    ax.set_ylabel('P&L (€)')
    ax.set_title('Sequenza dei Trade (Verde=Vincente, Rosso=Perdente)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(strategy_folder, f'analysis_{strategy_name.replace("/", "_")}.png'), dpi=150)
    plt.close()
    
    # 2. Salva dati CSV
    df_results.to_csv(os.path.join(strategy_folder, f'trades_analysis_{strategy_name.replace("/", "_")}.csv'), index=False)
    
    # 3. Crea file di riepilogo
    with open(os.path.join(strategy_folder, f'summary_{strategy_name.replace("/", "_")}.txt'), 'w') as f:
        f.write(f"ANALISI STRATEGIA: {strategy_name}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Trade totali: {len(df_results)}\n")
        f.write(f"Trade vincenti: {df_results['Win'].sum()} ({df_results['Win'].sum()/len(df_results)*100:.1f}%)\n")
        f.write(f"Profitto totale: €{df_results['Final_PL_EUR'].sum():,.2f}\n")
        f.write(f"Profitto medio: €{df_results['Final_PL_EUR'].mean():,.2f}\n")
        f.write(f"MAE medio: €{df_results['MAE_EUR'].mean():,.2f}\n")
        f.write(f"MFE medio: €{df_results['MFE_EUR'].mean():,.2f}\n")
        f.write(f"Utilizzo medio MFE: {df_results['MFE_Utilization'].mean():.1f}%\n")
        f.write(f"Evitamento medio MAE: {df_results['MAE_Avoidance'].mean():.1f}%\n\n")
        
        f.write("RECOMMENDATIONS:\n")
        f.write("-"*60 + "\n")
        
        # Raccomandazioni basate sui dati
        mfe_utilization = df_results['MFE_Utilization'].mean()
        if mfe_utilization < 30:
            f.write("⚠️ BASSO UTILIZZO MFE: La strategia cattura meno del 30% del potenziale profitto.\n")
            f.write("   Raccomandazione: Considera trailing stop più larghi o take profit più alti.\n\n")
        elif mfe_utilization < 50:
            f.write("⚠️ MEDIO UTILIZZO MFE: La strategia cattura meno del 50% del potenziale profitto.\n")
            f.write("   Raccomandazione: Ottimizza i livelli di uscita per catturare più profitto.\n\n")
        else:
            f.write("✅ BUON UTILIZZO MFE: La strategia cattura più del 50% del potenziale profitto.\n\n")
        
        # Analisi per tipo di trade
        f.write("ANALISI PER TIPO DI TRADE:\n")
        f.write("-"*60 + "\n")
        for trade_type in df_results['Type'].unique():
            type_data = df_results[df_results['Type'] == trade_type]
            win_rate = type_data['Win'].sum() / len(type_data) * 100
            avg_profit = type_data['Final_PL_EUR'].mean()
            f.write(f"{trade_type}:\n")
            f.write(f"  - Trade: {len(type_data)}\n")
            f.write(f"  - Win Rate: {win_rate:.1f}%\n")
            f.write(f"  - Profitto medio: €{avg_profit:,.2f}\n")
            
            if win_rate < 40:
                f.write(f"  ⚠️  Win Rate basso per {trade_type}. Considera se i segnali sono affidabili.\n")
            elif win_rate > 60:
                f.write(f"  ✅ Win Rate eccellente per {trade_type}.\n")
            
            if avg_profit < 0:
                f.write(f"  ⚠️  Profitto medio negativo per {trade_type}. Rivedi la strategia.\n")
            
            f.write("\n")
    
    print(f"✅ Report salvato in: {strategy_folder}")

def analyze_all_strategies():
    """Analizza tutte le strategie nel report"""
    df_trades = load_report()
    
    # Carica tutti i dati dei prezzi necessari
    all_symbols = df_trades['Symbol'].unique()
    for symbol in all_symbols:
        load_price_data(symbol)
    
    # Ottieni lista strategie
    strategies = df_trades['Strategy name (Global)'].unique()
    print(f"\n🔍 Analisi di {len(strategies)} strategie...")
    
    # Report aggregato
    summary_data = []
    
    for strategy in strategies:
        print(f"\n📋 Analisi strategia: {strategy}")
        
        # Filtra trade per strategia
        df_strategy = df_trades[df_trades['Strategy name (Global)'] == strategy].copy()
        
        # Calcola equity curve
        balance_eur, equity_min_eur, equity_max_eur, timestamps = calculate_strategy_equity(df_strategy)
        
        # Analizza performance
        df_results = analyze_strategy_performance(df_strategy, strategy)
        
        # Crea report dettagliato
        create_strategy_report(strategy, df_strategy, df_results, balance_eur, equity_min_eur, equity_max_eur, timestamps)
        
        # Aggrega dati per report finale
        summary_data.append({
            'Strategy': strategy,
            'Total_Trades': len(df_results),
            'Winning_Trades': df_results['Win'].sum(),
            'Win_Rate': df_results['Win'].sum() / len(df_results) * 100,
            'Total_Profit_EUR': df_results['Final_PL_EUR'].sum(),
            'Avg_Profit_EUR': df_results['Final_PL_EUR'].mean(),
            'Avg_MAE_EUR': df_results['MAE_EUR'].mean(),
            'Avg_MFE_EUR': df_results['MFE_EUR'].mean(),
            'Avg_MFE_Utilization': df_results['MFE_Utilization'].mean(),
            'Avg_MAE_Avoidance': df_results['MAE_Avoidance'].mean()
        })
    
    # Crea report comparativo
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('Total_Profit_EUR', ascending=False)
    
    # Salva report comparativo
    df_summary.to_csv(os.path.join(OUTPUT_FOLDER, 'strategies_comparison.csv'), index=False)
    
    # Grafico comparativo
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Profitto totale per strategia
    ax = axes[0, 0]
    bars = ax.barh(df_summary['Strategy'], df_summary['Total_Profit_EUR'])
    ax.set_xlabel('Profitto Totale (€)')
    ax.set_title('Profitto Totale per Strategia')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Win Rate per strategia
    ax = axes[0, 1]
    bars = ax.barh(df_summary['Strategy'], df_summary['Win_Rate'])
    ax.set_xlabel('Win Rate (%)')
    ax.set_title('Win Rate per Strategia')
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Utilizzo MFE per strategia
    ax = axes[1, 0]
    bars = ax.barh(df_summary['Strategy'], df_summary['Avg_MFE_Utilization'])
    ax.set_xlabel('Utilizzo MFE Medio (%)')
    ax.set_title('Efficienza di Chiusura (Utilizzo MFE)')
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% soglia')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Profitto medio per trade
    ax = axes[1, 1]
    bars = ax.barh(df_summary['Strategy'], df_summary['Avg_Profit_EUR'])
    ax.set_xlabel('Profitto Medio per Trade (€)')
    ax.set_title('Profitto Medio per Strategia')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'strategies_comparison.png'), dpi=150)
    
    print(f"\n{'='*80}")
    print("ANALISI COMPLETATA!")
    print(f"{'='*80}")
    print(f"\n📊 REPORT FINALE:")
    print(f"   Strategie analizzate: {len(strategies)}")
    print(f"   Strategie più profittevoli:")
    
    for i, row in df_summary.head(3).iterrows():
        print(f"     {row['Strategy']}: €{row['Total_Profit_EUR']:,.2f} ({row['Win_Rate']:.1f}% WR)")
    
    print(f"\n📈 Strategie con miglior utilizzo MFE:")
    df_mfe_sorted = df_summary.sort_values('Avg_MFE_Utilization', ascending=False)
    for i, row in df_mfe_sorted.head(3).iterrows():
        print(f"     {row['Strategy']}: {row['Avg_MFE_Utilization']:.1f}% utilizzo MFE")
    
    print(f"\n📁 I report dettagliati sono stati salvati in: {OUTPUT_FOLDER}")
    print(f"   - File CSV comparativo: strategies_comparison.csv")
    print(f"   - Grafico comparativo: strategies_comparison.png")
    print(f"   - Per ogni strategia: cartella con analisi completa")

def main():
    """Funzione principale"""
    print("🔍 ANALISI DETTAGLIATA STRATEGIE DI TRADING")
    print("="*80)
    print("Questo script analizza ogni strategia separatamente per:")
    print("1. Calcolare equity curve con MAE/MFE")
    print("2. Analizzare l'efficienza di chiusura dei trade")
    print("3. Identificare opportunità di ottimizzazione")
    print("="*80)
    
    analyze_all_strategies()

if __name__ == "__main__":
    main()
