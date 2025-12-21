import pandas as pd
import os
from datetime import datetime

# Config
REPORT_PATH = 'data/reports/multi_asset_report.csv'
DATASETS_FOLDER = 'data/datasets'

# Tassi medi mensili EUR/USD (1 € = X USD) - dati storici precisi da ECB, OFX, X-rates
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

def calculate_mae_mfe(trade, df_prices):
    if df_prices is None:
        return None, None

    entry_time = trade['Open time']
    exit_time = trade['Close time']
    entry_price = trade['Open price']
    direction = trade['Type'].strip()
    size = trade['Size']
    symbol = trade['Symbol']
    contract = CONTRACT_SIZE.get(symbol, 100)

    mask = (df_prices.index >= entry_time) & (df_prices.index <= exit_time)
    period = df_prices.loc[mask]

    if period.empty:
        return None, None

    high = period['HIGH'].max()
    low = period['LOW'].min()

    if direction == 'Buy':
        mfe_points = high - entry_price
        mae_points = entry_price - low
    else:
        mfe_points = entry_price - low
        mae_points = high - entry_price

    mae_points = max(mae_points, 0)
    mfe_points = max(mfe_points, 0)

    mae_eur = mae_points * size * contract
    mfe_eur = mfe_points * size * contract

    month_key = exit_time.strftime('%Y-%m')
    rate = RATES.get(month_key, 1.10)

    mae_usd = round(mae_eur * rate, 2)
    mfe_usd = round(mfe_eur * rate, 2)

    return mae_usd, mfe_usd

def main():
    df_trades = load_report()

    unique_symbols = df_trades['Symbol'].unique()
    for sym in unique_symbols:
        load_price_data(sym)

    print("\n🔥 Calcolo MAE/MFE con tasso mensile preciso...\n")

    results = []
    count = 0
    total = len(df_trades)

    for idx, trade in df_trades.iterrows():
        count += 1
        symbol = trade['Symbol']
        df_prices = prices_cache.get(symbol)

        mae, mfe = calculate_mae_mfe(trade, df_prices)

        profit_eur = trade['Profit/Loss (Global)']
        month_key = trade['Close time'].strftime('%Y-%m')
        rate = RATES.get(month_key, 1.10)
        profit_usd = round(profit_eur * rate, 2)

        print(f"Trade {count}/{total} | {trade['Strategy name (Global)']} | {symbol} | {trade['Type']} | Size: {trade['Size']:.3f}")
        print(f"   Entry: {trade['Open time'].strftime('%Y-%m-%d %H:%M')} @ {trade['Open price']}")
        print(f"   Profit: {profit_eur:.2f} € (mese {month_key}, tasso {rate}) → {profit_usd:.2f} $")
        print(f"   → MAE: ${mae if mae is not None else 'N/A'} | MFE: ${mfe if mfe is not None else 'N/A'}\n")

        results.append({
            'Strategy': trade['Strategy name (Global)'],
            'Symbol': symbol,
            'Profit USD': profit_usd,
            'MAE USD': mae if mae is not None else 0,
            'MFE USD': mfe if mfe is not None else 0
        })

    print("="*80)
    print("RIASSUNTO PER STRATEGIA (in $ reali - tasso mensile)")
    print("="*80)
    summary = pd.DataFrame(results)
    if not summary.empty:
        agg = summary.groupby('Strategy')[['Profit USD', 'MAE USD', 'MFE USD']].agg({
            'Profit USD': 'sum',
            'MAE USD': 'mean',
            'MFE USD': 'mean'
        }).round(2)
        agg['Count'] = summary.groupby('Strategy').size()
        print(agg.sort_values('Profit USD', ascending=False))

if __name__ == "__main__":
    main()
    