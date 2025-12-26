# 2_benchmark_equity.py
import pandas as pd
import os

# --- CONFIGURAZIONE ---
DATASETS_FOLDER = 'data/datasets'
OUTPUT_FOLDER = 'data/reports/comparative_analysis'
INITIAL_BALANCE = 100000.0  # Base 100k

BENCHMARKS = {
    'US500.pro': {'name': 'S&P 500', 'is_index': True},
    'US100.pro': {'name': 'NASDAQ 100', 'is_index': True},
    'GOLD.pro':  {'name': 'Gold', 'is_index': False}
}

def load_and_process_benchmark(symbol, config):
    path = os.path.join(DATASETS_FOLDER, f"{symbol}_M5_2020_2025.csv")
    if not os.path.exists(path):
        print(f"⚠️ {symbol} non trovato in {path}")
        return None

    print(f"🔄 Elaborazione {config['name']}...")
    df = pd.read_csv(path, sep='\t')
    df.columns = [c.replace('<', '').replace('>', '') for c in df.columns]
    df['dt'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y.%m.%d %H:%M:%S')
    df.set_index('dt', inplace=True)
    
    # Resample orario
    df_1h = df['CLOSE'].resample('1H').last().ffill()
    
    # Calcolo Equity su 100k (Buy & Hold)
    start_price = df_1h.iloc[0]
    equity_curve = (df_1h / start_price) * INITIAL_BALANCE
    
    return equity_curve

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"🚀 Generazione Benchmark su capitale: € {INITIAL_BALANCE:,.0f}...")
    
    combined_df = pd.DataFrame()
    for symbol, config in BENCHMARKS.items():
        equity = load_and_process_benchmark(symbol, config)
        if equity is not None:
            combined_df[config['name']] = equity

    combined_df.ffill(inplace=True)
    combined_df.dropna(inplace=True)
    
    output_path = f'{OUTPUT_FOLDER}/benchmarks_1h_equity.csv'
    combined_df.to_csv(output_path)
    print(f"✅ Benchmark salvati in: {output_path}")

if __name__ == "__main__":
    main()