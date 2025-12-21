import pandas as pd
from bs4 import BeautifulSoup
import os
import matplotlib.pyplot as plt
import numpy as np

# Percorsi file
report_path = 'data/reports/p1aSTOCHA_H1_GOLD_1_1.html'
csv_path = 'data/datasets/GOLD_M5_2020_2025.csv'
output_plot = 'data/reports/equity_curve.png'

# Fattore per GOLD
POINT_VALUE = 100

# Balance iniziale
INITIAL_BALANCE = 100000.0

def load_trades():
    if not os.path.exists(report_path):
        print(f"❌ Report non trovato: {report_path}")
        return None

    with open(report_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    tables = soup.find_all('table')
    if len(tables) < 6:
        print("❌ Non trovata la tabella dei trade")
        return None

    table = tables[5]
    rows = table.find_all('tr')

    headers = [th.text.strip() for th in rows[0].find_all(['th', 'td'])]
    data = []

    for row in rows[1:]:
        cols = row.find_all('td')
        if cols:
            values = [col.text.strip() for col in cols]
            if len(values) == len(headers):
                data.append(values)

    df_trades = pd.DataFrame(data, columns=headers)

    df_trades['Open time'] = pd.to_datetime(df_trades['Open time'], format='%d.%m.%Y %H:%M:%S')
    df_trades['Close time'] = pd.to_datetime(df_trades['Close time'], format='%d.%m.%Y %H:%M:%S')
    df_trades['Open price'] = df_trades['Open price'].astype(float)
    df_trades['Close price'] = df_trades['Close price'].astype(float)
    df_trades['Size'] = df_trades['Size'].astype(float)
    df_trades['Profit/Loss'] = df_trades['Profit/Loss'].str.replace('$', '').str.replace(' ', '').astype(float)

    df_trades = df_trades.sort_values('Open time').reset_index(drop=True)

    print(f"✅ Caricati {len(df_trades)} trade!")
    return df_trades

def load_prices():
    if not os.path.exists(csv_path):
        print(f"❌ CSV non trovato: {csv_path}")
        return None

    df_prices = pd.read_csv(csv_path, sep='\t')
    df_prices.columns = [c.replace('<', '').replace('>', '') for c in df_prices.columns]

    df_prices['Datetime'] = pd.to_datetime(df_prices['DATE'] + ' ' + df_prices['TIME'], format='%Y.%m.%d %H:%M:%S')
    df_prices.set_index('Datetime', inplace=True)
    df_prices = df_prices[['OPEN', 'HIGH', 'LOW', 'CLOSE']].astype(float)

    print(f"✅ Prezzi M5 caricati: {len(df_prices)} candele")
    return df_prices

def calculate_equity(df_trades, df_prices):
    timestamps = df_prices.index

    # Serie per balance, equity_min e equity_max
    balance = pd.Series(INITIAL_BALANCE, index=timestamps)
    equity_min = pd.Series(INITIAL_BALANCE, index=timestamps)
    equity_max = pd.Series(INITIAL_BALANCE, index=timestamps)

    # Calcolo cumulativo profitto realizzato
    df_trades = df_trades.sort_values('Close time')
    cumulative = INITIAL_BALANCE
    for _, trade in df_trades.iterrows():
        # Trova la prima candela dopo o uguale alla close time
        close_slice = timestamps[timestamps >= trade['Close time']]
        if not close_slice.empty:
            idx_start = timestamps.get_loc(close_slice[0])
            cumulative += trade['Profit/Loss']
            balance.iloc[idx_start:] = cumulative

    # Equity: per ogni candela, calcola unrealized delle posizioni aperte
    current_profit = INITIAL_BALANCE
    for i, t in enumerate(timestamps):
        # Aggiorna balance corrente (potrebbe essere già aggiornata sopra)
        current_profit = balance.iloc[i]

        # Trova trade aperti esattamente in questa candela
        open_trades = df_trades[
            (df_trades['Open time'] <= t) &
            (df_trades['Close time'] >= t)
        ]

        if open_trades.empty:
            equity_min.iloc[i] = current_profit
            equity_max.iloc[i] = current_profit
            continue

        high = df_prices.iloc[i]['HIGH']
        low = df_prices.iloc[i]['LOW']

        unreal_min = 0.0
        unreal_max = 0.0

        for _, trade in open_trades.iterrows():
            size = trade['Size']
            entry = trade['Open price']
            direction = trade['Type'].strip().lower()

            if direction == 'buy':
                unreal_min += (low - entry) * size * POINT_VALUE
                unreal_max += (high - entry) * size * POINT_VALUE
            else:  # sell
                unreal_min += (entry - high) * size * POINT_VALUE
                unreal_max += (entry - low) * size * POINT_VALUE

        equity_min.iloc[i] = current_profit + unreal_min
        equity_max.iloc[i] = current_profit + unreal_max

    return balance, equity_min, equity_max

def plot_curves(balance, equity_min, equity_max, df_prices):
    plt.figure(figsize=(16, 8))
    plt.plot(df_prices.index, balance, label='Balance (solo chiusure)', color='blue', linewidth=2)
    plt.plot(df_prices.index, equity_max, label='Equity Max (spike favorevoli)', color='green', alpha=0.7)
    plt.plot(df_prices.index, equity_min, label='Equity Min (spike avversi)', color='red', alpha=0.7)
    plt.fill_between(df_prices.index, equity_min, equity_max, color='gray', alpha=0.2, label='Range Equity (MAE/MFE)')
    plt.title('Balance vs Equity Effettiva con Spike Intra-Trade (GOLD M5)')
    plt.xlabel('Data')
    plt.ylabel('Valore Conto ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=200)
    print(f"✅ Plot salvato in: {output_plot}")
    plt.show()

def calculate_drawdowns(balance, equity_min):
    def max_dd(series):
        peak = np.maximum.accumulate(series)
        dd = (series - peak) / peak * 100
        return round(dd.min(), 2)

    print("\n📊 STATISTICHE DRAWDOWN")
    print(f"   Max Drawdown Balance:          {max_dd(balance)}%")
    print(f"   Max Drawdown Equity Min:       {max_dd(equity_min)}%")
    print(f"   Differenza (quanto è peggio l'equity reale): {round(max_dd(equity_min) - max_dd(balance), 2)}%")

def main():
    df_trades = load_trades()
    if df_trades is None:
        return

    df_prices = load_prices()
    if df_prices is None:
        return

    print("\n🔄 Calcolo equity curve in corso (può richiedere 20-40 secondi con 418k candele)...")
    balance, equity_min, equity_max = calculate_equity(df_trades, df_prices)

    calculate_drawdowns(balance, equity_min)
    plot_curves(balance, equity_min, equity_max, df_prices)

if __name__ == "__main__":
    main()