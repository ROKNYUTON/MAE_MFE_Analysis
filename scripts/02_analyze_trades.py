import pandas as pd
from bs4 import BeautifulSoup
import os
from datetime import datetime

# Percorsi file
report_path = 'data/reports/p1aSTOCHA_H1_GOLD_1_1.html'
csv_path = 'data/datasets/GOLD_M5_2020_2025.csv'

# Fattore per GOLD: 100 (contract size / tick value)
POINT_VALUE = 100  # Per convertire differenza prezzo in $ per lotto

def load_trades():
    if not os.path.exists(report_path):
        print(f"❌ Report non trovato: {report_path}")
        return None

    with open(report_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    # Prendiamo la tabella 6 (indice 5, perché Python parte da 0)
    tables = soup.find_all('table')
    if len(tables) < 6:
        print("❌ Non trovata la tabella dei trade (aspettavo almeno 6 tabelle)")
        return None

    table = tables[5]  # Tabella 6 nell'ordine
    rows = table.find_all('tr')

    # Estrai intestazioni
    headers = [th.text.strip() for th in rows[0].find_all(['th', 'td'])]
    data = []

    for row in rows[1:]:  # Salta header
        cols = row.find_all('td')
        if cols:
            values = [col.text.strip() for col in cols]
            if len(values) == len(headers):  # Solo righe complete
                data.append(values)

    df_trades = pd.DataFrame(data, columns=headers)

    # Converti date/orari (formato italiano: DD.MM.YYYY HH:MM:SS)
    df_trades['Open time'] = pd.to_datetime(df_trades['Open time'], format='%d.%m.%Y %H:%M:%S')
    df_trades['Close time'] = pd.to_datetime(df_trades['Close time'], format='%d.%m.%Y %H:%M:%S')

    # Converti prezzi e size in float
    df_trades['Open price'] = df_trades['Open price'].astype(float)
    df_trades['Close price'] = df_trades['Close price'].astype(float)
    df_trades['Size'] = df_trades['Size'].astype(float)

    print(f"✅ Caricati {len(df_trades)} trade dal report!")
    print("Prime 3 righe:")
    print(df_trades[['Ticket', 'Type', 'Open time', 'Open price', 'Size', 'Close time', 'Close price', 'Profit/Loss']].head(3))

    return df_trades

def load_prices():
    if not os.path.exists(csv_path):
        print(f"❌ CSV non trovato: {csv_path}")
        return None

    df_prices = pd.read_csv(csv_path, sep='\t')
    df_prices.columns = [c.replace('<', '').replace('>', '') for c in df_prices.columns]

    # Combina DATE e TIME
    df_prices['Datetime'] = pd.to_datetime(df_prices['DATE'] + ' ' + df_prices['TIME'], format='%Y.%m.%d %H:%M:%S')
    df_prices.set_index('Datetime', inplace=True)

    # Seleziona solo colonne utili e converti in float
    df_prices = df_prices[['OPEN', 'HIGH', 'LOW', 'CLOSE']]
    df_prices = df_prices.astype(float)

    print(f"✅ Prezzi M5 caricati: {len(df_prices)} candele dal {df_prices.index[0]} al {df_prices.index[-1]}")
    return df_prices

def calculate_mae_mfe(trade, df_prices):
    entry_time = trade['Open time']
    exit_time = trade['Close time']
    entry_price = trade['Open price']
    direction = trade['Type'].strip().lower()  # 'buy' o 'sell'
    size = trade['Size']

    # Filtra candele tra entry e exit (inclusi i bordi)
    mask = (df_prices.index >= entry_time) & (df_prices.index <= exit_time)
    period = df_prices.loc[mask]

    if period.empty:
        return None, None

    highs = period['HIGH']
    lows = period['LOW']

    if direction == 'buy':
        mfe_points = highs.max() - entry_price
        mae_points = entry_price - lows.min()
    elif direction == 'sell':
        mfe_points = entry_price - lows.min()
        mae_points = highs.max() - entry_price
    else:
        return None, None

    # Forza positivo (se negativo, significa 0 escursione)
    mae_points = max(mae_points, 0)
    mfe_points = max(mfe_points, 0)

    # Converti in $
    mae_dollar = round(mae_points * size * POINT_VALUE, 2)
    mfe_dollar = round(mfe_points * size * POINT_VALUE, 2)

    return mae_dollar, mfe_dollar

def main():
    df_trades = load_trades()
    if df_trades is None:
        return

    df_prices = load_prices()
    if df_prices is None:
        return

    print("\n" + "="*80)
    print("CALCOLO MAE E MFE PER OGNI TRADE (IN $)")
    print("="*80)

    for idx, trade in df_trades.iterrows():
        mae, mfe = calculate_mae_mfe(trade, df_prices)

        print(f"\nTrade {idx + 1} | Ticket: {trade['Ticket']} | {trade['Type']} | Size: {trade['Size']}")
        print(f"   Entrata:  {trade['Open time']} @ {trade['Open price']}")
        print(f"   Uscita:   {trade['Close time']} @ {trade['Close price']}")
        print(f"   Profitto: {trade['Profit/Loss']}")
        print(f"   → MAE (Max Adverse Excursion):  ${mae if mae is not None else 'N/A (no dati M5)'}")
        print(f"   → MFE (Max Favorable Excursion): ${mfe if mfe is not None else 'N/A (no dati M5)'}")
        if trade['Comment']:
            print(f"   Commento: {trade['Comment']}")

if __name__ == "__main__":
    main()