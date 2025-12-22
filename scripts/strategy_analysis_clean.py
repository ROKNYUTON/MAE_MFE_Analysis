# ============================================================
# STRATEGY ANALYSIS - CLEAN & ROBUST VERSION
# Autore: Quant Analysis Assistant
# READY TO PASTE - non serve modificare nulla per partire
# ============================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================
# CONFIGURAZIONE BASE
# ============================================================

REPORT_PATH = 'data/reports/multi_asset_report.csv'
DATASETS_FOLDER = 'data/datasets'
OUTPUT_FOLDER = 'data/reports/strategies_analysis'

INITIAL_BALANCE_EUR = 100000.0

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# CONTRATTI E VALUTE (FONDAMENTALE)
# ============================================================

CONTRACT_SIZE = {
    'GOLD.pro': 100,
    'US100.pro': 20,
    'US500.pro': 50,
    'DE30.pro': 25,
    'USDJPY.pro': 1000,
}

# Tutti questi strumenti sono quotati in USD
SYMBOL_QUOTE_CURRENCY = {
    'GOLD.pro': 'USD',
    'US100.pro': 'USD',
    'US500.pro': 'USD',
    'USDJPY.pro': 'USD',
    'DE30.pro': 'EUR',  # DE30 è già in EUR
}

# ============================================================
# TASSI EUR -> USD (1 EUR = X USD)
# ============================================================

RATES = {
    '2020-01': 1.1095, '2021-01': 1.2165,
    '2022-01': 1.1317, '2023-01': 1.0788,
    '2024-01': 1.0906, '2025-01': 1.0333,
}

def get_rate(date):
    return RATES.get(date.strftime('%Y-%m'), 1.10)

# ============================================================
# FUNZIONI DI SUPPORTO (SICURE)
# ============================================================

def safe_positive(x):
    """MAE e MFE NON possono essere negativi"""
    return max(0.0, float(x))

def usd_to_eur(value_usd, date):
    """Conversione CORRETTA USD -> EUR"""
    rate = get_rate(date)
    return value_usd / rate

# ============================================================
# CARICAMENTO DATI
# ============================================================

prices_cache = {}

def load_price_data(symbol):
    if symbol in prices_cache:
        return prices_cache[symbol]

    filename = f"{symbol}_M5_2020_2025.csv"
    path = os.path.join(DATASETS_FOLDER, filename)

    if not os.path.exists(path):
        print(f"❌ Dataset NON trovato: {filename}")
        prices_cache[symbol] = None
        return None

    df = pd.read_csv(path, sep='\t')
    df.columns = [c.replace('<', '').replace('>', '') for c in df.columns]

    df['Datetime'] = pd.to_datetime(
        df['DATE'] + ' ' + df['TIME'],
        format='%Y.%m.%d %H:%M:%S'
    )

    df.set_index('Datetime', inplace=True)
    df = df[['OPEN', 'HIGH', 'LOW', 'CLOSE']].astype(float)

    prices_cache[symbol] = df
    print(f"✅ Prezzi caricati: {symbol} ({len(df)} candele)")
    return df

def load_report():
    df = pd.read_csv(REPORT_PATH)
    df['Open time'] = pd.to_datetime(df['Open time'], format='%d.%m.%Y %H:%M:%S')
    df['Close time'] = pd.to_datetime(df['Close time'], format='%d.%m.%Y %H:%M:%S')
    return df

# ============================================================
# CALCOLO MAE / MFE CORRETTO
# ============================================================

def calculate_mae_mfe(trade, prices):
    if prices is None:
        return None

    entry = trade['Open price']
    exit_price = trade['Close price']
    size = trade['Size']
    symbol = trade['Symbol']
    direction = trade['Type'].strip()
    contract = CONTRACT_SIZE[symbol]
    quote = SYMBOL_QUOTE_CURRENCY[symbol]

    period = prices.loc[
        (prices.index >= trade['Open time']) &
        (prices.index <= trade['Close time'])
    ]

    if period.empty:
        return None

    if direction == 'Buy':
        mfe_pts = safe_positive(period['HIGH'].max() - entry)
        mae_pts = safe_positive(entry - period['LOW'].min())
        final_pts = exit_price - entry
    else:
        mfe_pts = safe_positive(entry - period['LOW'].min())
        mae_pts = safe_positive(period['HIGH'].max() - entry)
        final_pts = entry - exit_price

    mfe_money = mfe_pts * size * contract
    mae_money = mae_pts * size * contract
    final_money = final_pts * size * contract

    if quote == 'USD':
        mfe_eur = usd_to_eur(mfe_money, trade['Close time'])
        mae_eur = usd_to_eur(mae_money, trade['Close time'])
        final_eur = usd_to_eur(final_money, trade['Close time'])
    else:
        mfe_eur = mfe_money
        mae_eur = mae_money
        final_eur = final_money

    return mfe_eur, mae_eur, final_eur

# ============================================================
# ANALISI STRATEGIA
# ============================================================

def analyze_strategy(df_trades, strategy):

    print(f"\n{'='*60}")
    print(f"ANALISI STRATEGIA: {strategy}")
    print(f"{'='*60}")

    df_s = df_trades[df_trades['Strategy name (Global)'] == strategy].copy()

    results = []

    for _, trade in df_s.iterrows():
        prices = load_price_data(trade['Symbol'])
        out = calculate_mae_mfe(trade, prices)
        if out is None:
            continue

        mfe, mae, final_pl = out

        results.append({
            'MFE': mfe,
            'MAE': mae,
            'Final_PL': final_pl,
            'Win': final_pl > 0
        })

    df = pd.DataFrame(results)

    # ==============================
    # METRICHE ROBUSTE
    # ==============================

    df['MFE_Utilization'] = np.where(
        df['MFE'] > 1,
        df['Final_PL'] / df['MFE'] * 100,
        np.nan
    )

    total_trades = len(df)
    win_rate = df['Win'].mean() * 100

    print(f"Trade totali: {total_trades}")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Profitto totale: €{df['Final_PL'].sum():,.2f}")
    print(f"Profitto medio: €{df['Final_PL'].mean():.2f}")

    print("\nEFFICIENZA:")
    print(f"MFE medio: €{df['MFE'].median():.2f}")
    print(f"MAE medio: €{df['MAE'].median():.2f}")
    print(f"Utilizzo MFE MEDIANA: {df['MFE_Utilization'].median():.1f}%")

    # ==============================
    # RACCOMANDAZIONI
    # ==============================

    print("\nRACCOMANDAZIONI OPERATIVE:")

    if df['MFE_Utilization'].median() < 30:
        print("⚠️ Esci troppo presto: il mercato va a favore ma non lo sfrutti.")
        print("👉 Prova trailing stop basato su ATR (1.5x – 2x ATR).")

    if df['MAE'].median() > df['MFE'].median():
        print("⚠️ Il rischio è maggiore del potenziale.")
        print("👉 Riduci size o usa stop-loss dinamici.")

    if win_rate < 45:
        print("⚠️ Win rate basso.")
        print("👉 Aggiungi filtro trend (es. SMA 200).")

    if win_rate > 55:
        print("✅ Win rate buono.")
        print("👉 Punta a lasciare correre i trade vincenti.")

# ============================================================
# MAIN
# ============================================================

def main():
    print("🔍 AVVIO ANALISI STRATEGIE")
    df_trades = load_report()

    strategies = df_trades['Strategy name (Global)'].unique()

    for strat in strategies:
        analyze_strategy(df_trades, strat)

    print("\n✅ ANALISI COMPLETATA")

if __name__ == "__main__":
    main()
