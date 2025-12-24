import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Config
REPORT_PATH = 'data/reports/multi_asset_report.csv'
INITIAL_BALANCE = 100000.0

def calculate_drawdowns(series):
    """Calcola i periodi di drawdown (non solo i singoli punti)"""
    peak = series.cummax()
    drawdown = (series - peak)
    
    # Identifica i periodi
    is_in_dd = drawdown < 0
    runs = (is_in_dd != is_in_dd.shift()).cumsum()
    periods = []
    
    for run_id in runs[is_in_dd].unique():
        dd_period = drawdown[runs == run_id]
        if len(dd_period) > 0:
            periods.append({
                'Start': dd_period.index[0],
                'End': dd_period.index[-1],
                'Max_DD_Euro': dd_period.min(),
                'Duration': dd_period.index[-1] - dd_period.index[0]
            })
    
    return pd.DataFrame(periods).sort_values('Max_DD_Euro').head(10)

def main():
    if not os.path.exists(REPORT_PATH):
        print(f"❌ Errore: File {REPORT_PATH} non trovato.")
        return

    # 1. Caricamento Report
    df = pd.read_csv(REPORT_PATH)
    df.columns = [c.strip() for c in df.columns]
    
    # Pulizia date
    df['Close time'] = pd.to_datetime(df['Close time'], dayfirst=True)
    df['Profit/Loss (Global)'] = df['Profit/Loss (Global)'].astype(float)
    
    # 2. PROFITTO PER STRATEGIA (Tabella Comparativa)
    # Assumiamo che la colonna con 'S2dax', 'S3gold' si chiami 'Strategy' o 'Name'
    # Se la colonna ha un nome diverso, cambialo qui sotto (es. df['Comment'])
    strat_col = 'Name' if 'Name' in df.columns else 'Symbol' 
    
    summary = df.groupby(strat_col)['Profit/Loss (Global)'].agg(['sum', 'count']).rename(columns={'sum': 'Total Profit (€)', 'count': 'Trades'})
    summary['Avg Trade'] = summary['Total Profit (€)'] / summary['Trades']
    
    print("\n--- PROFITTO FINALE PER STRATEGIA ---")
    print(summary.to_string())

    # 3. ANALISI PORTAFOGLIO COMPLESSIVO
    df_sorted = df.sort_values('Close time')
    df_sorted['Equity'] = INITIAL_BALANCE + df_sorted['Profit/Loss (Global)'].cumsum()
    df_sorted.set_index('Close time', inplace=True)

    # Best & Worst Days (Basato sul P&L chiuso)
    daily_pnl = df_sorted['Profit/Loss (Global)'].resample('D').sum()
    best_days = daily_pnl.sort_values(ascending=False).head(10)
    worst_days = daily_pnl.sort_values(ascending=True).head(10)

    # Drawdown Periodi
    top_10_dd = calculate_drawdowns(df_sorted['Equity'])

    # 4. OUTPUT STATISTICHE
    print("\n--- TOP 10 BEST DAYS (€) ---")
    print(best_days[best_days > 0])

    print("\n--- TOP 10 WORST DAYS (€) ---")
    print(worst_days[worst_days < 0])

    print("\n--- TOP 10 DRAWDOWN PERIODS (Complessivi) ---")
    if not top_10_dd.empty:
        for i, row in top_10_dd.iterrows():
            print(f"{i+1}. Max DD: {row['Max_DD_Euro']:.2f}€ | Dal {row['Start'].date()} al {row['End'].date()} ({row['Duration'].days} giorni)")

    # 5. GRAFICO EQUITY GENERALE
    plt.figure(figsize=(12, 6))
    plt.plot(df_sorted.index, df_sorted['Equity'], label='Portfolio Equity', color='#2ecc71')
    plt.title('Curva Equity Portafoglio Aggregata (EUR)')
    plt.ylabel('Bilancio (€)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_path = 'data/reports/comparative_analysis/portfolio_stats.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"\n✅ Grafico salvato in: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()