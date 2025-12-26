# 3_comparative_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# --- CONFIGURAZIONE ---
REPORT_PATH = 'data/reports/multi_asset_report.csv'
BENCHMARK_PATH = 'data/reports/comparative_analysis/benchmarks_1h_equity.csv'
OUTPUT_DIR = 'data/reports/comparative_analysis'
INITIAL_BALANCE = 100000.0  # Base 100k

MARKET_EVENTS = [
    ('2020-02-20', '2020-03-23', 'COVID Crash', '#ffcccc', 0.92),
    ('2020-03-24', '2021-12-31', 'Post-COVID QE', '#ccffcc', 0.96),
    ('2022-01-01', '2022-10-13', 'Inflation/QT', '#fff2cc', 0.92),
    ('2022-10-14', '2023-12-31', 'AI Recovery', '#e6e6ff', 0.96),
]

def calculate_stats_monetary(df_val, df_pct):
    """Calcola statistiche con Max Drawdown in Euro"""
    stats_text = "STATISTICHE (Capitale 100k)\n" + "-"*30 + "\n"
    for col in df_val.columns:
        # Rendimento Totale %
        total_ret_pct = df_pct[col].iloc[-1]
        
        # Max Drawdown in EURO
        # Calcolo: Valore Attuale - Massimo Storico Precedente
        rolling_max = df_val[col].cummax()
        drawdown_euro = df_val[col] - rolling_max
        max_dd_euro = drawdown_euro.min()
        
        # Formattazione riga
        stats_text += f"{col:12}: {total_ret_pct:>6.1f}% Ret | DD: {max_dd_euro:>9.0f}€\n"
    return stats_text

def load_data():
    if not os.path.exists(REPORT_PATH):
        print("❌ File portafoglio non trovato!")
        return None, None
    
    # 1. Carica Portfolio
    df_p = pd.read_csv(REPORT_PATH)
    df_p.columns = [c.strip() for c in df_p.columns]
    df_p['Close time'] = pd.to_datetime(df_p['Close time'], dayfirst=True)
    if df_p['Profit/Loss (Global)'].dtype == object:
        df_p['Profit/Loss (Global)'] = df_p['Profit/Loss (Global)'].str.replace(',', '.').astype(float)
    
    df_p = df_p.sort_values('Close time')
    # Resample orario dei profitti
    p_series = df_p.set_index('Close time')['Profit/Loss (Global)'].resample('1H').sum().fillna(0)
    p_equity = INITIAL_BALANCE + p_series.cumsum()
    
    # 2. Carica Benchmark
    df_b = pd.read_csv(BENCHMARK_PATH, index_col=0, parse_dates=True)
    
    # 3. Sincronizzazione
    merged = pd.concat([p_equity, df_b], axis=1)
    merged.columns = ['Portfolio'] + list(df_b.columns)
    
    # Forza ri-basamento a 100k per sicurezza
    for col in merged.columns:
        if merged[col].first_valid_index() is None: continue
        start_val = merged[col].dropna().iloc[0]
        merged[col] = (merged[col] / start_val) * INITIAL_BALANCE
        
    merged = merged.ffill().dropna()
    merged_pct = ((merged / INITIAL_BALANCE) - 1) * 100
    
    return merged, merged_pct

def plot_final_monetary(df_val, df_pct):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 18))
    colors = {'S&P 500': '#27ae60', 'NASDAQ 100': '#2980b9', 'Gold': '#f1c40f', 'Portfolio': 'black'}
    
    stats_summary = calculate_stats_monetary(df_val, df_pct)

    # --- PLOT 1: PERCENTUALE ---
    for col in df_pct.columns:
        lw, z, alpha = (3.5, 100, 1.0) if col == 'Portfolio' else (1.5, 1, 0.7)
        ax1.plot(df_pct.index, df_pct[col], label=col, color=colors.get(col, 'gray'), lw=lw, alpha=alpha, zorder=z)
    
    ax1.set_title("RENDIMENTO PERCENTUALE (%)", fontsize=15, fontweight='bold')
    ax1.set_ylabel("Variazione %")
    ax1.legend(loc='upper left')

    # --- PLOT 2: MONETARIO (LOG) ---
    for col in df_val.columns:
        lw, z, alpha = (3.5, 100, 1.0) if col == 'Portfolio' else (1.5, 1, 0.7)
        ax2.plot(df_val.index, df_val[col], label=col, color=colors.get(col, 'gray'), lw=lw, alpha=alpha, zorder=z)
    
    ax2.set_yscale('log')
    ax2.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax2.set_title(f"EQUITY CURVE (€) - BASE {INITIAL_BALANCE:,.0f}€", fontsize=15, fontweight='bold')
    ax2.set_ylabel("Euro")

    # Box statistiche
    ax2.text(0.98, 0.02, stats_summary, transform=ax2.transAxes, 
             fontsize=11, family='monospace', verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black'))

    # Griglia ed Eventi
    for ax in [ax1, ax2]:
        ax.grid(True, which='both', alpha=0.2)
        for start, end, label, color, y_pos in MARKET_EVENTS:
            s, e = pd.to_datetime(start), pd.to_datetime(end)
            if s < df_val.index.max():
                ax.axvspan(s, e, color=color, alpha=0.15, zorder=0)
                ax.text(s + (e-s)/2, y_pos, label, transform=ax.get_xaxis_transform(),
                        ha='center', fontweight='bold', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.tight_layout(pad=4.0)
    out_path = f"{OUTPUT_DIR}/COMPARATIVE_MONETARY_STATS.png"
    plt.savefig(out_path, dpi=200)
    print(f"✅ Grafico salvato: {out_path}")
    print("\n" + stats_summary)

if __name__ == "__main__":
    v, p = load_data()
    if v is not None:
        plot_final_monetary(v, p)