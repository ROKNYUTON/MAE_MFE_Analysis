# 4_advanced_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

# --- CONFIGURAZIONE ---
REPORT_PATH = 'data/reports/multi_asset_report.csv'
BENCHMARK_PATH = 'data/reports/comparative_analysis/benchmarks_1h_equity.csv'
OUTPUT_DIR = 'data/reports/advanced_analysis'
INITIAL_BALANCE = 100000.0 

# Dati Storici CPI-U (Consumer Price Index All Urban Consumers)
CPI_HISTORY = {
    '2020-01': 257.97, '2020-02': 258.67, '2020-03': 258.11, '2020-04': 256.38, '2020-05': 256.39, '2020-06': 257.79,
    '2020-07': 259.10, '2020-08': 259.91, '2020-09': 260.28, '2020-10': 260.38, '2020-11': 260.22, '2020-12': 260.47,
    '2021-01': 261.58, '2021-02': 263.01, '2021-03': 264.87, '2021-04': 267.05, '2021-05': 269.19, '2021-06': 271.69,
    '2021-07': 273.00, '2021-08': 273.56, '2021-09': 274.31, '2021-10': 276.58, '2021-11': 277.94, '2021-12': 278.80,
    '2022-01': 281.14, '2022-02': 283.71, '2022-03': 287.50, '2022-04': 289.10, '2022-05': 292.29, '2022-06': 296.31,
    '2022-07': 296.27, '2022-08': 296.17, '2022-09': 296.80, '2022-10': 298.01, '2022-11': 297.71, '2022-12': 296.79,
    '2023-01': 299.17, '2023-02': 300.84, '2023-03': 301.83, '2023-04': 303.36, '2023-05': 304.12, '2023-06': 305.10,
    '2023-07': 305.69, '2023-08': 307.02, '2023-09': 307.78, '2023-10': 307.67, '2023-11': 307.05, '2023-12': 306.74,
    '2024-01': 308.41, '2024-02': 310.32, '2024-03': 312.33, '2024-04': 313.54, '2024-05': 314.06, '2024-06': 314.17,
    '2024-07': 314.54, '2024-08': 314.79, '2024-09': 315.30, '2024-10': 315.66, '2024-11': 315.50, '2024-12': 315.80,
    '2025-01': 316.50
}

def load_data():
    """Carica e unifica i dati (Portafoglio + Benchmark)"""
    if not os.path.exists(REPORT_PATH) or not os.path.exists(BENCHMARK_PATH):
        print("❌ File mancanti. Esegui script 2 e 3.")
        return None

    # 1. Portfolio
    df_p = pd.read_csv(REPORT_PATH)
    df_p.columns = [c.strip() for c in df_p.columns]
    df_p['Close time'] = pd.to_datetime(df_p['Close time'], dayfirst=True)
    if df_p['Profit/Loss (Global)'].dtype == object:
        df_p['Profit/Loss (Global)'] = df_p['Profit/Loss (Global)'].str.replace(',', '.').astype(float)
    
    df_p = df_p.sort_values('Close time')
    # FIX: '1H' -> '1h' per evitare FutureWarning
    p_series = df_p.set_index('Close time')['Profit/Loss (Global)'].resample('1h').sum().fillna(0)
    p_equity = INITIAL_BALANCE + p_series.cumsum()
    
    # 2. Benchmark
    df_b = pd.read_csv(BENCHMARK_PATH, index_col=0, parse_dates=True)
    
    # 3. Merge
    merged = pd.concat([p_equity, df_b], axis=1)
    merged.columns = ['Portfolio'] + list(df_b.columns)
    
    # Normalizza benchmark su 100k
    for col in merged.columns:
        if col != 'Portfolio':
             merged[col] = (merged[col] / merged[col].dropna().iloc[0]) * INITIAL_BALANCE
    
    return merged.ffill().dropna()

def analyze_inflation(df):
    """Calcola Equity Reale vs Nominale (Metodo sicuro per dimensioni)"""
    print("📊 Calcolo impatto inflazione...")
    
    # 1. Prepara Serie CPI interpolata giornaliera
    cpi_dates = pd.to_datetime(list(CPI_HISTORY.keys()), format='%Y-%m')
    cpi_values = list(CPI_HISTORY.values())
    # Interpoliamo per avere un valore CPI stimato per ogni giorno
    cpi_daily = pd.Series(cpi_values, index=cpi_dates).resample('D').interpolate(method='linear')
    
    # 2. Mappatura CPI sul DataFrame principale
    # Creiamo una colonna temporanea 'Date' (senza orario) per fare il match
    temp_df = df.copy()
    temp_df['Date_Only'] = temp_df.index.normalize()
    
    # Mappiamo il valore CPI giornaliero su ogni riga oraria
    temp_df['CPI_Value'] = temp_df['Date_Only'].map(cpi_daily)
    
    # Riempiamo eventuali buchi (es. date future o mancanti) con l'ultimo valore valido
    temp_df['CPI_Value'] = temp_df['CPI_Value'].ffill().bfill()
    
    # 3. Calcolo Deflator e Real Equity
    start_cpi = temp_df['CPI_Value'].iloc[0]
    # Se start_cpi è 0 o nan, evitiamo divisioni per zero
    if pd.isna(start_cpi) or start_cpi == 0:
        start_cpi = 1
        
    deflator = start_cpi / temp_df['CPI_Value']
    real_equity = temp_df['Portfolio'] * deflator
    
    # 4. Plot
    plt.figure(figsize=(14, 8))
    # Ora df.index e real_equity hanno GARANTITO la stessa lunghezza perché derivano dallo stesso df
    plt.plot(df.index, df['Portfolio'], label='Equity Nominale (Cifra sul conto)', color='blue', lw=2)
    plt.plot(df.index, real_equity, label='Equity Reale (Potere d\'acquisto)', color='red', lw=2, linestyle='--')
    
    plt.fill_between(df.index, df['Portfolio'], real_equity, color='red', alpha=0.1, label='Perdita da Inflazione')
    
    # Stats
    nom_ret = (df['Portfolio'].iloc[-1] / INITIAL_BALANCE - 1) * 100
    real_ret = (real_equity.iloc[-1] / INITIAL_BALANCE - 1) * 100
    inflation_drag = nom_ret - real_ret
    
    plt.title(f"Impatto Inflazione: Nominale +{nom_ret:.1f}% vs Reale +{real_ret:.1f}%", fontsize=16)
    plt.ylabel('Valore EUR (100k Start)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.text(0.02, 0.95, f"Inflazione accumulata: -{inflation_drag:.1f}%", transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    out_path = f'{OUTPUT_DIR}/inflation_impact.png'
    plt.savefig(out_path)
    print(f"✅ Grafico Inflazione salvato: {out_path}")

def analyze_yearly_breakdown(df):
    """Genera grafico comparativo anno per anno (ognuno parte da 100k)"""
    print("📅 Generazione breakdown annuale...")
    
    years = df.index.year.unique()
    n_rows = (len(years) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(18, 6 * n_rows))
    axes = axes.flatten()
    
    colors = {'Portfolio': 'black', 'S&P 500': '#27ae60', 'NASDAQ 100': '#2980b9', 'Gold': '#f1c40f'}
    
    for i, year in enumerate(years):
        ax = axes[i]
        
        # Filtra dati anno corrente
        subset = df[df.index.year == year].copy()
        
        if len(subset) < 10: continue 
        
        stats_txt = f"{year} STATS:\n"
        
        for col in subset.columns:
            # Ribase a 100k
            start_val = subset[col].iloc[0]
            subset[col] = (subset[col] / start_val) * INITIAL_BALANCE
            
            # Calcolo metriche anno
            ret = (subset[col].iloc[-1] / INITIAL_BALANCE - 1) * 100
            mdd = (subset[col] - subset[col].cummax()).min()
            
            lw = 2.5 if col == 'Portfolio' else 1.5
            alpha = 1.0 if col == 'Portfolio' else 0.6
            ax.plot(subset.index, subset[col], label=col, color=colors.get(col, 'gray'), lw=lw, alpha=alpha)
            
            stats_txt += f"{col:10} {ret:>6.1f}% | DD {mdd:>6.0f}€\n"
        
        ax.set_title(f"Performance Anno {year} (Base 100k)", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        
        ax.text(0.98, 0.02, stats_txt, transform=ax.transAxes, 
                fontsize=9, family='monospace', va='bottom', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        if i == 0: ax.legend(loc='upper left')

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    out_path = f'{OUTPUT_DIR}/yearly_breakdown_100k.png'
    plt.savefig(out_path, dpi=150)
    print(f"✅ Grafico Annuale salvato: {out_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Carica dati unificati
    df = load_data()
    if df is None: return
    
    # 2. Analisi Inflazione
    analyze_inflation(df)
    
    # 3. Analisi Anno per Anno
    analyze_yearly_breakdown(df)

if __name__ == "__main__":
    main()
    