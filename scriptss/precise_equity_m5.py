import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# --- CONFIGURAZIONE ---
REPORT_PATH = 'data/reports/multi_asset_report.csv'
DATASETS_FOLDER = 'data/datasets'
INITIAL_BALANCE = 100000.0
OUTPUT_DIR = 'data/reports/precise_m5_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTRACT_SIZE = {
    'GOLD.pro': 100, 'USDJPY.pro': 1000, 'US100.pro': 20,
    'US500.pro': 50, 'DE30.pro': 25,
}

def load_price_data(symbol):
    """Carica dati M5 senza resample."""
    path = os.path.join(DATASETS_FOLDER, f"{symbol}_M5_2020_2025.csv")
    if not os.path.exists(path): return None
    
    df = pd.read_csv(path, sep='\t')
    df.columns = [c.replace('<', '').replace('>', '') for c in df.columns]
    df['dt'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y.%m.%d %H:%M:%S')
    df.set_index('dt', inplace=True)
    df = df[['OPEN', 'HIGH', 'LOW', 'CLOSE']].astype(float)
    return df

def main():
    if not os.path.exists(REPORT_PATH):
        print("❌ Report non trovato")
        return

    print("🚀 Inizio calcolo Equity Precisa M5 Globale...")
    df_trades = pd.read_csv(REPORT_PATH)
    df_trades.columns = [c.strip() for c in df_trades.columns]
    df_trades['Open time'] = pd.to_datetime(df_trades['Open time'], dayfirst=True)
    df_trades['Close time'] = pd.to_datetime(df_trades['Close time'], dayfirst=True)
    
    all_symbols = df_trades['Symbol'].unique()
    m5_db = {}
    print("⏳ Caricamento Dataset Prezzi M5...")
    for s in all_symbols:
        data = load_price_data(s)
        if data is not None: m5_db[s] = data

    # Timeline M5 globale
    start_date = df_trades['Open time'].min()
    end_date = df_trades['Close time'].max()
    timeline_m5 = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    # Balance cumulativa su M5 (usa merge_asof per riempire backward)
    balance_changes = df_trades[['Close time', 'Profit/Loss (Global)']].copy().sort_values('Close time')
    balance_changes['Cumulative_Balance'] = INITIAL_BALANCE + balance_changes['Profit/Loss (Global)'].cumsum()
    
    high_res_port = pd.DataFrame(index=timeline_m5)
    high_res_port.index.name = 'time'
    merged_high = pd.merge_asof(high_res_port, balance_changes.rename(columns={'Close time': 'time'}), on='time', direction='backward')
    merged_high.set_index('time', inplace=True)
    merged_high['balance'] = merged_high['Cumulative_Balance'].fillna(INITIAL_BALANCE)
    merged_high['equity_close'] = merged_high['balance'].copy()  # Su CLOSE
    merged_high['equity_low'] = merged_high['balance'].copy()  # Su adjusted LOW

    print("🔄 Calcolo Equity Fluttuante M5 Precisa...")
    for _, trade in df_trades.iterrows():
        s = trade['Symbol']
        if s not in m5_db: continue
        mask = (merged_high.index >= trade['Open time']) & (merged_high.index < trade['Close time'])  # < close per escludere post-close
        if not mask.any(): continue
        p = m5_db[s].reindex(merged_high.index[mask], method='ffill')
        mult = trade['Size'] * CONTRACT_SIZE.get(s, 100)
        entry = trade['Open price']
        is_buy = 'Buy' in trade['Type']
        
        # Calcolo pl_close standard
        pl_close = (p['CLOSE'] - entry) * mult if is_buy else (entry - p['CLOSE']) * mult
        merged_high.loc[mask, 'equity_close'] += pl_close
        
        # Calcolo pl_low adjusted per close time
        if is_buy:
            pl_low = (p['LOW'] - entry) * mult
        else:
            pl_low = (entry - p['HIGH']) * mult
        
        # Adjust per trades che chiudono durante il bar: trova bar del close
        close_time = trade['Close time']
        close_bar_start_idx = merged_high.index.get_indexer([close_time], method='ffill')[0]
        close_bar_start = merged_high.index[close_bar_start_idx]
        
        if close_bar_start < close_time < close_bar_start + pd.Timedelta(minutes=5):
            # Bar del close: adjust if necessary
            pl_low_at_close_bar = pl_low.get(close_bar_start)
            realized_pl = trade['Profit/Loss (Global)']
            if pl_low_at_close_bar is not None and realized_pl > pl_low_at_close_bar:
                print(f"⚠️ Adjust per trade {trade.get('Ticket', 'N/A')}: pl_low {pl_low_at_close_bar:.2f} -> {realized_pl:.2f}")
                pl_low[close_bar_start] = realized_pl
        
        merged_high.loc[mask, 'equity_low'] += pl_low

    # Calcolo Max DD globale
    peak_close = merged_high['equity_close'].cummax()
    dd_close = merged_high['equity_close'] - peak_close
    max_dd_close = dd_close.min()
    time_dd_close = dd_close.idxmin()
    
    peak_low = merged_high['equity_low'].cummax()
    dd_low = merged_high['equity_low'] - peak_low
    max_dd_low = dd_low.min()
    time_dd_low = dd_low.idxmin()
    
    total_pl_global = merged_high['balance'].iloc[-1] - INITIAL_BALANCE
    
    print("\n" + "="*40)
    print(f"⚠️ MAX DRAWDOWN (M5 CLOSE): {max_dd_close:.2f} € a {time_dd_close}")
    print(f"⚠️ MAX DRAWDOWN (M5 LOW Adjusted): {max_dd_low:.2f} € a {time_dd_low}")
    print(f"💰 Total P&L Globale: {total_pl_global:.2f} €")
    print("="*40)

    # Grafico full equity M5
    plt.figure(figsize=(16, 8))
    plt.plot(merged_high.index, merged_high['balance'], color='blue', lw=1, label='Balance')
    plt.plot(merged_high.index, merged_high['equity_close'], color='green', lw=1, alpha=0.8, label='Equity (CLOSE)')
    plt.plot(merged_high.index, merged_high['equity_low'], color='red', lw=1, alpha=0.6, label='Equity (LOW Adjusted)')
    plt.scatter([time_dd_low], [merged_high.loc[time_dd_low]['equity_low']], color='black', s=100, label='Max DD Low')
    plt.title('Equity Curve M5 Precisa Globale')
    plt.ylabel('Equity (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Annotazione in basso a destra
    annot_text = f"Total P&L: {total_pl_global:.2f} € | Max DD: {max_dd_low:.2f} €"
    plt.annotate(annot_text, xy=(0.95, 0.05), xycoords='axes fraction', ha='right', va='bottom', fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    plt.savefig(f"{OUTPUT_DIR}/global_equity_m5.png")
    print(f"✅ Grafico Globale salvato: {OUTPUT_DIR}/global_equity_m5.png")
    plt.close()

    # Per anni
    years = merged_high.index.year.unique()
    for year in years:
        df_y = merged_high[merged_high.index.year == year]
        if df_y.empty: continue
        
        peak_y_low = df_y['equity_low'].cummax()
        dd_y_low = df_y['equity_low'] - peak_y_low
        max_dd_y_low = dd_y_low.min()
        time_dd_y_low = dd_y_low.idxmin()
        
        total_pl_year = df_y['balance'].iloc[-1] - df_y['balance'].iloc[0]
        
        print(f"Anno {year}: Max DD M5 LOW Adjusted {max_dd_y_low:.2f} € a {time_dd_y_low}")
        print(f"Anno {year}: Total P&L {total_pl_year:.2f} €")
        
        plt.figure(figsize=(14, 7))
        plt.plot(df_y.index, df_y['balance'], color='blue', lw=2, label='Balance')
        plt.plot(df_y.index, df_y['equity_close'], color='green', lw=1, label='Equity (CLOSE)')
        plt.plot(df_y.index, df_y['equity_low'], color='red', lw=1, label='Equity (LOW Adjusted)')
        plt.scatter([time_dd_y_low], [df_y.loc[time_dd_y_low]['equity_low']], color='black', s=100, label='Max DD Low')
        plt.title(f'Equity M5 Precisa {year}')
        plt.ylabel('Euro')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        
        # Annotazione in basso a destra
        annot_text = f"Total P&L: {total_pl_year:.2f} € | Max DD: {max_dd_y_low:.2f} €"
        plt.annotate(annot_text, xy=(0.95, 0.05), xycoords='axes fraction', ha='right', va='bottom', fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
        
        plt.savefig(f"{OUTPUT_DIR}/equity_m5_{year}.png")
        print(f"✅ Salvato Anno {year}")
        plt.close()

    # Analisi MAE vs Realized per trades con loss (opzionale, ma poiché menzionato, mantieni ma senza print massivi se troppi)
    print("\n🕵️ Analisi MAE vs Realized P/L...")
    count_overest = 0
    for _, trade in df_trades.iterrows():
        if trade['Profit/Loss (Global)'] >= 0: continue  # Solo losses
        s = trade['Symbol']
        if s not in m5_db: continue
        trade_mask = (m5_db[s].index >= trade['Open time']) & (m5_db[s].index <= trade['Close time'])
        if not trade_mask.any(): continue
        prices = m5_db[s].loc[trade_mask]
        entry = trade['Open price']
        mult = trade['Size'] * CONTRACT_SIZE.get(s, 100)
        is_buy = 'Buy' in trade['Type']
        
        if is_buy:
            mae_pl = (prices['LOW'].min() - entry) * mult
        else:
            mae_pl = (entry - prices['HIGH'].max()) * mult
        
        realized_pl = trade['Profit/Loss (Global)']
        if mae_pl < realized_pl:
            count_overest += 1
            # Print solo count, non tutti
    print(f"⚠️ Numero di trades con possibile overestimation MAE: {count_overest}")

if __name__ == "__main__":
    main()