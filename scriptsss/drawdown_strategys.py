"""
ANALISI DRAWDOWN PER STRATEGIA CON ML E VISUALIZZAZIONI COMPLETE
Analizza i top 20 e top 100 drawdown per ogni strategia separatamente.
Correla ogni drawdown con fattori esterni (VIX, news, asset correlations).
Utilizza ML per identificare fattori impattanti e genera visualizzazioni.
Output: report dettagliati per strategia, grafici e modelli ML.
"""

from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
import warnings
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, classification_report
import shap
import seaborn as sns
warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
BASE = Path('.')
TRADES_CSV = BASE / 'data' / 'reports' / 'multi_asset_report.csv'
NEWS_CSV = BASE / 'data' / 'reports' / 'news_analysis' / 'ml_dataset_news_events.csv'
VIX_LOCAL = BASE / 'data' / 'datasets' / 'VIX_D1_2020_2025.csv'

OUTPUT_DIR = BASE / 'data' / 'reports' / 'strategy_drawdown_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_BALANCE = 100000.0
TOP_N_DRAWDOWNS = [20, 100]  # Analizza top 20 e top 100

# ---------------- logging ----------------
def info(msg): print(f"[INFO] {msg}")
def warn(msg): print(f"[WARN] {msg}")

# ---------------- Data Loading ----------------
def load_trades():
    """Carica il file multi_asset_report.csv e verifica la colonna Strategy."""
    if not TRADES_CSV.exists():
        raise FileNotFoundError(f"Trades CSV not found: {TRADES_CSV}")
    
    df = pd.read_csv(TRADES_CSV)
    df.columns = [c.strip() for c in df.columns]
    
    # Verifica colonne obbligatorie
    required_cols = ['Open time', 'Close time', 'Profit/Loss (Global)']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Colonna obbligatoria mancante: {col}")
    
    # Verifica colonna strategia
    strategy_col = None
    for col in df.columns:
        if 'strategy' in col.lower() or 'Strategy' in col:
            strategy_col = col
            break
    
    if strategy_col is None:
        warn("Nessuna colonna 'Strategy' trovata. Creo strategia unica 'ALL'.")
        df['Strategy'] = 'ALL'
        strategy_col = 'Strategy'
    
    # Converti timestamp
    df['Open time'] = pd.to_datetime(df['Open time'], dayfirst=True, errors='coerce')
    df['Close time'] = pd.to_datetime(df['Close time'], dayfirst=True, errors='coerce')
    
    # PnL
    df['PL'] = df['Profit/Loss (Global)'].astype(float)
    
    info(f"Trades caricati: {len(df)} righe, {df[strategy_col].nunique()} strategie")
    info(f"Strategie trovate: {df[strategy_col].unique()[:10]}")
    
    return df, strategy_col

def load_news():
    """Carica eventi news dal CSV."""
    p = NEWS_CSV
    if not p.exists():
        warn(f"News CSV non trovato: {p}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(p)
        if 'event_time' in df.columns:
            df['time'] = pd.to_datetime(df['event_time'], errors='coerce')
        elif 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
        else:
            warn("Nessuna colonna tempo trovata nel news CSV")
            return pd.DataFrame()
        
        if 'event_type' in df.columns:
            df['event_type'] = df['event_type'].str.upper()
        else:
            df['event_type'] = 'UNKNOWN'
        
        df = df.dropna(subset=['time'])
        info(f"News caricate: {len(df)} eventi")
        return df[['time', 'event_type']]
    
    except Exception as e:
        warn(f"Errore caricamento news: {e}")
        return pd.DataFrame()

def load_vix():
    """Carica dati VIX da Yahoo Finance con fallback locale."""
    # Prova prima Yahoo Finance per dati più recenti
    try:
        info("Download VIX da Yahoo Finance...")
        vix = yf.download("^VIX", period="5y", interval="1d", progress=False)
        if not vix.empty:
            vix['VIX'] = vix['Close']
            vix.index = pd.to_datetime(vix.index)
            info(f"VIX da Yahoo: {len(vix)} righe, {vix.index.min()} a {vix.index.max()}")
            return vix[['VIX']]
    except Exception as e:
        warn(f"Yahoo VIX fallito: {e}")
    
    # Fallback a file locale
    if VIX_LOCAL.exists():
        try:
            df = pd.read_csv(VIX_LOCAL)
            cols = [c.replace('<','').replace('>','').strip() for c in df.columns]
            df.columns = cols
            
            if 'DATE' in df.columns and 'TIME' in df.columns:
                df['time'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str), errors='coerce')
                df.set_index('time', inplace=True)
            elif 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
                df.set_index('time', inplace=True)
            
            if 'CLOSE' in df.columns:
                df['VIX'] = df['CLOSE']
            elif 'Close' in df.columns:
                df['VIX'] = df['Close']
            
            df = df[['VIX']].dropna()
            df.index = pd.to_datetime(df.index)  # FORZA conversione a DatetimeIndex
            info(f"VIX da locale: {len(df)} righe")
            return df
        except Exception as e:
            warn(f"Errore caricamento VIX locale: {e}")
    
    warn("Nessun dato VIX disponibile")
    return pd.DataFrame()

# ---------------- Equity per Strategia ----------------
def build_strategy_equity(trades_df, strategy_name, strategy_col):
    """Costruisce equity curve per una singola strategia."""
    strategy_trades = trades_df[trades_df[strategy_col] == strategy_name].copy()
    
    if len(strategy_trades) == 0:
        warn(f"Nessun trade per strategia {strategy_name}")
        return None
    
    # Ordina per tempo di chiusura
    strategy_trades = strategy_trades.sort_values('Close time').reset_index(drop=True)
    
    # Costruisci equity
    balances = []
    bal = float(INITIAL_BALANCE)
    
    # Punto iniziale
    first_close = strategy_trades['Close time'].iloc[0]
    balances.append((first_close - timedelta(seconds=1), INITIAL_BALANCE))
    
    for _, trade in strategy_trades.iterrows():
        bal += float(trade['PL'])
        balances.append((trade['Close time'], bal))
    
    # Crea serie temporale
    df_bal = pd.DataFrame(balances, columns=['time', 'balance'])
    df_bal['time'] = pd.to_datetime(df_bal['time'])
    df_bal = df_bal.groupby('time').last().sort_index()
    
    # Riempimento orario
    if len(df_bal) > 1:
        full_idx = pd.date_range(start=df_bal.index.min(), end=df_bal.index.max(), freq='1h')
        equity_series = df_bal['balance'].reindex(full_idx, method='ffill')
    else:
        equity_series = df_bal['balance']
    
    equity_series.name = 'equity'
    info(f"Strategia {strategy_name}: {len(strategy_trades)} trades, equity finale €{equity_series.iloc[-1]:.2f}")
    
    return equity_series

# ---------------- Drawdown Detection ----------------
def detect_drawdowns(equity_series):
    """Rileva tutti i drawdown in una serie di equity."""
    if equity_series is None or len(equity_series) < 2:
        return []
    
    equity = equity_series.dropna()
    peak = equity.expanding(min_periods=1).max()
    drawdown = (equity - peak) / peak * 100  # in percentuale
    
    # Trova inizio e fine drawdown
    drawdown_events = []
    in_drawdown = False
    current_start = None
    current_peak = None
    
    for i in range(1, len(equity)):
        if drawdown.iloc[i] < 0 and not in_drawdown:
            # Inizio drawdown
            in_drawdown = True
            current_start = equity.index[i-1]  # Picco è il punto prima
            current_peak = equity.iloc[i-1]
            current_trough = equity.index[i]
            current_trough_value = equity.iloc[i]
            current_min_dd = drawdown.iloc[i]
        
        elif in_drawdown:
            if drawdown.iloc[i] < current_min_dd:
                # Nuovo minimo
                current_trough = equity.index[i]
                current_trough_value = equity.iloc[i]
                current_min_dd = drawdown.iloc[i]
            
            if equity.iloc[i] >= current_peak:
                # Fine drawdown (recupero)
                drawdown_events.append({
                    'peak_time': current_start,
                    'peak_value': current_peak,
                    'trough_time': current_trough,
                    'trough_value': current_trough_value,
                    'recovery_time': equity.index[i],
                    'recovery_value': equity.iloc[i],
                    'drawdown_pct': current_min_dd,
                    'drawdown_eur': current_trough_value - current_peak,
                    'duration_days': (current_trough - current_start).total_seconds() / (24*3600)
                })
                in_drawdown = False
    
    # Se ancora in drawdown alla fine
    if in_drawdown:
        drawdown_events.append({
            'peak_time': current_start,
            'peak_value': current_peak,
            'trough_time': current_trough,
            'trough_value': current_trough_value,
            'recovery_time': None,
            'recovery_value': None,
            'drawdown_pct': current_min_dd,
            'drawdown_eur': current_trough_value - current_peak,
            'duration_days': (current_trough - current_start).total_seconds() / (24*3600)
        })
    
    return drawdown_events

# ---------------- Enrich Drawdown with External Factors ----------------
def enrich_drawdown_with_vix(drawdown_events, vix_df):
    """Arricchisce i drawdown con dati VIX."""
    if vix_df.empty:
        for event in drawdown_events:
            event.update({
                'vix_at_peak': np.nan,
                'vix_at_trough': np.nan,
                'vix_mean_dd': np.nan,
                'vix_max_dd': np.nan,
                'vix_regime': 'UNKNOWN'
            })
        return drawdown_events
    
    for event in drawdown_events:
        start = event['peak_time']
        end = event['trough_time']
        
        # VIX al picco
        vix_peak = vix_df.loc[vix_df.index <= start, 'VIX']
        event['vix_at_peak'] = float(vix_peak.iloc[-1]) if len(vix_peak) > 0 else np.nan
        
        # VIX al trough
        vix_trough = vix_df.loc[vix_df.index <= end, 'VIX']
        event['vix_at_trough'] = float(vix_trough.iloc[-1]) if len(vix_trough) > 0 else np.nan
        
        # VIX durante il drawdown
        mask = (vix_df.index >= start) & (vix_df.index <= end)
        vix_during = vix_df.loc[mask, 'VIX']
        
        if len(vix_during) > 0:
            event['vix_mean_dd'] = float(vix_during.mean())
            event['vix_max_dd'] = float(vix_during.max())
            
            # Classifica regime VIX
            vix_mean = event['vix_mean_dd']
            if vix_mean < 15:
                event['vix_regime'] = 'LOW'
            elif vix_mean < 20:
                event['vix_regime'] = 'MEDIUM'
            elif vix_mean < 25:
                event['vix_regime'] = 'HIGH'
            elif vix_mean < 30:
                event['vix_regime'] = 'VERY_HIGH'
            else:
                event['vix_regime'] = 'EXTREME'
        else:
            event['vix_mean_dd'] = np.nan
            event['vix_max_dd'] = np.nan
            event['vix_regime'] = 'UNKNOWN'
    
    return drawdown_events

def enrich_drawdown_with_news(drawdown_events, news_df):
    """Arricchisce i drawdown con dati news."""
    if news_df.empty:
        for event in drawdown_events:
            event.update({
                'news_count': 0,
                'nfp_count': 0,
                'cpi_count': 0,
                'fomc_count': 0,
                'news_during_dd': 'NO'
            })
        return drawdown_events
    
    for event in drawdown_events:
        start = event['peak_time']
        end = event['trough_time']
        
        # News durante il drawdown
        mask = (news_df['time'] >= start) & (news_df['time'] <= end)
        news_during = news_df.loc[mask]
        
        event['news_count'] = len(news_during)
        event['nfp_count'] = len(news_during[news_during['event_type'] == 'NFP'])
        event['cpi_count'] = len(news_during[news_during['event_type'] == 'CPI'])
        event['fomc_count'] = len(news_during[news_during['event_type'] == 'FOMC'])
        event['news_during_dd'] = 'YES' if len(news_during) > 0 else 'NO'
    
    return drawdown_events

# ---------------- Statistical Analysis ----------------
def analyze_drawdown_correlations(drawdown_df):
    """Analizza correlazioni tra drawdown e fattori esterni."""
    results = {}
    
    if len(drawdown_df) < 3:
        return results
    
    # Correlazione drawdown magnitude vs VIX
    if 'drawdown_eur' in drawdown_df.columns and 'vix_mean_dd' in drawdown_df.columns:
        corr, pval = stats.pearsonr(
            drawdown_df['drawdown_eur'].abs(),
            drawdown_df['vix_mean_dd'].fillna(0)
        )
        results['corr_dd_vix'] = {
            'correlation': float(corr),
            'p_value': float(pval),
            'significant': pval < 0.05
        }
    
    # Distribuzione per regime VIX
    if 'vix_regime' in drawdown_df.columns:
        regime_stats = {}
        for regime in drawdown_df['vix_regime'].unique():
            regime_dd = drawdown_df[drawdown_df['vix_regime'] == regime]
            if len(regime_dd) > 0:
                regime_stats[regime] = {
                    'count': len(regime_dd),
                    'mean_dd_eur': float(regime_dd['drawdown_eur'].abs().mean()),
                    'max_dd_eur': float(regime_dd['drawdown_eur'].abs().max()),
                    'pct_of_total': len(regime_dd) / len(drawdown_df) * 100
                }
        results['vix_regime_distribution'] = regime_stats
    
    # News impact
    if 'news_during_dd' in drawdown_df.columns:
        news_dd = drawdown_df[drawdown_df['news_during_dd'] == 'YES']
        no_news_dd = drawdown_df[drawdown_df['news_during_dd'] == 'NO']
        
        if len(news_dd) > 0 and len(no_news_dd) > 0:
            results['news_impact'] = {
                'with_news': {
                    'count': len(news_dd),
                    'mean_dd_eur': float(news_dd['drawdown_eur'].abs().mean()),
                    'pct_of_total': len(news_dd) / len(drawdown_df) * 100
                },
                'without_news': {
                    'count': len(no_news_dd),
                    'mean_dd_eur': float(no_news_dd['drawdown_eur'].abs().mean()),
                    'pct_of_total': len(no_news_dd) / len(drawdown_df) * 100
                }
            }
    
    return results

# ---------------- Machine Learning Analysis ----------------
def perform_ml_analysis(drawdown_df, strategy_name):
    """Esegue analisi ML per identificare fattori che impattano i drawdown."""
    if len(drawdown_df) < 10:
        warn(f"Troppi pochi drawdown ({len(drawdown_df)}) per analisi ML per {strategy_name}")
        return None
    
    # Prepara i dati
    features = []
    target_reg = []  # Regressione: magnitudo del drawdown
    target_clf = []  # Classificazione: drawdown severo (sopra mediana)
    
    for _, row in drawdown_df.iterrows():
        # Feature: VIX mean, news count, duration
        feature_vec = [
            row.get('vix_mean_dd', 0),
            row.get('news_count', 0),
            row.get('nfp_count', 0),
            row.get('cpi_count', 0),
            row.get('fomc_count', 0),
            row.get('duration_days', 0)
        ]
        features.append(feature_vec)
        
        # Target regressione: valore assoluto del drawdown in EUR
        target_reg.append(abs(row.get('drawdown_eur', 0)))
        
        # Target classificazione: 1 se drawdown > mediana, 0 altrimenti
        target_clf.append(1 if abs(row.get('drawdown_eur', 0)) > drawdown_df['drawdown_eur'].abs().median() else 0)
    
    features = np.array(features)
    target_reg = np.array(target_reg)
    target_clf = np.array(target_clf)
    
    # Split dati
    if len(features) < 20:
        # Troppo pochi dati per split, usa tutti per training
        X_train, X_test = features, features
        y_reg_train, y_reg_test = target_reg, target_reg
        y_clf_train, y_clf_test = target_clf, target_clf
    else:
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
            features, target_reg, target_clf, test_size=0.3, random_state=42
        )
    
    # Feature names
    feature_names = ['vix_mean', 'news_count', 'nfp_count', 'cpi_count', 'fomc_count', 'duration_days']
    
    # 1. Regressione con RandomForest
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train, y_reg_train)
    y_pred = rf_reg.predict(X_test)
    mae = mean_absolute_error(y_reg_test, y_pred)
    
    # Feature importance per regressione
    reg_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_reg.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 2. Classificazione con RandomForest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_clf_train)
    y_pred_clf = rf_clf.predict(X_test)
    clf_report = classification_report(y_clf_test, y_pred_clf, output_dict=True)
    
    # Feature importance per classificazione
    clf_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 3. SHAP analysis (se abbastanza dati)
    shap_values = None
    if len(features) >= 20:
        try:
            explainer = shap.TreeExplainer(rf_reg)
            shap_values = explainer.shap_values(X_test[:min(50, len(X_test))])
        except:
            shap_values = None
    
    ml_results = {
        'regression': {
            'mae': mae,
            'feature_importance': reg_importance.to_dict('records'),
            'predictions': {
                'actual': y_reg_test.tolist(),
                'predicted': y_pred.tolist()
            }
        },
        'classification': {
            'report': clf_report,
            'feature_importance': clf_importance.to_dict('records'),
            'accuracy': clf_report['accuracy']
        },
        'shap_available': shap_values is not None,
        'feature_names': feature_names
    }
    
    # Salva risultati ML
    ml_df = pd.DataFrame({
        'feature': feature_names,
        'reg_importance': rf_reg.feature_importances_,
        'clf_importance': rf_clf.feature_importances_
    }).sort_values('reg_importance', ascending=False)
    
    ml_df.to_csv(OUTPUT_DIR / f'ml_importance_{strategy_name.replace("/", "_")}.csv', index=False)
    
    # Crea grafico feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(ml_df['feature'], ml_df['reg_importance'])
    plt.xlabel('Importance')
    plt.title(f'Feature Importance (Regression) - {strategy_name}')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'ml_importance_{strategy_name.replace("/", "_")}.png', dpi=150)
    plt.close()
    
    return ml_results

# ---------------- Visualizations ----------------
def create_detailed_visualizations(equity, drawdown_events, news_df, vix_df, strategy_name):
    """Crea visualizzazioni dettagliate per una strategia."""
    safe_name = strategy_name.replace("/", "_").replace("\\", "_")
    
    # 1. Equity curve con drawdown e news
    plt.figure(figsize=(14, 10))
    
    # Subplot 1: Equity con drawdown
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(equity.index, equity.values, 'b-', linewidth=1, label='Equity')
    
    # Evidenzia i 5 drawdown più grandi
    if len(drawdown_events) > 0:
        dd_df = pd.DataFrame(drawdown_events)
        top5 = dd_df.nlargest(5, 'drawdown_eur', keep='all')
        
        for i, (_, dd) in enumerate(top5.iterrows()):
            start = dd['peak_time']
            end = dd['trough_time']
            mask = (equity.index >= start) & (equity.index <= end)
            if mask.any():
                ax1.fill_between(equity.index[mask], equity[mask].min(), equity[mask].max(),
                               alpha=0.3, color='red', label=f'DD {i+1}' if i == 0 else "")
    
    # Aggiungi eventi news
    if not news_df.empty:
        for _, news in news_df.iterrows():
            ax1.axvline(x=news['time'], color='orange', alpha=0.3, linestyle='--', linewidth=0.5)
    
    ax1.set_title(f'Equity Curve con Drawdown e News - {strategy_name}')
    ax1.set_ylabel('Equity (€)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: VIX durante il periodo
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    if not vix_df.empty:
        # Resample VIX per allineare con equity
        vix_resampled = vix_df['VIX'].resample('D').mean().reindex(equity.index, method='ffill')
        ax2.plot(vix_resampled.index, vix_resampled.values, 'g-', linewidth=1, label='VIX')
        ax2.set_ylabel('VIX')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Drawdown magnitude nel tempo
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    if len(drawdown_events) > 0:
        dd_times = [dd['trough_time'] for dd in drawdown_events]
        dd_magnitudes = [abs(dd['drawdown_eur']) for dd in drawdown_events]
        ax3.scatter(dd_times, dd_magnitudes, color='red', alpha=0.7, s=50, label='Drawdown Magnitude')
        ax3.set_ylabel('Drawdown (€)')
        ax3.set_xlabel('Data')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'equity_detailed_{safe_name}.png', dpi=150)
    plt.close()
    
    # 2. Heatmap correlazioni
    if len(drawdown_events) >= 5:
        dd_df = pd.DataFrame(drawdown_events)
        
        # Seleziona colonne numeriche per correlazione
        numeric_cols = dd_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 3:
            corr_matrix = dd_df[numeric_cols].corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title(f'Correlazioni Drawdown - {strategy_name}')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f'correlation_heatmap_{safe_name}.png', dpi=150)
            plt.close()
    
    # 3. Boxplot drawdown per regime VIX
    if len(drawdown_events) >= 5:
        dd_df = pd.DataFrame(drawdown_events)
        if 'vix_regime' in dd_df.columns and 'drawdown_eur' in dd_df.columns:
            plt.figure(figsize=(10, 6))
            plot_data = dd_df[['vix_regime', 'drawdown_eur']].copy()
            plot_data['drawdown_eur_abs'] = plot_data['drawdown_eur'].abs()
            sns.boxplot(x='vix_regime', y='drawdown_eur_abs', data=plot_data)
            plt.title(f'Distribuzione Drawdown per Regime VIX - {strategy_name}')
            plt.xlabel('Regime VIX')
            plt.ylabel('Drawdown (€, abs)')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f'boxplot_vix_regime_{safe_name}.png', dpi=150)
            plt.close()

# ---------------- Strategy Analysis ----------------
def analyze_strategy(strategy_name, trades_df, strategy_col, vix_df, news_df):
    """Analizza una singola strategia in dettaglio."""
    info(f"\nAnalisi strategia: {strategy_name}")
    
    # 1. Costruisci equity
    equity = build_strategy_equity(trades_df, strategy_name, strategy_col)
    if equity is None:
        return None
    
    # 2. Rileva drawdown
    drawdown_events = detect_drawdowns(equity)
    info(f"  Drawdown rilevati: {len(drawdown_events)}")
    
    if len(drawdown_events) == 0:
        return None
    
    # 3. Arricchisci con VIX e news
    drawdown_events = enrich_drawdown_with_vix(drawdown_events, vix_df)
    drawdown_events = enrich_drawdown_with_news(drawdown_events, news_df)
    
    # 4. Converti in DataFrame
    dd_df = pd.DataFrame(drawdown_events)
    
    # 5. Analisi per top N drawdown
    strategy_results = {}
    
    for top_n in TOP_N_DRAWDOWNS:
        if len(dd_df) >= top_n:
            top_dd = dd_df.nlargest(top_n, 'drawdown_eur', keep='all')
        else:
            top_dd = dd_df.copy()
        
        # Salva CSV
        safe_name = strategy_name.replace("/", "_").replace("\\", "_")
        top_dd.to_csv(OUTPUT_DIR / f'drawdown_top{top_n}_{safe_name}.csv', index=False)
        
        # Analisi statistica
        stats = analyze_drawdown_correlations(top_dd)
        
        # ML Analysis (solo per top 20 se abbastanza dati)
        ml_results = None
        if top_n == 20 and len(top_dd) >= 10:
            ml_results = perform_ml_analysis(top_dd, strategy_name)
        
        strategy_results[f'top_{top_n}'] = {
            'drawdown_count': len(top_dd),
            'total_dd_eur': float(top_dd['drawdown_eur'].abs().sum()),
            'avg_dd_eur': float(top_dd['drawdown_eur'].abs().mean()),
            'max_dd_eur': float(top_dd['drawdown_eur'].abs().max()),
            'stats': stats,
            'ml_results': ml_results
        }
    
    # 6. Metriche complessive strategia
    total_trades = len(trades_df[trades_df[strategy_col] == strategy_name])
    winning_trades = len(trades_df[(trades_df[strategy_col] == strategy_name) & (trades_df['PL'] > 0)])
    
    strategy_summary = {
        'strategy_name': strategy_name,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
        'total_pnl': float(trades_df[trades_df[strategy_col] == strategy_name]['PL'].sum()),
        'avg_trade_pnl': float(trades_df[trades_df[strategy_col] == strategy_name]['PL'].mean()),
        'final_equity': float(equity.iloc[-1]),
        'total_return_pct': (float(equity.iloc[-1]) - INITIAL_BALANCE) / INITIAL_BALANCE * 100,
        'total_drawdowns': len(drawdown_events),
        'analysis_results': strategy_results
    }
    
    # 7. Crea visualizzazioni dettagliate
    create_detailed_visualizations(equity, drawdown_events, news_df, vix_df, strategy_name)
    
    # 8. Genera report testuale con consigli
    generate_strategy_recommendations(strategy_summary, strategy_results, strategy_name)
    
    return strategy_summary

def generate_strategy_recommendations(strategy_summary, strategy_results, strategy_name):
    """Genera report con consigli per la strategia."""
    safe_name = strategy_name.replace("/", "_").replace("\\", "_")
    
    report_lines = []
    report_lines.append(f"REPORT STRATEGIA: {strategy_name}")
    report_lines.append("="*60)
    
    # Performance generale
    report_lines.append("\n1. PERFORMANCE GENERALE:")
    report_lines.append(f"   Trades totali: {strategy_summary['total_trades']}")
    report_lines.append(f"   Win Rate: {strategy_summary['win_rate']:.1f}%")
    report_lines.append(f"   PnL Totale: €{strategy_summary['total_pnl']:,.0f}")
    report_lines.append(f"   Equity Finale: €{strategy_summary['final_equity']:,.0f}")
    report_lines.append(f"   Rendimento Totale: {strategy_summary['total_return_pct']:.1f}%")
    report_lines.append(f"   Drawdown totali: {strategy_summary['total_drawdowns']}")
    
    # Analisi drawdown top 20
    if 'top_20' in strategy_results:
        top20 = strategy_results['top_20']
        report_lines.append("\n2. ANALISI TOP 20 DRAWDOWN:")
        report_lines.append(f"   Drawdown medio (top20): €{top20['avg_dd_eur']:,.0f}")
        report_lines.append(f"   Drawdown massimo: €{top20['max_dd_eur']:,.0f}")
        
        # Statistiche VIX
        if 'stats' in top20 and 'corr_dd_vix' in top20['stats']:
            corr = top20['stats']['corr_dd_vix']
            report_lines.append(f"\n   CORRELAZIONE CON VIX:")
            report_lines.append(f"     Coefficiente: {corr['correlation']:.3f}")
            report_lines.append(f"     Significativo: {'SÌ' if corr['significant'] else 'NO'}")
            
            if corr['significant']:
                if corr['correlation'] > 0.3:
                    report_lines.append("     → I drawdown sono correlati con VIX alto")
                    report_lines.append("     CONSIGLIO: Considera ridurre exposure quando VIX > 25")
                elif corr['correlation'] < -0.3:
                    report_lines.append("     → I drawdown sono correlati con VIX basso")
                    report_lines.append("     CONSIGLIO: Attenzione quando VIX è molto basso (<15)")
        
        # Distribuzione regime VIX
        if 'stats' in top20 and 'vix_regime_distribution' in top20['stats']:
            regime_stats = top20['stats']['vix_regime_distribution']
            report_lines.append(f"\n   DISTRIBUZIONE PER REGIME VIX:")
            for regime, stats in regime_stats.items():
                if stats['count'] > 0:
                    report_lines.append(f"     {regime}: {stats['count']} drawdown ({stats['pct_of_total']:.1f}%), media €{stats['mean_dd_eur']:,.0f}")
        
        # Impatto news
        if 'stats' in top20 and 'news_impact' in top20['stats']:
            news = top20['stats']['news_impact']
            report_lines.append(f"\n   IMPATTO NEWS:")
            report_lines.append(f"     Con news: {news['with_news']['count']} drawdown ({news['with_news']['pct_of_total']:.1f}%), media €{news['with_news']['mean_dd_eur']:,.0f}")
            report_lines.append(f"     Senza news: {news['without_news']['count']} drawdown ({news['without_news']['pct_of_total']:.1f}%), media €{news['without_news']['mean_dd_eur']:,.0f}")
            
            if news['with_news']['mean_dd_eur'] > news['without_news']['mean_dd_eur'] * 1.5:
                report_lines.append("     → I drawdown con news sono più severi!")
                report_lines.append("     CONSIGLIO: Considera pausa trading 1h prima/dopo news major")
    
    # Consigli ML
    if 'top_20' in strategy_results and 'ml_results' in strategy_results['top_20']:
        ml_results = strategy_results['top_20']['ml_results']
        if ml_results is not None:
            report_lines.append("\n3. ANALISI MACHINE LEARNING:")
            
            # Feature importance per regressione
            if 'regression' in ml_results and 'feature_importance' in ml_results['regression']:
                report_lines.append("   FEATURE IMPORTANCE (regressione):")
                for feat in ml_results['regression']['feature_importance'][:3]:
                    report_lines.append(f"     {feat['feature']}: {feat['importance']:.3f}")
                
                # Consigli basati sulla feature più importante
                top_feature = ml_results['regression']['feature_importance'][0]['feature']
                top_importance = ml_results['regression']['feature_importance'][0]['importance']
                
                report_lines.append(f"\n   CONSIGLI BASATI SU ML:")
                report_lines.append(f"     Feature più importante: {top_feature} ({top_importance:.3f})")
                
                if 'vix' in top_feature.lower():
                    report_lines.append("     → Il VIX è il fattore più predittivo dei drawdown")
                    report_lines.append("     AZIONE: Implementa regole di position sizing basate su VIX")
                elif 'news' in top_feature.lower() or 'nfp' in top_feature.lower() or 'cpi' in top_feature.lower():
                    report_lines.append("     → Le news sono il fattore più predittivo dei drawdown")
                    report_lines.append("     AZIONE: Evita trading durante eventi economici major")
                elif 'duration' in top_feature.lower():
                    report_lines.append("     → La durata è correlata con drawdown severi")
                    report_lines.append("     AZIONE: Implementa stop loss temporali")
            
            # Accuratezza classificazione
            if 'classification' in ml_results:
                report_lines.append(f"   ACCURATEZZA CLASSIFICAZIONE: {ml_results['classification']['accuracy']:.3f}")
    
    # Consigli generali
    report_lines.append("\n4. RACCOMANDAZIONI OPERATIVE:")
    
    # Basato su win rate
    if strategy_summary['win_rate'] < 40:
        report_lines.append("   • Win Rate basso (<40%): considera strategie più selettive")
    elif strategy_summary['win_rate'] > 60:
        report_lines.append("   • Win Rate eccellente (>60%): mantieni l'approccio attuale")
    
    # Basato su drawdown
    if 'top_20' in strategy_results:
        avg_dd = strategy_results['top_20']['avg_dd_eur']
        if avg_dd > 5000:
            report_lines.append("   • Drawdown medi molto elevati (>€5,000): riduci position size del 30%")
        elif avg_dd > 2000:
            report_lines.append("   • Drawdown medi elevati (>€2,000): riduci position size del 15%")
    
    # Basato su numero trades
    if strategy_summary['total_trades'] > 1000:
        report_lines.append("   • Alto numero di trades: valida la qualità vs quantità")
    elif strategy_summary['total_trades'] < 100:
        report_lines.append("   • Basso numero di trades: campione statistico limitato")
    
    report_lines.append("\n" + "="*60)
    
    # Salva report
    report_path = OUTPUT_DIR / f'recommendations_{safe_name}.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    info(f"  Report consigli salvato: {report_path}")
    
    return report_lines

# ---------------- Main Analysis ----------------
def run_complete_analysis():
    """Esegue l'analisi completa per tutte le strategie."""
    info("="*60)
    info("ANALISI DRAWDOWN PER STRATEGIA CON ML")
    info("="*60)
    
    # 1. Carica dati
    info("\n1. Caricamento dati...")
    trades_df, strategy_col = load_trades()
    news_df = load_news()
    vix_df = load_vix()
    
    # 2. Identifica strategie
    strategies = trades_df[strategy_col].unique()
    info(f"\n2. Strategie identificate: {len(strategies)}")
    
    # 3. Analizza ogni strategia
    all_results = {}
    
    for strategy in strategies:
        try:
            result = analyze_strategy(strategy, trades_df, strategy_col, vix_df, news_df)
            if result is not None:
                all_results[strategy] = result
        except Exception as e:
            warn(f"Errore analisi strategia {strategy}: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. Crea report comparativo
    if all_results:
        create_comparative_report(all_results)
    
    info(f"\nAnalisi completata. Risultati salvati in: {OUTPUT_DIR}")
    
    # 5. Crea dashboard riassuntiva
    create_summary_dashboard(all_results)
    
    return all_results

def create_comparative_report(strategy_results):
    """Crea report comparativo tra tutte le strategie."""
    
    # Estrai metriche chiave
    summary_rows = []
    
    for strategy, data in strategy_results.items():
        # Metriche base
        row = {
            'strategy': strategy,
            'total_trades': data['total_trades'],
            'win_rate': data['win_rate'],
            'total_pnl': data['total_pnl'],
            'final_equity': data['final_equity'],
            'total_return_pct': data['total_return_pct'],
            'total_drawdowns': data['total_drawdowns']
        }
        
        # Metriche dai top drawdown (usa top 20 se disponibile)
        if 'top_20' in data['analysis_results']:
            top20 = data['analysis_results']['top_20']
            row.update({
                'avg_dd_top20': top20['avg_dd_eur'],
                'max_dd_top20': top20['max_dd_eur'],
                'total_dd_eur_top20': top20['total_dd_eur']
            })
            
            # Aggiungi correlazioni se disponibili
            if 'stats' in top20 and 'corr_dd_vix' in top20['stats']:
                corr = top20['stats']['corr_dd_vix']
                row['corr_dd_vix'] = corr['correlation']
                row['vix_corr_sig'] = corr['significant']
        
        summary_rows.append(row)
    
    # Crea DataFrame
    summary_df = pd.DataFrame(summary_rows)
    
    # Ordina per performance
    summary_df = summary_df.sort_values('total_return_pct', ascending=False)
    
    # Salva CSV
    summary_df.to_csv(OUTPUT_DIR / 'strategy_comparison.csv', index=False)
    
    # Crea report di sintesi
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("REPORT COMPARATIVO STRATEGIE - ANALISI DRAWDOWN CON ML")
    report_lines.append("="*80)
    report_lines.append(f"Strategie analizzate: {len(summary_df)}")
    report_lines.append(f"Periodo analisi: {summary_df['total_trades'].sum()} trades totali")
    report_lines.append("")
    
    # Top 5 strategie per rendimento
    report_lines.append("TOP 5 STRATEGIE PER RENDIMENTO:")
    report_lines.append("-"*40)
    for i, (_, row) in enumerate(summary_df.head(5).iterrows()):
        report_lines.append(f"{i+1}. {row['strategy']}:")
        report_lines.append(f"   Rendimento: {row['total_return_pct']:.1f}%")
        report_lines.append(f"   Trades: {row['total_trades']}, Win Rate: {row['win_rate']:.1f}%")
        report_lines.append(f"   PnL Totale: €{row['total_pnl']:,.0f}")
        report_lines.append(f"   Drawdown totali: {row['total_drawdowns']}")
        if 'avg_dd_top20' in row:
            report_lines.append(f"   Avg DD (top20): €{row['avg_dd_top20']:,.0f}")
        report_lines.append("")
    
    # Analisi correlazioni VIX
    report_lines.append("ANALISI CORRELAZIONE DRAWDOWN-VIX:")
    report_lines.append("-"*40)
    
    if 'corr_dd_vix' in summary_df.columns:
        # Separa strategie con correlazione significativa
        sig_corr = summary_df[summary_df['vix_corr_sig'] == True]
        if len(sig_corr) > 0:
            report_lines.append(f"Strategie con correlazione significativa VIX-Drawdown: {len(sig_corr)}")
            for _, row in sig_corr.iterrows():
                corr_dir = "positiva" if row['corr_dd_vix'] > 0 else "negativa"
                strength = "forte" if abs(row['corr_dd_vix']) > 0.5 else "moderata" if abs(row['corr_dd_vix']) > 0.3 else "debole"
                report_lines.append(f"  • {row['strategy']}: {corr_dir} {strength} (r={row['corr_dd_vix']:.2f})")
        else:
            report_lines.append("Nessuna correlazione significativa trovata tra VIX e drawdown.")
    else:
        report_lines.append("Dati correlazione VIX non disponibili.")
    
    # Raccomandazioni generali
    report_lines.append("")
    report_lines.append("RACCOMANDAZIONI GENERALI:")
    report_lines.append("-"*40)
    
    # Identifica strategie resilienti (alto rendimento, basso drawdown)
    if 'avg_dd_top20' in summary_df.columns:
        summary_df['dd_performance_ratio'] = summary_df['total_return_pct'] / summary_df['avg_dd_top20'].abs().clip(lower=1)
        resilient = summary_df.nlargest(3, 'dd_performance_ratio')
        
        report_lines.append("Strategie più resilienti (alto rendimento/basso drawdown):")
        for i, (_, row) in enumerate(resilient.iterrows()):
            report_lines.append(f"  {i+1}. {row['strategy']}: Ratio={row['dd_performance_ratio']:.2f}")
    
    # Identifica strategie rischiose
    risky = summary_df.nlargest(3, 'avg_dd_top20')
    report_lines.append("\nStrategie più rischiose (drawdown medi più alti):")
    for i, (_, row) in enumerate(risky.iterrows()):
        report_lines.append(f"  {i+1}. {row['strategy']}: Avg DD €{row['avg_dd_top20']:,.0f}")
    
    # Consigli basati su analisi
    report_lines.append("\nCONSIGLI OPERATIVI:")
    report_lines.append("  1. Per strategie con alta correlazione VIX: implementa regole di sizing basate su VIX")
    report_lines.append("  2. Per strategie con drawdown elevati: riduci position size del 20-30%")
    report_lines.append("  3. Diversifica tra strategie con correlazioni diverse a VIX/news")
    report_lines.append("  4. Monitora performance in periodi di news economiche major")
    
    # Salva report
    report_path = OUTPUT_DIR / 'strategies_comparative_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    info(f"Report comparativo salvato: {report_path}")
    
    return summary_df

def create_summary_dashboard(strategy_results):
    """Crea una dashboard riassuntiva delle strategie."""
    if not strategy_results:
        return
    
    # Prepara dati per grafici
    strategies = list(strategy_results.keys())
    returns = [s['total_return_pct'] for s in strategy_results.values()]
    win_rates = [s['win_rate'] for s in strategy_results.values()]
    
    # Drawdown medi (top 20)
    avg_dds = []
    for s in strategy_results.values():
        if 'top_20' in s['analysis_results']:
            avg_dds.append(s['analysis_results']['top_20']['avg_dd_eur'])
        else:
            avg_dds.append(0)
    
    # Crea figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Rendimenti per strategia
    axes[0, 0].barh(strategies, returns, color='skyblue')
    axes[0, 0].set_xlabel('Rendimento (%)')
    axes[0, 0].set_title('Rendimento Totale per Strategia')
    axes[0, 0].axvline(x=0, color='black', linewidth=0.5)
    
    # 2. Win Rate per strategia
    axes[0, 1].barh(strategies, win_rates, color='lightgreen')
    axes[0, 1].set_xlabel('Win Rate (%)')
    axes[0, 1].set_title('Win Rate per Strategia')
    axes[0, 1].set_xlim(0, 100)
    
    # 3. Drawdown medi per strategia
    axes[1, 0].barh(strategies, avg_dds, color='salmon')
    axes[1, 0].set_xlabel('Drawdown Medio (€, abs)')
    axes[1, 0].set_title('Drawdown Medio (Top 20) per Strategia')
    
    # 4. Scatter plot: Rendimento vs Drawdown
    axes[1, 1].scatter(avg_dds, returns, s=100, alpha=0.6)
    axes[1, 1].set_xlabel('Drawdown Medio (€)')
    axes[1, 1].set_ylabel('Rendimento (%)')
    axes[1, 1].set_title('Rendimento vs Drawdown')
    axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 1].axvline(x=0, color='black', linewidth=0.5)
    
    # Aggiungi etichette strategie allo scatter plot
    for i, strategy in enumerate(strategies):
        axes[1, 1].annotate(strategy, (avg_dds[i], returns[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'summary_dashboard.png', dpi=150)
    plt.close()
    
    info(f"Dashboard riassuntiva salvata: {OUTPUT_DIR}/summary_dashboard.png")

# ========== MAIN ==========
if __name__ == "__main__":
    run_complete_analysis()