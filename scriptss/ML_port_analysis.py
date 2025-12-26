import pandas as pd
import numpy as np
from polygon import RESTClient
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import shap
import matplotlib.pyplot as plt
import os
from datetime import datetime

# --- CONFIGURAZIONE ---
REPORT_PATH = 'data/reports/multi_asset_report.csv'  # Path al tuo CSV trades
OUTPUT_DIR = 'data/reports/portfolio_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Features esterne con mapping Polygon tickers
EXTERNAL_TICKERS = {
    'VIX': 'I:VIX',
    'Treasury_Yield': 'B:US10Y',
    'Oil': 'X:CLUSD',
    'BTC': 'X:BTCUSD',
    'EURUSD': 'C:EURUSD'
}

# Periodo dati
START_DATE = '2020-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')

class PortfolioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PortfolioNN(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(PortfolioNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)  # Output: predicted return
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_portfolio_data():
    """Carica trades e calcola daily returns del portafoglio."""
    df_trades = pd.read_csv(REPORT_PATH)
    df_trades.columns = [c.strip() for c in df_trades.columns]
    df_trades['Close time'] = pd.to_datetime(df_trades['Close time'], dayfirst=True)
    df_trades.set_index('Close time', inplace=True)
    
    # Aggrega daily P&L
    daily_pl = df_trades['Profit/Loss (Global)'].resample('D').sum().fillna(0)
    daily_returns = daily_pl / 100000.0  # Normalize per initial capital ~100k
    return daily_returns

def fetch_external_data():
    """Fetcha dati da Polygon API."""
    client = RESTClient()  # API key configured
    data = {}
    for name, ticker in EXTERNAL_TICKERS.items():
        try:
            aggs = client.get_aggs(ticker, 1, "day", START_DATE, END_DATE)
            if aggs:
                agg_dicts = [a.dict() for a in aggs]
                df = pd.DataFrame(agg_dicts)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df['close'].pct_change().fillna(0)
                data[name] = df
            else:
                print(f"No data for {ticker}")
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    
    # Aggiungi SPX
    try:
        aggs_spx = client.get_aggs('I:SPX', 1, "day", START_DATE, END_DATE)
        if aggs_spx:
            agg_dicts = [a.dict() for a in aggs_spx]
            df_spx = pd.DataFrame(agg_dicts)
            df_spx['timestamp'] = pd.to_datetime(df_spx['timestamp'], unit='ms')
            df_spx.set_index('timestamp', inplace=True)
            data['SPX'] = df_spx['close'].pct_change().fillna(0)
        else:
            print("No data for I:SPX")
    except Exception as e:
        print(f"Error fetching SPX: {e}")
    
    if not data:
        raise ValueError("No external data fetched. Check tickers and dates.")
    
    df_external = pd.DataFrame(data)
    return df_external

def prepare_features(target_returns, external_df):
    """Prepara dataset: features + lags."""
    df = external_df.reindex(target_returns.index).fillna(0)
    
    # Aggiungi lags (es. 1-5 days)
    for col in df.columns:
        for lag in range(1, 6):
            df[f'{col}_lag{lag}'] = df[col].shift(lag).fillna(0)
    
    # Allinea con target
    df['target'] = target_returns
    df = df.dropna()
    
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, df.drop('target', axis=1).columns, scaler

def train_model(X, y, input_size):
    """Train PyTorch NN."""
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    train_dataset = PortfolioDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = PortfolioNN(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(200):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
    
    # Eval
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE Test: {rmse:.4f}")
    
    return model

def explain_model(model, X, feature_names, scaler):
    """Usa SHAP per feature importance."""
    background_size = min(32, len(X))  # Riduci a 32 per evitare mismatch
    background = torch.tensor(X[:background_size], dtype=torch.float32)
    explainer = shap.DeepExplainer(model, background)  # Sample per speed
    shap_values = explainer.shap_values(torch.tensor(X, dtype=torch.float32))
    
    # Aggrega abs mean SHAP
    importance = np.abs(shap_values).mean(0).squeeze()
    df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    df_importance = df_importance.sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(df_importance['Feature'][:15], df_importance['Importance'][:15])
    plt.title('Top Features Influencing Portfolio Returns')
    plt.xlabel('SHAP Importance')
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png")
    plt.close()
    
    return df_importance

def generate_insights(df_importance):
    """Output concreti per migliorare."""
    top_positive = df_importance[df_importance['Importance'] > 0].head(5)['Feature'].tolist()  # Assume positive impact high
    top_negative = df_importance.tail(5)['Feature'].tolist()  # Low importance or negative
    
    print("\nInsights per Migliorare:")
    print("Fattori Positivi (Influenzano Gains):")
    for f in top_positive:
        if 'VIX' in f:
            print("- Low Volatility (VIX basso): Aumenta exposure in mercati calmi.")
        elif 'SPX' in f:
            print("- Equity Bull (S&P up): Focus su US100/US500 in uptrends.")
        # Aggiungi custom per altri
    
    print("Fattori Negativi (Causano Losses/DD):")
    for f in top_negative:
        if 'lag' in f:
            print(f"- Lags in {f}: Monitora momentum; usa MA cross per entries.")
        elif 'Yield' in f:
            print("- Rising Rates: Hedge con shorts o reduce duration.")
    
    print("Suggerimenti:")
    print("- Se VIX alto (>20), riduci size 50% o pausa trades.")
    print("- Diversifica: Aggiungi inverse ETF (es. SQQQ) per equity DD.")
    print("- ML Alert: Predict next return con model, threshold per trades.")

if __name__ == "__main__":
    # Step 1: Carica portfolio returns
    portfolio_returns = load_portfolio_data()
    
    # Step 2: Fetch external
    external_df = fetch_external_data()
    
    # Step 3: Prepara data
    X, y, feature_names, scaler = prepare_features(portfolio_returns, external_df)
    
    # Step 4: Train
    model = train_model(X, y, X.shape[1])
    
    # Step 5: Explain
    df_importance = explain_model(model, X, feature_names, scaler)
    print(df_importance.head(10))  # Top 10
    
    # Step 6: Insights
    generate_insights(df_importance)
    
    print(f"Analisi salvata in {OUTPUT_DIR}")
