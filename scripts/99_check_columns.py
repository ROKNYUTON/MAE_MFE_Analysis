import pandas as pd

# Percorso del tuo file
file_path = 'data/reports/Portfolio_aggregato.html'

try:
    print(f"📂 Analizzo il file: {file_path}")
    dfs = pd.read_html(file_path)
    print(f"ℹ️ Trovate {len(dfs)} tabelle nel file HTML.\n")

    for i, df in enumerate(dfs):
        # Stampa solo se la tabella ha più di 1 riga (per ignorare intestazioni vuote)
        if len(df) > 1:
            print(f"--- TABELLA {i} ---")
            print(f"Nomi Colonne trovati: \n{list(df.columns)}")
            print("Esempio prima riga dati:")
            print(df.iloc[0].values)
            print("-" * 30 + "\n")
            
except Exception as e:
    print(f"❌ Errore: {e}")