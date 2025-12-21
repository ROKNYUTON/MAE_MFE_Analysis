import pandas as pd
import os

# Percorso aggiornato
csv_path = 'data/datasets/GOLD_M5_2020_2025.csv'

def load_and_clean_data():
    if os.path.exists(csv_path):
        # sep='\t' serve per dividere le colonne correttamente
        df = pd.read_csv(csv_path, sep='\t')
        
        print("✅ File caricato correttamente!")
        print(f"Numero di righe: {len(df)}")
        print("\nPrime 5 righe pulite:")
        print(df.head())
        
        # Pulizia nomi colonne (rimuove i simboli < >)
        df.columns = [c.replace('<', '').replace('>', '') for c in df.columns]
        return df
    else:
        print(f"❌ File non trovato in: {csv_path}")
        return None

if __name__ == "__main__":
    data = load_and_clean_data()