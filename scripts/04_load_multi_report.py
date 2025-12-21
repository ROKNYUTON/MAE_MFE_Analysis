import pandas as pd
import os

report_path = 'data/reports/multi_asset_report.csv'

def load_report():
    if not os.path.exists(report_path):
        print(f"❌ Report non trovato: {report_path}")
        return None

    df = pd.read_csv(report_path, sep=',')
    df.columns = [c.strip() for c in df.columns]  # Pulisci spazi

    print(f"✅ Caricato report con {len(df)} trade!")
    print(f"\n📋 Tutte le colonne ({len(df.columns)}):")
    print(list(df.columns))

    print("\n📄 Prime 8 righe complete:")
    print(df.head(8).to_string(index=False))

    print("\n🔖 Esempi di Strategy name (Global):")
    print(df['Strategy name (Global)'].unique()[:20])

    print("\n📊 Esempi di Symbol unici:")
    print(sorted(df['Symbol'].unique()))

    return df

if __name__ == "__main__":
    load_report()