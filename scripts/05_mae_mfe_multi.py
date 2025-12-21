import pandas as pd
import os
from datetime import datetime

# Config
REPORT_PATH = 'data/reports/multi_asset_report.csv'
DATASETS_FOLDER = 'data/datasets'

# Monthly EUR/USD averages (1 EUR = X USD)
RATES = {
    '2020-01': 1.11,
    '2020-02': 1.091,
    '2020-03': 1.1067,
    '2020-04': 1.0868,
    '2020-05': 1.0911,
    '2020-06': 1.1256,
    '2020-07': 1.1466,
    '2020-08': 1.1831,
    '2020-09': 1.1786,
    '2020-10': 1.1772,
    '2020-11': 1.1895,
    '2020-12': 1.2176,
    '2021-01': 1.2165,
    '2021-02': 1.2095,
    '2021-03': 1.1899,
    '2021-04': 1.1968,
    '2021-05': 1.2142,
    '2021-06': 1.2037,
    '2021-07': 1.1825,
    '2021-08': 1.1766,
    '2021-09': 1.1768,
    '2021-10': 1.1597,
    '2021-11': 1.1409,
    '2021-12': 1.1304,
    '2022-01': 1.1317,
    '2022-02': 1.1336,
    '2022-03': 1.1007,
    '2022-04': 1.0822,
    '2022-05': 1.0576,
    '2022-06': 1.0564,
    '2022-07': 1.019,
    '2022-08': 1.0112,
    '2022-09': 0.9912,
    '2022-10': 0.9835,
    '2022-11': 1.0216,
    '2022-12': 1.0595,
    '2023-01': 1.0788,
    '2023-02': 1.0705,
    '2023-03': 1.0716,
    '2023-04': 1.0989,
    '2023-05': 1.0873,
    '2023-06': 1.0843,
    '2023-07': 1.1064,
    '2023-08': 1.0909,
    '2023-09': 1.0679,
    '2023-10': 1.0565,
    '2023-11': 1.0821,
    '2023-12': 1.0921,
    '2024-01': 1.0906,
    '2024-02': 1.0793,
    '2024-03': 1.0871,
    '2024-04': 1.0723,
    '2024-05': 1.081,
    '2024-06': 1.0762,
    '2024-07': 1.0848,
    '2024-08': 1.1022,
    '2024-09': 1.1105,
    '2024-10': 1.0896,
    '2024-11': 1.0633,
    '2024-12': 1.047,
    '2025-01': 1.0333,
    '2025-02': 1.0412,
    '2025-03': 1.0804,
    '2025-04': 1.1228,
    '2025-05': 1.1273,
    '2025-06': 1.153,
    '2025-07': 1.1682,
    '2025-08': 1.1653,
    '2025-09': 1.1733,
    '2025-10': 1.1638,
    '2025-11': 1.1561,
    '2025-12': 1.1687,
}

CONTRACT_SIZE = {
    'GOLD.pro': 100,
    'USDJPY.pro': 1000,  # Fixato per JPY pairs
    'US100.pro': 20,
    'US500.pro': 50,
    'DE30.pro': 25,
}

prices_cache = {}

# ... (il resto del codice rimane identico, solo cambia il calcolo profit_usd, mae_usd, mfe_usd usando rate = RATES.get(month_key, 1.10))

def calculate_mae_mfe(trade, df_prices):
    if df_prices is None:
        return None, None

    # ... (calcolo mae_points, mfe_points come prima)

    mae_eur = mae_points * size * contract
    mfe_eur = mfe_points * size * contract

    month_key = trade['Close time'].strftime('%Y-%m')
    rate = RATES.get(month_key, 1.10)

    mae_usd = round(mae_eur * rate, 2)
    mfe_usd = round(mfe_eur * rate, 2)

    return mae_usd, mfe_usd

def main():
    # ... (stesso, ma per profit_usd)
    month_key = trade['Close time'].strftime('%Y-%m')
    rate = RATES.get(month_key, 1.10)
    profit_usd = round(profit_eur * rate, 2)
    # ... resto uguale

# (fine script)