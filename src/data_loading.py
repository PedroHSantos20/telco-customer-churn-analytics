"""
Utility functions for loading the Telco churn data.

This module centralizes the logic to read the original Excel files
from data/raw and return a single consolidated Polars DataFrame.

Having this in /src makes it easier to reuse the same loading logic
in different notebooks or future scripts.
"""

from pathlib import Path
import polars as pl


def load_telco_data(raw_data_dir: str = "data/raw") -> pl.DataFrame:
    """
    Load and merge the Telco churn datasets from the specified raw data folder.

    Parameters
    ----------
    raw_data_dir : str, optional
        Path to the folder containing the original Excel files,
        by default "data/raw".

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame with the merged Telco customer data.

    Raises
    ------
    FileNotFoundError
        If one or more expected Excel files are missing.
    """
    raw_path = Path(raw_data_dir)

    # Expected files (adjust names if yours differ)
    files = {
        "demographics": "Telco_customer_churn_demographics.xlsx",
        "services": "Telco_customer_churn_services.xlsx",
        "status": "Telco_customer_churn_status.xlsx",
        "population": "Telco_customer_churn_population.xlsx",
        "location": "Telco_customer_churn_location.xlsx",
    }

    for key, fname in files.items():
        if not (raw_path / fname).exists():
            raise FileNotFoundError(f"Missing file for {key}: {raw_path / fname}")

    # Load all datasets
    df_demo = pl.read_excel(raw_path / files["demographics"])
    df_serv = pl.read_excel(raw_path / files["services"])
    df_stat = pl.read_excel(raw_path / files["status"])
    df_pop = pl.read_excel(raw_path / files["population"])
    df_loc = pl.read_excel(raw_path / files["location"])

    # Merge step by step on Customer ID / Zip Code (adapt if needed)
    df = (
        df_demo
        .join(df_serv, on="Customer ID", how="left")
        .join(df_stat, on="Customer ID", how="left")
        .join(df_loc, on="Zip Code", how="left")
        .join(df_pop, on="Zip Code", how="left")
    )

    return df
