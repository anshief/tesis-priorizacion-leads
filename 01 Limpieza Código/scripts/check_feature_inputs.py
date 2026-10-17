import pandas as pd
from pathlib import Path

from schema_contract import validate_analytical_label_schema


REFERENCE_DIR = Path("data/reference")
FEATURES_DIR = Path("data/features")
CLEANED_CSV = Path("data/processed/leads_cleaned_final.csv")


def main():
    for directory in (REFERENCE_DIR, FEATURES_DIR):
        if not directory.exists():
            raise FileNotFoundError(f"Carpeta no encontrada: {directory}")

    if not CLEANED_CSV.exists() or CLEANED_CSV.stat().st_size == 0:
        raise FileNotFoundError(f"Archivo no encontrado o vacío: {CLEANED_CSV}")

    df = pd.read_csv(CLEANED_CSV, low_memory=False)
    validate_analytical_label_schema(df, CLEANED_CSV.name)
    if "target_won" not in df.columns:
        raise AssertionError("target_won no existe en leads_cleaned_final.csv")

    rows, cols = df.shape
    print(f"OK: setup FE listo | {rows:,} filas × {cols} columnas")


if __name__ == "__main__":
    main()
