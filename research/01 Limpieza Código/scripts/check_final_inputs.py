from pathlib import Path

import pandas as pd


REQUIRED_FILES = [
    Path("data/raw/Leads UVG proyecto01.xlsx"),
    Path("data/raw/Leads UVG proyecto02.xlsx"),
    Path("data/raw/leads_final_combined.csv"),
    Path("data/processed/leads_cleaned_final.csv"),
    Path("data/features/leads_enriched_final.csv"),
]

PT2_FILES = [
    Path("data/raw/data_pt2/Leads_proyecto03_UVG.csv"),
    Path("data/raw/data_pt2/Leads_proyecto03_UVG.xlsx"),
    Path("data/raw/data_pt2/Leads_proyecto05_UVG.csv"),
    Path("data/raw/data_pt2/Leads_proyecto04_UVG.csv"),
]

REQUIRED_FILES.extend(PT2_FILES)


def verify_exists_and_not_empty(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"Archivo vacio: {path}")


def read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)

    return pd.read_csv(
        path,
        sep=";",
        encoding="utf-8",
        low_memory=False,
        on_bad_lines="skip",
    )


def main() -> None:
    shapes = {}

    try:
        for path in REQUIRED_FILES:
            verify_exists_and_not_empty(path)

        for path in PT2_FILES:
            df = read_input(path)
            rows, cols = df.shape

            if rows <= 1_000 or cols <= 50:
                raise ValueError(
                    f"{path} no cumple umbral: shape=({rows}, {cols}); "
                    "se requieren > 1000 filas y > 50 columnas"
                )

            shapes[str(path)] = (rows, cols)

    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc

    print("OK: todos los inputs finales verificados")
    for path, shape in shapes.items():
        print(f"{path}: shape={shape}")


if __name__ == "__main__":
    main()
