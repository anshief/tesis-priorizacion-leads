import hashlib
from pathlib import Path

import pandas as pd

from cleaning_helpers import normalize_project as canonicalize_project_name
from schema_contract import (
    keep_human_readable_labels,
    validate_analytical_label_schema,
)


BASE_INPUT_DIR = Path("data/raw")
BASE_INPUT_PATTERN = "Leads UVG *.xlsx"
PT2_PATH = Path("data/raw/data_pt2/harmonized/pt2_pt3_merged.csv")
OUTPUT_PATH = Path("data/raw/leads_final_combined.csv")

REQUIRED_BASE_PROJECTS = {"proyecto01", "proyecto02", "proyecto03", "proyecto05", "proyecto04"}


def discover_base_input_files() -> list[Path]:
    files = sorted(BASE_INPUT_DIR.glob(BASE_INPUT_PATTERN))
    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos {BASE_INPUT_PATTERN} en {BASE_INPUT_DIR}"
        )
    return files


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def read_base_inputs(files: list[Path]) -> pd.DataFrame:
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Faltan archivos base: {', '.join(missing)}")

    frames = [pd.read_excel(path) for path in files]
    columns = []
    for frame in frames:
        for column in frame.columns:
            if column not in columns:
                columns.append(column)

    aligned = [frame.reindex(columns=columns) for frame in frames]
    return pd.concat(aligned, ignore_index=True, sort=False)


def normalize_project(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def prepare_base_projects(df: pd.DataFrame) -> pd.DataFrame:
    if "Proyecto" not in df.columns:
        raise ValueError("Los archivos base no tienen columna Proyecto")

    prepared = df.copy()
    prepared["Proyecto"] = normalize_project(prepared["Proyecto"])
    prepared = prepared.loc[prepared["Proyecto"].notna() & prepared["Proyecto"].ne("")].copy()
    # fuente_pt2 identifica la fuente/proyecto de origen del lead. Para las filas que
    # vienen de los Excel base (proyecto01, proyecto02) se usa el nombre canónico del
    # proyecto en vez del genérico "base_excel", para que sean distinguibles entre sí.
    prepared["fuente_pt2"] = prepared["Proyecto"].apply(canonicalize_project_name)
    return prepared


def align_schemas(
    df_base: pd.DataFrame,
    df_pt2: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_cols = sorted(set(df_base.columns) | set(df_pt2.columns))
    return df_base.reindex(columns=all_cols), df_pt2.reindex(columns=all_cols)


def dedupe_by_id(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "id_Trato" not in df.columns:
        raise ValueError("No existe id_Trato para deduplicar")

    before = len(df)
    work = df.copy()
    work["_non_null_count"] = work.notna().sum(axis=1)
    work["_original_order"] = range(len(work))
    work = work.sort_values(
        ["id_Trato", "_non_null_count", "_original_order"],
        ascending=[True, False, True],
        na_position="last",
    )
    work = work.drop_duplicates(subset=["id_Trato"], keep="first")
    work = work.sort_values("_original_order").drop(columns=["_non_null_count", "_original_order"])
    return work.reset_index(drop=True), before - len(work)


def validate_output(
    combined: pd.DataFrame,
    base_raw_rows: int,
    base_input_hashes: dict[Path, str],
    obra_gris_count: int,
) -> None:
    if len(combined) <= base_raw_rows:
        raise ValueError(
            f"Output final tiene {len(combined)} filas; debe ser mayor que base "
            f"proyecto01/proyecto02 ({base_raw_rows})"
        )

    projects = set(combined["Proyecto"].dropna().astype(str))
    missing_projects = REQUIRED_BASE_PROJECTS - projects
    if missing_projects:
        raise ValueError(f"Faltan proyectos en output final: {sorted(missing_projects)}")

    for path, original_hash in base_input_hashes.items():
        if file_md5(path) != original_hash:
            raise RuntimeError(f"{path} fue modificado durante la combinacion")

    duplicated_ids = int(combined["id_Trato"].duplicated().sum())
    if duplicated_ids:
        raise ValueError(f"Output final tiene duplicados por id_Trato: {duplicated_ids}")

    if obra_gris_count <= 0:
        raise ValueError("Obra gris no esta presente en el output final")

    validate_analytical_label_schema(combined, OUTPUT_PATH.name)


def main() -> None:
    if not PT2_PATH.exists():
        raise FileNotFoundError(
            f"No existe {PT2_PATH}. Ejecuta 01_harmonize_new_exports.py y "
            "01b_harmonize_pt3_exports.py primero."
        )

    base_input_files = discover_base_input_files()
    base_input_hashes = {path: file_md5(path) for path in base_input_files}
    df_base_raw = read_base_inputs(base_input_files)
    df_pt2 = read_csv(PT2_PATH)

    df_base = prepare_base_projects(df_base_raw)
    df_base, _ = keep_human_readable_labels(df_base)
    df_pt2, _ = keep_human_readable_labels(df_pt2)
    df_base_aligned, df_pt2_aligned = align_schemas(df_base, df_pt2)
    combined_before_dedupe = pd.concat(
        [df_base_aligned, df_pt2_aligned],
        ignore_index=True,
        sort=False,
    )
    combined, duplicates_removed = dedupe_by_id(combined_before_dedupe)
    obra_gris_count = int(combined["Etapa Trato"].eq("Obra gris").sum())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    validate_output(
        combined=combined,
        base_raw_rows=len(df_base_raw),
        base_input_hashes=base_input_hashes,
        obra_gris_count=obra_gris_count,
    )

    project_distribution = combined["Proyecto"].value_counts(dropna=False)

    print("COMBINACIÓN FINAL COMPLETADA")
    print(f"Archivos Excel base ({len(base_input_files)}): {len(df_base)} filas")
    print(f"PT2: {len(df_pt2)} filas")
    print(f"Combinado final: {combined.shape[0]} filas × {combined.shape[1]} columnas")
    print(f"Duplicados eliminados: {duplicates_removed}")
    print("Distribución por proyecto:")
    print(project_distribution.to_string())
    print(f"Obra gris conservado: {obra_gris_count}")


if __name__ == "__main__":
    main()
