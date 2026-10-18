from pathlib import Path
import re
import unicodedata

import pandas as pd

from schema_contract import (
    keep_human_readable_labels,
    validate_analytical_label_schema,
)


INPUT_DIR = Path("data/raw/data_pt2")
OUTPUT_DIR = Path("data/raw/data_pt2/harmonized")
REPORT_PATH = Path("reports/final/label_replacement_report.xlsx")

PROYECTO_MAP = {
    "proyecto03": "proyecto03",
    "proyecto05": "proyecto05",
    "proyecto04": "proyecto04",
}


def discover_input_files() -> list[Path]:
    files = sorted(INPUT_DIR.glob("Leads_*_UVG.csv"))
    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos Leads_*_UVG.csv en {INPUT_DIR}"
        )
    return files


def source_name(path: Path) -> str:
    match = re.match(r"Leads_(.+)_UVG$", path.stem, flags=re.IGNORECASE)
    return match.group(1) if match else path.stem


def safe_filename(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_value.lower()).strip("_")

LABEL_RULE = [
    ("canal_de_anuencia", "anuencia_label", "Canal de Anuencia"),
    ("canal_de_contacto", "contacto_label", "Canal de Contacto"),
    ("sexo", "sexo_label", "Sexo"),
    ("estado_civil", "estado_civil_label", "Estado Cívil"),
    ("rango_de_edad", "rango_de_edad_label", "Rango de Edad"),
    ("lugar_vivienda", "lugvivi_label", "Lugar Vivienda"),
    ("lugar_trabajo", "lugtrab_label", "Lugar Trabajo"),
    ("razon_de_compra", "razoncompra_label", "Razón de Compra"),
    ("ingreso_familiar", "ingfam_label", "Ingreso Familiar"),
    ("nucleo_familiar", "nucleofam_label", "Núcleo Familiar"),
    ("opciones_visitadas", "opciones_visitadas_label", "Opciones Visitadas"),
    ("tipo_de_apartamento", "tipoapa_label", "Tipo Unidad"),
    ("participantes_en_credito", "partcredi_label", "Participantes en Crédito"),
]

FINAL_SCHEMA_RENAMES = {
    "id_trato": "id_Trato",
    "stage_label": "Etapa Trato",
    "date_insert": "Fecha_Creación_Trato",
    "fecha_reserva": "Fecha Reserva",
    "lost_time": "Fecha Desistimiento",
    "pipe_label": "name",
    "tipotrato": "Tipo",
    "proyecto": "Proyecto",
}

LABEL_COLUMNS = {label for _, label, _ in LABEL_RULE}
ORIGINAL_LABEL_SOURCES = {original for original, _, _ in LABEL_RULE}


def read_pt2_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=";",
        encoding="utf-8",
        on_bad_lines="skip",
        low_memory=False,
    )


def normalize_source_schema(df: pd.DataFrame) -> pd.DataFrame:
    renames = {}
    if "stage_label" not in df.columns and "name" in df.columns:
        renames["name"] = "stage_label"
    if "pipe_label" not in df.columns and "name.1" in df.columns:
        renames["name.1"] = "pipe_label"
    if "tipotrato" not in df.columns and "tipo" in df.columns:
        renames["tipo"] = "tipotrato"
    return df.rename(columns=renames)


def clean_stage_spaces(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)


def sample_values(series: pd.Series, n: int = 3) -> str:
    values = [str(value) for value in series.dropna().unique()[:n]]
    return " | ".join(values)


def pct_non_null(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return 0.0
    return round(df[col].notna().mean() * 100, 2)


def apply_label_rule(df: pd.DataFrame, file_name: str) -> tuple[pd.DataFrame, list[dict]]:
    df = df.copy()
    report_rows = []

    for original_col, label_col, final_col in LABEL_RULE:
        original_exists = original_col in df.columns
        label_exists = label_col in df.columns

        if label_exists and original_exists:
            final_values = df[label_col].where(df[label_col].notna(), df[original_col])
            action = "usar_label_con_fallback_original"
        elif label_exists:
            final_values = df[label_col]
            action = "usar_label"
        elif original_exists:
            final_values = df[original_col]
            action = "usar_original"
        else:
            final_values = pd.Series(pd.NA, index=df.index)
            action = "columna_no_encontrada"

        df[final_col] = final_values
        report_rows.append(
            {
                "archivo": file_name,
                "columna_original": original_col,
                "columna_label": label_col,
                "columna_final": final_col,
                "acción aplicada": action,
                "cobertura_original": pct_non_null(df, original_col),
                "cobertura_label": pct_non_null(df, label_col),
                "muestra_label": sample_values(df[label_col]) if label_exists else "",
            }
        )

    drop_cols = [col for col in LABEL_COLUMNS | ORIGINAL_LABEL_SOURCES if col in df.columns]
    df = df.drop(columns=drop_cols)
    return df, report_rows


def harmonize_project(project_name: str, path: Path) -> tuple[pd.DataFrame, list[dict]]:
    df = read_pt2_csv(path)
    df = normalize_source_schema(df)

    df, report_rows = apply_label_rule(df, project_name)
    df, removed_code_columns = keep_human_readable_labels(df)
    for column in removed_code_columns:
        report_rows.append(
            {
                "archivo": project_name,
                "columna_original": column,
                "columna_label": "",
                "columna_final": "",
                "acción aplicada": "homologar_label_con_fallback_original",
                "cobertura_original": "",
                "cobertura_label": "",
                "muestra_label": "",
            }
        )
    df = df.rename(columns=FINAL_SCHEMA_RENAMES)

    if "Etapa Trato" in df.columns:
        df["Etapa Trato"] = clean_stage_spaces(df["Etapa Trato"])

    if "Proyecto" not in df.columns:
        raise ValueError(f"{project_name}: falta columna Proyecto despues del rename")
    df["Proyecto"] = df["Proyecto"].map(PROYECTO_MAP).fillna(df["Proyecto"])
    df["fuente_pt2"] = project_name

    return df, report_rows


def validate_harmonized(
    frames: dict[str, pd.DataFrame],
    combined: pd.DataFrame,
    row_counts: dict[str, int],
) -> int:
    for project_name, df in frames.items():
        if "Etapa Trato" not in df.columns:
            raise ValueError(f"{project_name}: el archivo armonizado no tiene Etapa Trato")
        if "Proyecto" not in df.columns:
            raise ValueError(f"{project_name}: el archivo armonizado no tiene Proyecto")

        if df["Proyecto"].dropna().empty:
            raise ValueError(f"{project_name}: Proyecto no contiene valores")

        duplicated_label_cols = sorted(set(df.columns) & LABEL_COLUMNS)
        if duplicated_label_cols:
            raise ValueError(
                f"{project_name}: columnas label siguen presentes: {duplicated_label_cols}"
            )

        validate_analytical_label_schema(
            df,
            f"{project_name}_harmonized.csv",
        )

    obra_gris_count = int(combined["Etapa Trato"].eq("Obra gris").sum())
    if obra_gris_count <= 0:
        raise ValueError("Obra gris no fue conservado en PT2 armonizado")

    expected_total = sum(row_counts.values())
    if len(combined) != expected_total:
        raise ValueError(
            f"pt2_all_harmonized tiene {len(combined)} filas; esperado {expected_total}"
        )

    combined_path = OUTPUT_DIR / "pt2_all_harmonized.csv"
    if not combined_path.exists():
        raise FileNotFoundError(f"No existe {combined_path}")

    return obra_gris_count


def write_report(rows: list[dict]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows)
    with pd.ExcelWriter(REPORT_PATH, engine="openpyxl") as writer:
        report.to_excel(writer, sheet_name="label_replacement", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = {}
    report_rows = []
    row_counts = {}

    for input_path in discover_input_files():
        project_name = source_name(input_path)
        harmonized, project_report = harmonize_project(project_name, input_path)
        frames[project_name] = harmonized
        report_rows.extend(project_report)
        row_counts[project_name] = len(harmonized)
        output_path = OUTPUT_DIR / f"{safe_filename(project_name)}_harmonized.csv"
        harmonized.to_csv(output_path, index=False, encoding="utf-8-sig")

    combined = pd.concat(frames.values(), ignore_index=True, sort=False)
    combined_path = OUTPUT_DIR / "pt2_all_harmonized.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    write_report(report_rows)

    obra_gris_count = validate_harmonized(frames, combined, row_counts)

    print("ARMONIZACIÓN PT2 COMPLETADA")
    for project_name, count in row_counts.items():
        print(f"{project_name}: {count} filas")
    print(f"Total PT2 armonizado: {len(combined)} filas")
    print(f"Obra gris conservado para clasificación WON: {obra_gris_count} registros")


if __name__ == "__main__":
    main()
