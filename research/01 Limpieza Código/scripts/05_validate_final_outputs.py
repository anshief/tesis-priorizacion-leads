import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schema_contract import validate_analytical_label_schema

RAW_FINAL = PROJECT_ROOT / "data/raw/leads_final_combined.csv"
CLEANED_FINAL = PROJECT_ROOT / "data/processed/leads_cleaned_final.csv"
CLEANED_FINAL_WITH_LEAKAGE = (
    PROJECT_ROOT / "data/processed/leads_cleaned_with_leakage_final.csv"
)
LEAKAGE_REFERENCE_FINAL = PROJECT_ROOT / "data/processed/leads_leakage_reference_final.csv"
ENRICHED_FINAL = PROJECT_ROOT / "data/features/leads_enriched_final.csv"
ENRICHED_FINAL_WITH_LEAKAGE = (
    PROJECT_ROOT / "data/features/leads_enriched_with_leakage_final.csv"
)
ENRICHED_QUALIFIED_FINAL = PROJECT_ROOT / "data/features/leads_enriched_qualified_final.csv"
EDA_READY_SUMMARY = PROJECT_ROOT / "reports/final/eda_ready_summary.md"

REQUIRED_FINAL_OUTPUTS = [
    RAW_FINAL,
    CLEANED_FINAL,
    CLEANED_FINAL_WITH_LEAKAGE,
    LEAKAGE_REFERENCE_FINAL,
    ENRICHED_FINAL,
    ENRICHED_FINAL_WITH_LEAKAGE,
    ENRICHED_QUALIFIED_FINAL,
    EDA_READY_SUMMARY,
]

REQUIRED_READ_FILES = [
    RAW_FINAL,
    CLEANED_FINAL,
    CLEANED_FINAL_WITH_LEAKAGE,
    ENRICHED_FINAL,
    ENRICHED_FINAL_WITH_LEAKAGE,
    ENRICHED_QUALIFIED_FINAL,
    EDA_READY_SUMMARY,
]

REQUIRED_BASE_PROJECTS = {
    "proyecto01",
    "proyecto02",
    "proyecto03",
    "proyecto05",
    "proyecto04",
}

LEAKAGE_COLS = {
    "status",
    "Etapa Trato",
    "etapa_trato",
    "stage_id",
    "lost_reason",
    "nota_de_perdido",
    "close_time",
    "Fecha Reserva",
    "fecha_reserva",
    "Fecha Desistimiento",
    "fecha_desistimiento",
    "Tipo",
    "tipo",
    "pipeline_id",
    "name",
    "Ingreso CRM",
    "ingreso_crm",
}

REQUIRED_ENRICHED_COLUMNS = {
    "campania_plataforma",
    "campania_tipo",
    "campania_audiencia",
    "rango_edad_modelo",
    "zona_proyecto",
    "precio_m2_promedio",
    "fuente_dataset",
}


def assert_exists_and_nonempty(path: Path) -> None:
    assert path.exists(), f"No existe: {path.relative_to(PROJECT_ROOT)}"
    assert path.stat().st_size > 0, f"Archivo vacío: {path.relative_to(PROJECT_ROOT)}"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def validate_target_binary(df: pd.DataFrame, label: str) -> None:
    assert "target_won" in df.columns, f"{label} no contiene target_won"
    values = set(df["target_won"].dropna().unique())
    assert values.issubset({0, 1}), (
        f"{label} target_won contiene valores fuera de 0/1: {sorted(values)}"
    )


def validate_no_duplicates(df: pd.DataFrame, label: str) -> None:
    assert "id_trato" in df.columns, f"{label} no contiene id_trato"
    duplicate_count = int(df["id_trato"].duplicated().sum())
    assert duplicate_count == 0, f"{label} tiene {duplicate_count} duplicados por id_trato"


def validate_markdown() -> None:
    markdown = EDA_READY_SUMMARY.read_text(encoding="utf-8")
    assert markdown.strip(), "eda_ready_summary.md está vacío"


def validate_obra_gris_sin_garantias(df_wl: pd.DataFrame) -> dict[str, str]:
    status = {"Obra gris": "SKIP", "Sin Garantías": "SKIP"}
    if "etapa_trato" not in df_wl.columns:
        return status

    etapas = df_wl["etapa_trato"].astype(str).str.lower().str.strip()
    for etapa in ["obra gris", "sin garantías"]:
        subset = df_wl[etapas.eq(etapa)]
        label = "Obra gris" if etapa == "obra gris" else "Sin Garantías"
        if len(subset) > 0:
            assert (subset["target_won"] == 1).all(), f"{etapa} no quedó como WON"
            status[label] = "PASS"
    return status


def main() -> None:
    for path in REQUIRED_FINAL_OUTPUTS:
        assert_exists_and_nonempty(path)

    df_final = read_csv(CLEANED_FINAL)
    df_wl = read_csv(CLEANED_FINAL_WITH_LEAKAGE)
    df_enriched = read_csv(ENRICHED_FINAL)
    df_enriched_wl = read_csv(ENRICHED_FINAL_WITH_LEAKAGE)
    df_qualified = read_csv(ENRICHED_QUALIFIED_FINAL)

    for dataset, path in [
        (df_final, CLEANED_FINAL),
        (df_wl, CLEANED_FINAL_WITH_LEAKAGE),
        (df_enriched, ENRICHED_FINAL),
        (df_enriched_wl, ENRICHED_FINAL_WITH_LEAKAGE),
        (df_qualified, ENRICHED_QUALIFIED_FINAL),
    ]:
        validate_analytical_label_schema(dataset, path.name)

    validate_target_binary(df_final, "leads_cleaned_final.csv")
    validate_target_binary(df_wl, "leads_cleaned_with_leakage_final.csv")
    validate_target_binary(df_enriched, "leads_enriched_final.csv")
    validate_target_binary(
        df_enriched_wl,
        "leads_enriched_with_leakage_final.csv",
    )
    validate_target_binary(df_qualified, "leads_enriched_qualified_final.csv")

    assert len(df_final) == len(df_enriched), (
        "leads_enriched_final.csv no conserva las filas del dataset limpio final"
    )
    assert len(df_wl) == len(df_enriched_wl), (
        "leads_enriched_with_leakage_final.csv no conserva las filas del dataset con leakage"
    )
    assert list(df_qualified.columns) == list(df_enriched.columns), (
        "leads_enriched_qualified_final.csv debe tener exactamente las mismas columnas que leads_enriched_final.csv"
    )
    assert len(df_qualified) <= len(df_enriched), (
        "leads_enriched_qualified_final.csv no puede tener más filas que leads_enriched_final.csv"
    )
    assert len(df_wl) == len(df_final), (
        "El dataset con leakage debe tener las mismas filas que el dataset modelable"
    )

    actual_projects = set(df_final["proyecto"].dropna().astype(str).unique())
    missing_projects = REQUIRED_BASE_PROJECTS - actual_projects
    assert not missing_projects, (
        f"Faltan proyectos base: {sorted(missing_projects)}. "
        f"Actuales={sorted(actual_projects)}"
    )

    cleaned_leakage_cols = sorted(LEAKAGE_COLS & set(df_final.columns))
    assert not cleaned_leakage_cols, (
        "leads_cleaned_final.csv contiene columnas de leakage: "
        f"{cleaned_leakage_cols}"
    )
    enriched_leakage_cols = sorted(LEAKAGE_COLS & set(df_enriched.columns))
    assert not enriched_leakage_cols, (
        "leads_enriched_final.csv contiene columnas de leakage: "
        f"{enriched_leakage_cols}"
    )

    with_leakage_cols = LEAKAGE_COLS & set(df_wl.columns)
    enriched_with_leakage_cols = LEAKAGE_COLS & set(df_enriched_wl.columns)
    required_with_leakage = {"status", "etapa_trato"}
    missing_required_wl = required_with_leakage - set(df_wl.columns)
    assert not missing_required_wl, (
        "leads_cleaned_with_leakage_final.csv no contiene columnas requeridas de "
        f"leakage: {sorted(missing_required_wl)}"
    )
    assert with_leakage_cols, (
        "leads_cleaned_with_leakage_final.csv no contiene columnas de leakage"
    )
    assert enriched_with_leakage_cols, (
        "leads_enriched_with_leakage_final.csv no contiene columnas de leakage"
    )

    etapa_status = validate_obra_gris_sin_garantias(df_wl)

    missing_enriched_cols = REQUIRED_ENRICHED_COLUMNS - set(df_enriched.columns)
    assert not missing_enriched_cols, (
        f"Faltan columnas en leads_enriched_final.csv: {sorted(missing_enriched_cols)}"
    )

    validate_no_duplicates(df_final, "leads_cleaned_final.csv")
    validate_no_duplicates(df_wl, "leads_cleaned_with_leakage_final.csv")
    validate_no_duplicates(df_enriched, "leads_enriched_final.csv")
    validate_no_duplicates(
        df_enriched_wl,
        "leads_enriched_with_leakage_final.csv",
    )
    validate_no_duplicates(df_qualified, "leads_enriched_qualified_final.csv")

    qualified_stage_values = (
        df_enriched_wl.loc[
            df_enriched_wl["id_trato"].isin(df_qualified["id_trato"].astype(str)),
            "etapa_trato",
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    assert not qualified_stage_values.isin({"contactados", "ingresados"}).any(), (
        "leads_enriched_qualified_final.csv contiene etapas de contactados o ingresados"
    )

    validate_markdown()

    target_counts = df_final["target_won"].value_counts().to_dict()
    total = len(df_final)
    won_count = int(target_counts.get(1, 0))
    lost_count = int(target_counts.get(0, 0))
    won_pct = won_count / total * 100
    lost_pct = lost_count / total * 100
    project_counts = df_final["proyecto"].value_counts().sort_index()

    print("VALIDACIÓN FINAL — TODOS LOS CHECKS PASARON")
    print()
    print(f"leads_cleaned_final.csv: {len(df_final):,} filas")
    print(f"leads_enriched_final.csv: {len(df_enriched):,} filas")
    print(f"leads_enriched_qualified_final.csv: {len(df_qualified):,} filas")
    print(f"leads_enriched_with_leakage_final.csv: {len(df_enriched_wl):,} filas")
    print(f"target_won=1: {won_count:,} ({won_pct:.2f}%)")
    print(f"target_won=0: {lost_count:,} ({lost_pct:.2f}%)")
    print()
    print("Proyectos:")
    for project, count in project_counts.items():
        print(f"- {project}: {int(count):,}")
    print()
    print(f"Obra gris como WON: {etapa_status['Obra gris']}")
    print(f"Sin Garantías como WON: {etapa_status['Sin Garantías']}")
    print("Dataset con leakage creado: PASS")
    print("Dataset modelado sin leakage: PASS")


if __name__ == "__main__":
    main()
