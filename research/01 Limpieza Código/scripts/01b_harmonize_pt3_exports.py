"""Armoniza los reportes nuevos de data_pt3 y los fusiona con PT2.

data_pt3 contiene reexportaciones más recientes y confiables de proyecto03,
proyecto04 y proyecto05 (mismas columnas "bonitas" que los Excel base proyecto01/proyecto02, en vez
del formato código+label de data_pt2) más un proyecto nuevo, proyecto06.

Regla de fusión con PT2 (decidida junto con la usuaria, ver
COMO_AGREGAR_PROYECTOS.md): para cada id_Trato que exista en PT3, se conserva
la versión de PT3 (es más reciente/confiable; PT2 tiene un bug conocido en
Fecha_Creación_Trato, constante en todas las filas). Los id_Trato que sólo
existen en PT2 se conservan tal cual. Esto implica que si para un proyecto los
conjuntos de id_Trato de PT2 y PT3 son idénticos, el resultado es
efectivamente "usar sólo PT3"; si difieren, el resultado es una unión (join).
"""

from pathlib import Path

import pandas as pd

from schema_contract import (
    ANALYTICAL_LABEL_REPLACEMENTS,
    validate_analytical_label_schema,
)


INPUT_DIR = Path("data/raw/data_pt3")
OUTPUT_DIR = Path("data/raw/data_pt3/harmonized")
PT2_ALL_HARMONIZED_PATH = Path("data/raw/data_pt2/harmonized/pt2_all_harmonized.csv")
MERGED_OUTPUT_PATH = Path("data/raw/data_pt2/harmonized/pt2_pt3_merged.csv")
MERGE_SUMMARY_PATH = Path("reports/final/pt3_merge_summary.md")

# data_pt3 ya trae nombres "bonitos" (idénticos a los Excel base proyecto01/proyecto02) para
# variables que en PT2 sólo existen como columnas *_label. Se renombran a la
# forma canónica *_label para que sobrevivan a la limpieza (03) igual que en
# PT2, en vez de ser eliminadas por alta ausencia de datos como ocurre con las
# columnas homónimas de proyecto01/proyecto02 en HIGH_MISSING_DROP_COLS.
RENAME_TO_CANONICAL_LABEL = {
    "Unidad #": "apartamento_no_label",
    "Proyecto Referidor": "proyref_label",
    "Unidad Referidora": "torre_referidora_label",
    "Cuota deseada": "cuota_deseada_label",
    "Tipo Visita": "tipovisita_label",
    "Razón Inversión": "razon_inversion_label",
    "Modelo": "modelo_label",
    "Tipo Modelo": "tipo_modelo_label",
    "Bodega Asignada": "bodega_asignada_label",
    "Cantidad Parqueos": "cantidad_parqueos_label",
    "CSI Conocimiento Asesor": "csi_conocimiento_asesor_label",
    "CSI Tiempo Seguimiento": "csi_tiempo_seguimiento_label",
    "CSI Calificación Amenidades": "csi_calificacion_amenidades_label",
    "CSI Calificación Apartamento": "csi_calificacion_apartamento_label",
    "CSI Calificación Proyecto": "csi_calificacion_proyecto_label",
    "NPS Recomendación Proyecto": "nps_recomendacion_proyecto_label",
    "CSI Comentarios Adicionales": "csi_comentarios_adicionales_label",
    "Ubicación Parqueos": "ubicacion_parqueos_label",
    "Parqueo 1": "parqueo_1_label",
    "Parqueo 2": "parqueo_2_label",
    "Parqueo 3": "parqueo_3_label",
    "Ingreso CRM": "ingreso_crm_label",
    "Ubicación Bodega": "ubicacion_bodega_label",
}

# "Sector Referidor" en PT3 no tiene un equivalente inequívoco en el esquema
# de PT2 (podría corresponder a apto_referidor_label, pero el nombre no
# coincide con confianza), así que se deja sin mapear para no mezclar
# variables distintas por error. apto_referidor_label queda vacío para las
# filas que vienen de PT3.
MISSING_FROM_PT3 = sorted(
    set(ANALYTICAL_LABEL_REPLACEMENTS.values()) - set(RENAME_TO_CANONICAL_LABEL.values())
)


def discover_pt3_files() -> list[Path]:
    files = sorted(INPUT_DIR.glob("Reporte UVG *.xlsx"))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos Reporte UVG *.xlsx en {INPUT_DIR}")
    return files


def project_name_from_data(df: pd.DataFrame, path: Path) -> str:
    values = df["Proyecto"].dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{path.name}: se esperaba un único valor de Proyecto, se encontró {values}")
    return str(values[0])


def harmonize_pt3_file(path: Path) -> tuple[str, pd.DataFrame]:
    df = pd.read_excel(path)
    project_name = project_name_from_data(df, path)

    df = df.rename(columns=RENAME_TO_CANONICAL_LABEL)
    df = df.drop(columns=["Proyecto.1"], errors="ignore")

    for missing_label in MISSING_FROM_PT3:
        if missing_label not in df.columns:
            df[missing_label] = pd.NA

    # fuente_pt2 identifica el proyecto de origen de la exportación armonizada
    # (nombre histórico de la columna; ahora también se usa para PT3).
    df["fuente_pt2"] = project_name

    return project_name, df


def merge_pt2_pt3(df_pt2: pd.DataFrame, df_pt3: pd.DataFrame) -> pd.DataFrame:
    if "id_Trato" not in df_pt2.columns or "id_Trato" not in df_pt3.columns:
        raise ValueError("id_Trato debe existir en ambos datasets para fusionar")

    all_cols = []
    for frame in (df_pt2, df_pt3):
        for column in frame.columns:
            if column not in all_cols:
                all_cols.append(column)

    df_pt2_aligned = df_pt2.reindex(columns=all_cols)
    df_pt3_aligned = df_pt3.reindex(columns=all_cols)

    ids_in_pt3 = set(df_pt3_aligned["id_Trato"])
    pt2_unique_rows = df_pt2_aligned.loc[~df_pt2_aligned["id_Trato"].isin(ids_in_pt3)]

    merged = pd.concat([pt2_unique_rows, df_pt3_aligned], ignore_index=True, sort=False)
    return merged


def write_merge_summary(rows: list[dict]) -> None:
    MERGE_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Resumen de fusión PT2 + PT3",
        "",
        "Regla aplicada: para cada `id_Trato` presente en PT3 se conserva la fila",
        "de PT3 (fuente más reciente/confiable). Las filas cuyo `id_Trato` sólo",
        "existe en PT2 se conservan sin cambios. Si los conjuntos de id_Trato de",
        "PT2 y PT3 son idénticos para un proyecto, el resultado equivale a usar",
        "sólo PT3.",
        "",
        "| Proyecto | Filas PT2 | Filas PT3 | ids sólo en PT2 | ids sólo en PT3 | Filas fusionadas |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['proyecto']} | {row['filas_pt2']} | {row['filas_pt3']} | "
            f"{row['solo_pt2']} | {row['solo_pt3']} | {row['filas_fusionadas']} |"
        )
    MERGE_SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not PT2_ALL_HARMONIZED_PATH.exists():
        raise FileNotFoundError(
            f"No existe {PT2_ALL_HARMONIZED_PATH}. Ejecuta 01_harmonize_new_exports.py primero."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pt3_frames = {}
    for path in discover_pt3_files():
        project_name, harmonized = harmonize_pt3_file(path)
        pt3_frames[project_name] = harmonized
        validate_analytical_label_schema(harmonized, f"{project_name}_pt3_harmonized.csv")
        safe_name = project_name.lower().replace(" ", "_")
        harmonized.to_csv(OUTPUT_DIR / f"{safe_name}_pt3_harmonized.csv", index=False, encoding="utf-8-sig")

    df_pt3_all = pd.concat(pt3_frames.values(), ignore_index=True, sort=False)
    df_pt3_all.to_csv(OUTPUT_DIR / "pt3_all_harmonized.csv", index=False, encoding="utf-8-sig")

    df_pt2_all = pd.read_csv(PT2_ALL_HARMONIZED_PATH, encoding="utf-8-sig", low_memory=False)

    merged = merge_pt2_pt3(df_pt2_all, df_pt3_all)

    duplicated_ids = int(merged["id_Trato"].duplicated().sum())
    if duplicated_ids:
        raise ValueError(f"pt2_pt3_merged quedó con {duplicated_ids} id_Trato duplicados")

    validate_analytical_label_schema(merged, MERGED_OUTPUT_PATH.name)

    MERGED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    summary_rows = []
    all_projects = sorted(set(df_pt2_all["Proyecto"].dropna()) | set(df_pt3_all["Proyecto"].dropna()))
    for project in all_projects:
        ids_pt2 = set(df_pt2_all.loc[df_pt2_all["Proyecto"] == project, "id_Trato"])
        ids_pt3 = set(df_pt3_all.loc[df_pt3_all["Proyecto"] == project, "id_Trato"])
        ids_fusionadas = (ids_pt2 - ids_pt3) | ids_pt3
        summary_rows.append(
            {
                "proyecto": project,
                "filas_pt2": len(ids_pt2),
                "filas_pt3": len(ids_pt3),
                "solo_pt2": len(ids_pt2 - ids_pt3),
                "solo_pt3": len(ids_pt3 - ids_pt2),
                "filas_fusionadas": len(ids_fusionadas),
            }
        )
    write_merge_summary(summary_rows)

    print("ARMONIZACIÓN Y FUSIÓN PT3 COMPLETADA")
    for row in summary_rows:
        print(
            f"{row['proyecto']}: PT2={row['filas_pt2']} | PT3={row['filas_pt3']} | "
            f"solo_PT2={row['solo_pt2']} | solo_PT3={row['solo_pt3']} | "
            f"fusionadas={row['filas_fusionadas']}"
        )
    print(f"Total PT2 original: {len(df_pt2_all)} filas")
    print(f"Total PT3 armonizado: {len(df_pt3_all)} filas")
    print(f"Total fusionado (PT2 + PT3): {len(merged)} filas")


if __name__ == "__main__":
    main()
