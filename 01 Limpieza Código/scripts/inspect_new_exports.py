import re
import unicodedata
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

proyecto03_CSV_PATH = PROJECT_ROOT / "data/raw/data_pt2/Leads_proyecto03_UVG.csv"
proyecto03_XLSX_PATH = PROJECT_ROOT / "data/raw/data_pt2/Leads_proyecto03_UVG.xlsx"
proyecto05_CSV_PATH = PROJECT_ROOT / "data/raw/data_pt2/Leads_proyecto05_UVG.csv"
proyecto04_CSV_PATH = PROJECT_ROOT / "data/raw/data_pt2/Leads_proyecto04_UVG.csv"
FINAL_RAW_PATH = PROJECT_ROOT / "data/raw/leads_final_combined.csv"
REPORT_PATH = PROJECT_ROOT / "reports/final/schema_new_exports_report.xlsx"

NON_WON_STAGES = {
    "contactados",
    "ingresados",
    "interesados",
    "visita modelo",
    "visita programada",
    "corto plazos",
    "cortos plazos",
}

WON_STAGES = {
    "reserva",
    "suspensión",
    "armado expediente",
    "análisis crédito",
    "análisis de crédito",
    "aprobación",
    "resolución",
    "expediente técnico",
    "formalización",
    "promesa",
    "escrituración",
    "liquidación y desembolsos",
    "conformidad final",
    "entrega fallida",
    "gestión minutas",
    "gestión rgp",
    "liberación",
    "liquidación",
    "minutas aprobadas",
    "orden escrituración",
    "programación",
    "habitado",
    "operaciones",
    "reparaciones",
    "resguardo",
    "sin garantías",
    "obra gris",
}

EXCLUDE_STAGES = {
    "1er testimonio",
    "acabados",
    "avisos notariales",
}

LABEL_PAIRS = [
    ("canal_de_anuencia", "anuencia_label"),
    ("canal_de_contacto", "contacto_label"),
    ("sexo", "sexo_label"),
    ("estado_civil", "estado_civil_label"),
    ("rango_de_edad", "rango_de_edad_label"),
    ("lugar_vivienda", "lugvivi_label"),
    ("lugar_trabajo", "lugtrab_label"),
    ("razon_de_compra", "razoncompra_label"),
    ("ingreso_familiar", "ingfam_label"),
    ("nucleo_familiar", "nucleofam_label"),
    ("opciones_visitadas", "opciones_visitadas_label"),
    ("tipo_de_apartamento", "tipoapa_label"),
    ("participantes_en_credito", "partcredi_label"),
]

COLUMN_EQUIVALENCES = {
    "Etapa Trato": "stage_label",
    "name": "pipe_label en base final; stage_label/pipe_label en exports nuevos segun fuente",
    "Proyecto.1": "proyecto",
    "Tipo": "tipotrato",
    "Canal de Anuencia": "canal_de_anuencia",
    "Canal de Contacto": "canal_de_contacto",
    "Sexo": "sexo",
    "Sector Referidor": "proyref_label",
    "Proyecto": "proyecto",
    "Tipo Unidad": "tipo_de_apartamento",
    "Participantes en Crédito": "participantes_en_credito",
    "Tipo Visita": "tipovisita_label",
    "Lugar Vivienda": "lugar_vivienda",
    "Lugar Trabajo": "lugar_trabajo",
    "Razón de Compra": "razon_de_compra",
    "Ingreso Familiar": "ingreso_familiar",
    "Núcleo Familiar": "nucleo_familiar",
    "Cuota deseada": "cuota_deseada_label",
    "Opciones Visitadas": "opciones_visitadas",
    "Estado Cívil": "estado_civil",
    "Rango de Edad": "rango_de_edad",
    "Fecha_Creación_Trato": "fecha_creacion_trato",
    "Fecha Reserva": "fecha_reserva",
    "Fecha Desistimiento": "lost_time",
}


def read_pt2_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=";",
        encoding="utf-8",
        on_bad_lines="skip",
        low_memory=False,
    )


def normalize_stage(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def normalize_schema_name(value: str) -> str:
    text = str(value).replace("\ufeff", "").strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def canonical_column_name(value: str) -> str:
    aliases = {
        "etapa_trato": "stage_label",
        "fecha_reserva": "fecha_reserva",
        "fecha_creacion_trato": "fecha_creacion_trato",
        "fecha_desistimiento": "lost_time",
        "tipo_unidad": "tipo_de_apartamento",
        "tipo": "tipotrato",
        "estado_civil": "estado_civil",
        "razon_de_compra": "razon_de_compra",
        "nucleo_familiar": "nucleo_familiar",
        "rango_de_edad": "rango_de_edad",
        "canal_de_anuencia": "canal_de_anuencia",
        "canal_de_contacto": "canal_de_contacto",
        "participantes_en_credito": "participantes_en_credito",
        "id_trato": "id_trato",
    }
    normalized = normalize_schema_name(value)
    return aliases.get(normalized, normalized)


def compact_list(values, max_items: int = 30) -> str:
    cleaned = [str(value) for value in values if str(value) != ""]
    if len(cleaned) <= max_items:
        return " | ".join(cleaned)
    return " | ".join(cleaned[:max_items]) + f" | ... (+{len(cleaned) - max_items} mas)"


def compact_counts(series: pd.Series, top_n: int | None = 10) -> str:
    counts = series.value_counts(dropna=False)
    if top_n is not None:
        counts = counts.head(top_n)
    parts = []
    for value, count in counts.items():
        label = "<NA>" if pd.isna(value) else str(value)
        parts.append(f"{label}: {int(count)}")
    return " | ".join(parts)


def date_range(series: pd.Series) -> tuple[str, str]:
    dates = pd.to_datetime(series, errors="coerce")
    if dates.notna().sum() == 0:
        return "", ""
    return str(dates.min().date()), str(dates.max().date())


def unique_dates(series: pd.Series) -> list[str]:
    dates = pd.to_datetime(series, errors="coerce")
    return sorted({str(value.date()) for value in dates.dropna()})


def prepare_proyecto04(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "name": "stage_label",
            "name.1": "pipe_label",
            "tipo": "tipotrato",
        }
    )


def detect_mojibake(series: pd.Series) -> int:
    return int(series.astype(str).str.contains(r"Ã|Â|�", regex=True, na=False).sum())


def proyecto03_recommendation(proyecto03_xlsx: pd.DataFrame) -> str:
    if "stage_label" in proyecto03_xlsx.columns and detect_mojibake(proyecto03_xlsx["stage_label"]) > 0:
        return "USE_CSV"
    return "USE_BOTH_DEDUPLICATE"


def compare_proyecto03(proyecto03_csv: pd.DataFrame, proyecto03_xlsx: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    csv_ids = set(proyecto03_csv["id_trato"].dropna().astype(str))
    xlsx_ids = set(proyecto03_xlsx["id_trato"].dropna().astype(str))
    only_csv = sorted(csv_ids - xlsx_ids)
    only_xlsx = sorted(xlsx_ids - csv_ids)
    both = sorted(csv_ids & xlsx_ids)
    recommendation = proyecto03_recommendation(proyecto03_xlsx)

    common_columns = [col for col in proyecto03_csv.columns if col in proyecto03_xlsx.columns]
    value_diff_columns = []
    stage_examples = []

    csv_cmp = proyecto03_csv.set_index(proyecto03_csv["id_trato"].astype(str)).sort_index()
    xlsx_cmp = proyecto03_xlsx.set_index(proyecto03_xlsx["id_trato"].astype(str)).sort_index()
    common_index = csv_cmp.index.intersection(xlsx_cmp.index)
    csv_cmp = csv_cmp.loc[common_index]
    xlsx_cmp = xlsx_cmp.loc[common_index]

    for col in common_columns:
        left = csv_cmp[col].where(csv_cmp[col].notna(), "<NA>").astype(str)
        right = xlsx_cmp[col].where(xlsx_cmp[col].notna(), "<NA>").astype(str)
        diff_mask = left.ne(right)
        if diff_mask.any():
            value_diff_columns.append(col)
            if col == "stage_label":
                for idx in left[diff_mask].index[:5]:
                    stage_examples.append(
                        f"{idx}: CSV={left.loc[idx]} / XLSX={right.loc[idx]}"
                    )

    mojibake_count = (
        detect_mojibake(proyecto03_xlsx["stage_label"])
        if "stage_label" in proyecto03_xlsx.columns
        else 0
    )

    rows = [
        {"metrica": "filas_csv", "valor": len(proyecto03_csv)},
        {"metrica": "filas_xlsx", "valor": len(proyecto03_xlsx)},
        {"metrica": "columnas_csv", "valor": len(proyecto03_csv.columns)},
        {"metrica": "columnas_xlsx", "valor": len(proyecto03_xlsx.columns)},
        {"metrica": "id_trato_unicos_csv", "valor": len(csv_ids)},
        {"metrica": "id_trato_unicos_xlsx", "valor": len(xlsx_ids)},
        {"metrica": "ids_solo_csv_count", "valor": len(only_csv)},
        {"metrica": "ids_solo_csv_muestra", "valor": compact_list(only_csv, 20)},
        {"metrica": "ids_solo_xlsx_count", "valor": len(only_xlsx)},
        {"metrica": "ids_solo_xlsx_muestra", "valor": compact_list(only_xlsx, 20)},
        {"metrica": "ids_en_ambos_count", "valor": len(both)},
        {
            "metrica": "columnas_con_valores_distintos",
            "valor": compact_list(value_diff_columns, 80),
        },
        {
            "metrica": "stage_label_diferencias_muestra",
            "valor": " | ".join(stage_examples),
        },
        {"metrica": "xlsx_stage_label_encoding_roto_count", "valor": mojibake_count},
        {
            "metrica": "nota_metodologica",
            "valor": (
                "Si el XLSX tiene encoding roto en stage_label, "
                "se recomienda usar CSV."
            ),
        },
        {"metrica": "recomendacion", "valor": recommendation},
    ]
    return pd.DataFrame(rows), recommendation


def build_file_overview(named_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in named_frames.items():
        date_min, date_max = date_range(df["date_insert"]) if "date_insert" in df else ("", "")
        close_min, close_max = date_range(df["close_time"]) if "close_time" in df else ("", "")
        reserva_min, reserva_max = (
            date_range(df["fecha_reserva"]) if "fecha_reserva" in df else ("", "")
        )
        stage_col = "stage_label" if "stage_label" in df.columns else "name"
        id_duplicates = int(df["id_trato"].duplicated().sum()) if "id_trato" in df else 0

        rows.append(
            {
                "archivo": name,
                "filas": len(df),
                "columnas": len(df.columns),
                "date_insert_min": date_min,
                "date_insert_max": date_max,
                "close_time_min": close_min,
                "close_time_max": close_max,
                "fecha_reserva_min": reserva_min,
                "fecha_reserva_max": reserva_max,
                "fecha_reserva_pct_non_null": round(df["fecha_reserva"].notna().mean() * 100, 2)
                if "fecha_reserva" in df
                else 0,
                "status_distribucion": compact_counts(df["status"], top_n=None)
                if "status" in df
                else "",
                "stage_label_top10": compact_counts(df[stage_col], top_n=10)
                if stage_col in df
                else "",
                "proyecto_valores_unicos": compact_list(sorted(df["proyecto"].dropna().astype(str).unique()))
                if "proyecto" in df
                else "",
                "id_trato_duplicados": id_duplicates,
            }
        )
    return pd.DataFrame(rows)


def classify_stage(stage_normalized: str) -> str:
    if stage_normalized in {normalize_stage(value) for value in NON_WON_STAGES}:
        return "NON_WON"
    if stage_normalized in {normalize_stage(value) for value in WON_STAGES}:
        return "WON"
    return "UNKNOWN"


def recommend_stage(classification: str) -> str:
    if classification == "NON_WON":
        return "target_won_0"
    if classification == "WON":
        return "target_won_1"
    return "revisar_manual"


def stage_action(stage_normalized: str, classification: str) -> str:
    if stage_normalized in EXCLUDE_STAGES:
        return "excluir"
    return recommend_stage(classification)


def build_stage_classification(named_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for file_name, df in named_frames.items():
        temp = df[["stage_label", "status", "proyecto"]].copy()
        temp["stage_label_original"] = temp["stage_label"].astype(str).str.strip()
        temp["stage_label_normalizado"] = temp["stage_label"].apply(normalize_stage)
        temp = temp[temp["stage_label_normalizado"].ne("")]

        grouped = temp.groupby(
            ["stage_label_original", "stage_label_normalizado", "proyecto"],
            dropna=False,
        )
        for (stage_original, stage_normalized, project), group in grouped:
            classification = classify_stage(stage_normalized)
            rows.append(
                {
                    "stage_label_original": stage_original,
                    "stage_label_normalizado": stage_normalized,
                    "archivo": file_name,
                    "proyecto": "<NA>" if pd.isna(project) else str(project),
                    "conteo": len(group),
                    "status_distribucion": compact_counts(group["status"], top_n=None),
                    "clasificacion": classification,
                    "recomendacion": stage_action(stage_normalized, classification),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["clasificacion", "stage_label_normalizado", "archivo", "proyecto"]
    )


def build_label_pairs(combined_pt2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(combined_pt2)
    for original, label in LABEL_PAIRS:
        original_exists = original in combined_pt2.columns
        label_exists = label in combined_pt2.columns
        original_coverage = (
            round(combined_pt2[original].notna().mean() * 100, 2)
            if original_exists and total
            else 0
        )
        label_coverage = (
            round(combined_pt2[label].notna().mean() * 100, 2)
            if label_exists and total
            else 0
        )
        rows.append(
            {
                "columna_original": original,
                "columna_label": label,
                "cobertura_original": original_coverage,
                "cobertura_label": label_coverage,
                "muestra_valores_label": compact_list(
                    combined_pt2[label].dropna().astype(str).unique()[:3], 3
                )
                if label_exists
                else "",
                "accion_recomendada": "usar_label"
                if label_exists and label_coverage > 0
                else "usar_original",
            }
        )
    return pd.DataFrame(rows)


def proposed_equivalence(canonical: str, originals: list[str]) -> str:
    for original in originals:
        if original in COLUMN_EQUIVALENCES:
            return COLUMN_EQUIVALENCES[original]
    for original, target in COLUMN_EQUIVALENCES.items():
        if canonical_column_name(original) == canonical:
            return target
    return ""


def build_schema_diff(final_raw: pd.DataFrame, named_pt2: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pt2_columns = []
    for df in named_pt2.values():
        pt2_columns.extend(df.columns)

    final_map = {}
    for col in final_raw.columns:
        final_map.setdefault(canonical_column_name(col), []).append(str(col))

    pt2_map = {}
    for col in pt2_columns:
        pt2_map.setdefault(canonical_column_name(col), []).append(str(col))

    rows = []
    for col in sorted(set(final_map) - set(pt2_map)):
        originals = sorted(set(final_map[col]))
        rows.append(
            {
                "tipo_diff": "presente_en_final_no_en_exports_nuevos",
                "columna_canonica": col,
                "nombres_originales": compact_list(originals),
                "equivalencias_propuestas": proposed_equivalence(col, originals),
                "advertencia": "",
            }
        )

    for col in sorted(set(pt2_map) - set(final_map)):
        originals = sorted(set(pt2_map[col]))
        rows.append(
            {
                "tipo_diff": "presente_en_exports_nuevos_no_en_final",
                "columna_canonica": col,
                "nombres_originales": compact_list(originals),
                "equivalencias_propuestas": proposed_equivalence(col, originals),
                "advertencia": "",
            }
        )

    date_values = sorted(
        {
            value
            for df in named_pt2.values()
            if "date_insert" in df.columns
            for value in unique_dates(df["date_insert"])
        }
    )
    warning = ""
    if len(date_values) == 1:
        warning = (
            f"Todos los registros PT2 tienen date_insert={date_values[0]}; "
            "date_insert es fecha de exportacion y solo debe usarse como proxy temporal."
        )
    rows.append(
        {
            "tipo_diff": "date_insert_proxy_temporal",
            "columna_canonica": "date_insert",
            "nombres_originales": "date_insert",
            "equivalencias_propuestas": "",
            "advertencia": warning,
        }
    )
    return pd.DataFrame(rows)


def autofit_workbook(path: Path) -> None:
    from openpyxl import load_workbook

    workbook = load_workbook(path)
    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        for column_cells in worksheet.columns:
            letter = column_cells[0].column_letter
            max_width = 12
            for cell in column_cells:
                value = "" if cell.value is None else str(cell.value)
                max_width = max(max_width, min(len(value) + 2, 80))
            worksheet.column_dimensions[letter].width = max_width
    workbook.save(path)


def main() -> None:
    proyecto03_csv = read_pt2_csv(proyecto03_CSV_PATH)
    proyecto03_xlsx = pd.read_excel(proyecto03_XLSX_PATH)
    proyecto05_csv = read_pt2_csv(proyecto05_CSV_PATH)
    proyecto04_csv = prepare_proyecto04(read_pt2_csv(proyecto04_CSV_PATH))
    final_raw = pd.read_csv(FINAL_RAW_PATH, encoding="utf-8-sig", low_memory=False)

    named_csvs = {
        "proyecto03 CSV": proyecto03_csv,
        "proyecto05 CSV": proyecto05_csv,
        "proyecto04 CSV": proyecto04_csv,
    }
    named_all = {
        "proyecto03 CSV": proyecto03_csv,
        "proyecto03 XLSX": proyecto03_xlsx,
        "proyecto05 CSV": proyecto05_csv,
        "proyecto04 CSV": proyecto04_csv,
    }

    proyecto03_report, recommendation = compare_proyecto03(proyecto03_csv, proyecto03_xlsx)
    file_overview = build_file_overview(named_all)
    stage_classification = build_stage_classification(named_csvs)
    label_pairs = build_label_pairs(pd.concat(named_csvs.values(), ignore_index=True, sort=False))
    schema_diff = build_schema_diff(final_raw, named_csvs)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(REPORT_PATH, engine="openpyxl") as writer:
        proyecto03_report.to_excel(writer, sheet_name="proyecto03_csv_vs_xlsx", index=False)
        file_overview.to_excel(writer, sheet_name="file_overview", index=False)
        stage_classification.to_excel(
            writer,
            sheet_name="stages_pt2_classification",
            index=False,
        )
        label_pairs.to_excel(writer, sheet_name="label_pairs", index=False)
        schema_diff.to_excel(writer, sheet_name="schema_diff", index=False)

    autofit_workbook(REPORT_PATH)

    unknown_stages = sorted(
        stage_classification.loc[
            stage_classification["clasificacion"].eq("UNKNOWN"),
            "stage_label_original",
        ].unique()
    )

    print("DIAGNÓSTICO PT2 COMPLETADO")
    print(f"proyecto03 CSV: {proyecto03_csv.shape[0]} filas × {proyecto03_csv.shape[1]} cols")
    print(f"proyecto03 XLSX: {proyecto03_xlsx.shape[0]} filas × {proyecto03_xlsx.shape[1]} cols")
    print(f"proyecto05 CSV: {proyecto05_csv.shape[0]} filas × {proyecto05_csv.shape[1]} cols")
    print(f"proyecto04 CSV: {proyecto04_csv.shape[0]} filas × {proyecto04_csv.shape[1]} cols")
    print(f"Recomendación proyecto03: {recommendation}")
    print(f"Etapas UNKNOWN a eliminar: {compact_list(unknown_stages, 100) if unknown_stages else 'ninguna'}")
    print("Obra gris: clasificada como WON, no excluir")
    print("Sin Garantías: clasificada como WON")
    print("date_insert: proxy temporal")


if __name__ == "__main__":
    main()
