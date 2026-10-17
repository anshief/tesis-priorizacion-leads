import logging
import re
import sys
import unicodedata
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cleaning_helpers import (
    STAGE_PRIORITY,
    TEST_RECORD_PATTERN,
    build_variable_info,
    calculate_column_fullness,
    clean_campaign_spacing,
    clean_channel_value_conservative,
    clean_repeated_campaign_text,
    clean_text_value,
    find_email_columns,
    find_phone_columns,
    fix_duplicated_campaign_text,
    format_summary_value,
    normalize_for_comparison,
    normalize_marital_status,
    normalize_project,
    normalize_sex,
    normalize_tiene_hijos,
    print_column_fullness_report,
    readable_title,
    remove_copia_suffix,
)
from scripts.schema_contract import (
    keep_human_readable_labels,
    validate_analytical_label_schema,
)


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
warnings.filterwarnings("ignore")

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports/final")

INPUT_CSV = RAW_DIR / "leads_final_combined.csv"
CLEANED_CSV = PROCESSED_DIR / "leads_cleaned_final.csv"
CLEANED_WITH_LEAKAGE_CSV = PROCESSED_DIR / "leads_cleaned_with_leakage_final.csv"
FEATURES_CSV = PROCESSED_DIR / "leads_features_final.csv"
LEAKAGE_REF_CSV = PROCESSED_DIR / "leads_leakage_reference_final.csv"

REPORT_QUALITY = REPORTS_DIR / "data_quality_report.xlsx"
REPORT_SUMMARY = REPORTS_DIR / "cleaning_summary.md"
REPORT_DICT = REPORTS_DIR / "variable_dictionary.xlsx"
REPORT_REMOVED = REPORTS_DIR / "columns_removed.csv"
REPORT_NORM = REPORTS_DIR / "normalization_before_after.xlsx"


NON_WON_STAGES_RAW = {
    "contactados",
    "ingresados",
    "interesados",
    "visita modelo",
    "visita programada",
    "corto plazos",
    "cortos plazos",
}

WON_STAGES_FINAL_RAW = {
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

NON_WON_STAGES = {normalize_for_comparison(stage) for stage in NON_WON_STAGES_RAW}
WON_STAGES = {normalize_for_comparison(stage) for stage in WON_STAGES_FINAL_RAW}

LEAKAGE_COLS = [
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
]

HIGH_MISSING_DROP_COLS = [
    "probability",
    "CSI FOR Tiempo Seguimiento",
    "Ubicación Parqueos",
    "Parqueo 1",
    "Parqueo 2",
    "Parqueo 3",
    "enganche_pactado",
    "monto_prestamo",
    "mts_2",
    "moneda_de_enganche_pactado",
    "Bodega Asignada",
    "Unidad Referidora",
    "Razón Inversión",
    "Proyecto Referidor",
    "Tipo Modelo",
    "Cantidad Parqueos",
    "Modelo",
    "precio_de_venta",
    "valor_bien",
    "Unidad #",
    "Sector",
    "CSI Comentarios Adicionales",
    "CSI Calificación Proyecto",
    "CSI Conocimiento Asesor",
    "CSI Calificación Apartamento",
    "CSI Calificación Amenidades",
    "CSI Tiempo Seguimiento",
    "NPS Recomendación Proyecto",
    "Ubicación Bodega",
    "fecha_visita",
    "Tipo Visita",
    "moneda_de_precio_de_venta",
    "moneda_de_monto_prestamo",
    "moneda_de_valor_bien",
]

USER_DECISION_DROP_COLS = [
    "disponible",
    "Rango de Disponible",
    "Cuota deseada",
    "moneda_de_disponible",
]

# Columnas adicionales sin valor predictivo
STRUCTURAL_DROP_COLS = ["Proyecto.1", "visible_to"]

# Campañas donde el nombre de campaña y el canal se concatenaron por error.
# Valor: (nombre_campaña_correcto, canal_correcto)
CAMPANIA_CONCAT_CANAL = {
    "proyecto01 TOPPágina Web": ("proyecto01 TOP", "Página Web"),
    "proyecto01 FOOTERPágina Web": ("proyecto01 FOOTER", "Página Web"),
}

# Campañas que deben convertirse a NA (no son nombres de campaña válidos)
CAMPANIA_TO_NA = {
    "portal_demo proyecto02 LEADS ABR 2022, proyecto02 agente_demo LEADS ABR 2022",
    "proyecto02 LEADS ABR 2022, proyecto02 agente_demo LEADS ABR 2022",
    "Vivienda",
}

# Mapeo punto a punto para casos específicos aprobados:
# - Formato de fecha WhatsApp (NOV 24 -> NOV 2024)
# - Guión por espacio (proyecto01-PRINCIPAL-MAYO-25 -> proyecto01 PRINCIPAL MAYO 25)
# - Fusión de versiones proyecto01-PRINCIPAL-ABR-26
CAMPANIA_SPECIFIC_MAP = {
    # Grupo 4: variante de formato de año en WhatsApp
    "proyecto02 WHATSAPP NOV 24": "proyecto02 WHATSAPP NOV 2024",
    # Grupo 6: guión vs espacio - misma campaña
    "proyecto01-PRINCIPAL-MAYO-25": "proyecto01 PRINCIPAL MAYO 25",
    # Fusión de versiones -2 y -3 a la campaña base
    "proyecto01-PRINCIPAL-ABR-26-2": "proyecto01-PRINCIPAL-ABR-26",
    "proyecto01-PRINCIPAL-ABR-26-3": "proyecto01-PRINCIPAL-ABR-26",
}

COLUMNAS_CATEGORICAS_REPORTAR = [
    "Canal de Contacto",
    "Canal de Anuencia",
    "Proyecto",
    "Sexo",
    "Estado Cívil",
    "Razón de Compra",
    "Tipo Unidad",
    "Ingreso Familiar",
    "Núcleo Familiar",
    "Lugar Vivienda",
    "Lugar Trabajo",
    "tiene_hijos",
    "campania",
]

# Log de eliminaciones (lista de dicts)
removal_log = []


def run_initial_diagnosis(df: pd.DataFrame, report_path: Path) -> pd.DataFrame:
    """Genera diagnóstico inicial y lo guarda en la primera hoja del reporte."""
    rows = []
    for col in df.columns:
        s = df[col]
        missing = s.isna().sum()
        n_total = len(s)
        n_unique = s.nunique(dropna=True)
        sample = [str(v) for v in s.dropna().unique()[:5]]
        row = {
            "nombre_original": col,
            "dtype": str(s.dtype),
            "count_no_null": n_total - missing,
            "missing_count": missing,
            "missing_pct": round(missing / n_total * 100, 2),
            "unique_values": n_unique,
            "sample_values": " | ".join(sample),
        }
        if pd.api.types.is_numeric_dtype(s):
            row["min_value"] = s.min()
            row["max_value"] = s.max()
            row["mean_value"] = round(s.mean(), 2) if n_total - missing > 0 else None
        else:
            row["min_value"] = None
            row["max_value"] = None
            row["mean_value"] = None
        rows.append(row)
    diag = pd.DataFrame(rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        diag.to_excel(writer, sheet_name="diagnostico_inicial", index=False)
    over95 = (diag["missing_pct"] > 95).sum()
    over80 = ((diag["missing_pct"] > 80) & (diag["missing_pct"] <= 95)).sum()
    single = (diag["unique_values"] == 1).sum()
    logging.info(
        f"Diagnóstico: {over95} columnas >95% missing | "
        f"{over80} entre 80-95% | {single} con un único valor"
    )
    return diag


def standardize_etapa_trato(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize Etapa Trato only when the column exists.
    # This makes the stage names cleaner and easier to group later.
    if "Etapa Trato" not in df.columns:
        return df

    df["Etapa Trato"] = (
        df["Etapa Trato"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    return df


def create_target_won(df: pd.DataFrame) -> pd.DataFrame:
    # Start target_won as missing so uncertain rows can be removed later.
    # The final target should only contain 0 or 1 after filtering.
    df["target_won"] = pd.NA

    # Normalize stage and status before comparison.
    # These helper Series are not saved; they only make the rules reliable.
    etapa_normalized = df["Etapa Trato"].apply(normalize_for_comparison)
    status_normalized = df["status"].apply(normalize_for_comparison)

    # Non-won stages are target 0 when Etapa Trato is present.
    # These are early funnel stages, so they are treated as not won.
    df.loc[etapa_normalized.isin(NON_WON_STAGES), "target_won"] = 0

    # Reserva and later post-reserva stages are treated as won outcomes.
    df.loc[etapa_normalized.isin(WON_STAGES), "target_won"] = 1

    # Won status always means target 1, including rows with missing Etapa Trato.
    # This rule intentionally overrides the stage-based 0 if status says won.
    df.loc[status_normalized == "won", "target_won"] = 1

    # Use pandas nullable integer so values can be 0, 1, or missing.
    # Rows that are still missing are removed later before saving the final CSV.
    df["target_won"] = df["target_won"].astype("Int64")
    return df


def filter_and_log_target(df: pd.DataFrame) -> pd.DataFrame:
    undefined_count = int(df["target_won"].isna().sum())
    won_count = int((df["target_won"] == 1).sum())
    not_won_count = int((df["target_won"] == 0).sum())

    removal_log.append(
        {
            "tipo": "filas",
            "razon": "target_won_indefinido",
            "cantidad": undefined_count,
        }
    )

    df = df[df["target_won"].notna()].copy()
    logging.info(
        f"Distribución target_won después de filtrar indefinidos | "
        f"won: {won_count:,} | not-won: {not_won_count:,} | "
        f"indefinidos eliminados: {undefined_count:,}"
    )
    return df


def save_leakage_reference(df, leakage_cols, output_path):
    existing_leakage_cols = [col for col in leakage_cols if col in df.columns]
    reference_cols = [
        col
        for col in ["id_Trato", "target_won", *existing_leakage_cols]
        if col in df.columns
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[reference_cols].to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info(
        f"Referencia leakage guardada: {output_path} | "
        f"{len(reference_cols)} columnas"
    )


def build_cleaned_with_leakage_output(
    cleaned_df: pd.DataFrame,
    leakage_source_df: pd.DataFrame,
    leakage_cols,
) -> pd.DataFrame:
    existing_leakage_cols = [col for col in leakage_cols if col in leakage_source_df.columns]
    source_cols = [
        col
        for col in ["id_Trato", *existing_leakage_cols]
        if col in leakage_source_df.columns
    ]
    if "id_Trato" not in source_cols or "id_trato" not in cleaned_df.columns:
        return cleaned_df.copy()

    leakage_source = leakage_source_df[source_cols].copy()
    leakage_source, _ = standardize_column_names(leakage_source)
    leakage_source = leakage_source.loc[:, ~leakage_source.columns.duplicated()]

    merge_cols = [
        col
        for col in leakage_source.columns
        if col == "id_trato" or col not in cleaned_df.columns
    ]
    leakage_source = leakage_source[merge_cols].drop_duplicates(
        subset=["id_trato"],
        keep="first",
    )

    return cleaned_df.merge(leakage_source, on="id_trato", how="left")


def validate_final_outputs(cleaned_df: pd.DataFrame, with_leakage_df: pd.DataFrame) -> None:
    target_values = set(cleaned_df["target_won"].dropna().astype(int).unique())
    if target_values - {0, 1}:
        raise ValueError(f"target_won contiene valores inválidos: {target_values}")

    cleaned_leakage_cols = sorted(
        {
            snake_case_name(col)
            for col in LEAKAGE_COLS
        }
        & set(cleaned_df.columns)
    )
    if cleaned_leakage_cols:
        raise ValueError(
            f"Columnas de leakage presentes en leads_cleaned_final.csv: {cleaned_leakage_cols}"
        )

    required_with_leakage_cols = {"status", "etapa_trato"}
    missing_with_leakage = required_with_leakage_cols - set(with_leakage_df.columns)
    if missing_with_leakage:
        raise ValueError(
            "Columnas de leakage faltantes en leads_cleaned_with_leakage_final.csv: "
            f"{sorted(missing_with_leakage)}"
        )

    etapa = with_leakage_df["etapa_trato"].apply(normalize_for_comparison)
    obra_gris_bad = with_leakage_df.loc[
        etapa.eq("obra gris") & with_leakage_df["target_won"].ne(1)
    ]
    if len(obra_gris_bad):
        raise ValueError("Obra gris existe con target_won distinto de 1")

    sin_garantias_bad = with_leakage_df.loc[
        etapa.eq("sin garantias") & with_leakage_df["target_won"].ne(1)
    ]
    if len(sin_garantias_bad):
        raise ValueError("Sin Garantías existe con target_won distinto de 1")


def drop_columns_with_log(df, cols_to_drop, reason) -> pd.DataFrame:
    existing_cols = [col for col in cols_to_drop if col in df.columns]

    for col in existing_cols:
        series = df[col]
        missing_pct = round(series.isna().sum() / len(series) * 100, 2) if len(series) else 0
        unique_values = int(series.nunique(dropna=True))
        removal_log.append(
            {
                "nombre_original": col,
                "razon": reason,
                "missing_pct": missing_pct,
                "unique_values": unique_values,
            }
        )

    df = df.drop(columns=existing_cols)
    logging.info(f"Eliminadas {len(existing_cols)} columnas por: {reason}")
    return df


def choose_age_column(df) -> pd.DataFrame:
    edad_numeric = pd.to_numeric(df["edad"], errors="coerce")
    valid_edad = edad_numeric.between(18, 90).sum() / len(df) * 100
    valid_rango = df["Rango de Edad"].notna().sum() / len(df) * 100
    logging.info(
        f"DECISIÓN EDAD: 'edad' válida = {valid_edad:.1f}% | "
        f"'Rango de Edad' = {valid_rango:.1f}%"
    )

    if valid_edad >= valid_rango:
        df["edad"] = edad_numeric
        invalid_age = df["edad"].notna() & ~df["edad"].between(18, 90)
        invalid_count = int(invalid_age.sum())
        if invalid_count:
            df = df.loc[~invalid_age].copy()
        logging.info(f"Registros eliminados por 'edad' fuera de 18-90: {invalid_count:,}")
        df = drop_columns_with_log(df, ["Rango de Edad"], "eliminada_en_favor_de_edad")
        logging.info("DECISIÓN: se conserva 'edad', se elimina 'Rango de Edad'")
    else:
        df = drop_columns_with_log(df, ["edad"], "eliminada_en_favor_de_rango_de_edad")
        logging.info("DECISIÓN: se conserva 'Rango de Edad', se elimina 'edad'")

    return df


def clean_categorical_columns(df) -> pd.DataFrame:
    categorical_cleaners = [
        ("Canal de Anuencia", clean_channel_value_conservative),
        ("Canal de Contacto", clean_channel_value_conservative),
        ("Proyecto", normalize_project),
        ("Sexo", normalize_sex),
        ("Estado Cívil", normalize_marital_status),
    ]

    text_only_columns = [
        "Razón de Compra",
        "Tipo Unidad",
        "Ingreso Familiar",
        "Núcleo Familiar",
        "Lugar Vivienda",
        "Lugar Trabajo",
        "Participantes en Crédito",
        "Opciones Visitadas",
        "Sector Referidor",
        "Rango de Edad",
    ]

    for col, cleaner in categorical_cleaners:
        if col in df.columns:
            n_antes = df[col].nunique(dropna=True)
            df[col] = df[col].apply(cleaner)
            n_despues = df[col].nunique(dropna=True)
            n_na = df[col].isna().sum()
            logging.info(f"[{col}]: {n_antes} únicos antes → {n_despues} únicos después | {n_na} NAs")

    for col in text_only_columns:
        if col in df.columns:
            n_antes = df[col].nunique(dropna=True)
            df[col] = df[col].apply(clean_text_value)
            n_despues = df[col].nunique(dropna=True)
            n_na = df[col].isna().sum()
            logging.info(f"[{col}]: {n_antes} únicos antes → {n_despues} únicos después | {n_na} NAs")

    return df


def apply_children_signal_from_marital_status(df) -> pd.DataFrame:
    if "Estado Cívil" not in df.columns or "tiene_hijos" not in df.columns:
        return df

    estado_key = df["Estado Cívil"].apply(normalize_for_comparison)
    has_children_in_marital_status = (
        estado_key.str.contains("solter", na=False)
        & estado_key.str.contains("hijo|hija", na=False)
    )
    n_marked = int(has_children_in_marital_status.sum())
    df.loc[has_children_in_marital_status, "tiene_hijos"] = "con hijos"
    logging.info(
        "Señal de hijos movida desde Estado Cívil a tiene_hijos: "
        f"{n_marked:,} filas"
    )
    return df


def assert_no_no_registrado(df) -> None:
    columns_with_no_registrado = []
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            has_no_registrado = df[col].dropna().apply(
                lambda value: normalize_for_comparison(value) == "no registrado"
            ).any()
            if has_no_registrado:
                columns_with_no_registrado.append(col)

    assert not columns_with_no_registrado, (
        "'No registrado' aparece en columnas: "
        f"{columns_with_no_registrado}"
    )


def build_normalization_report(df_before, df_after, col) -> pd.DataFrame:
    """
    Construye tabla de mapeo valor_original → valor_normalizado
    con frecuencias y clasificación de la acción.
    """
    total = len(df_before)
    freq_before = df_before[col].apply(
        lambda value: value if pd.notna(value) else "<<NA>>"
    ).value_counts(dropna=False)

    # Obtener mapeo original → normalizado por índice (mismo df)
    pairs = list(zip(df_before[col].values, df_after[col].values))
    mapping = {}
    for orig, norm in pairs:
        orig_key = orig if pd.notna(orig) else "<<NA>>"
        if orig_key not in mapping:
            mapping[orig_key] = norm

    rows = []
    for orig_display in df_before[col].unique():
        orig_key = orig_display if pd.notna(orig_display) else "<<NA>>"
        norm_val = mapping.get(orig_key, pd.NA)
        freq_orig = int(freq_before.get(orig_key, 0))
        pct_orig = round(freq_orig / total * 100, 3)

        # Clasificar acción
        if pd.isna(orig_display) and pd.isna(norm_val):
            accion = "conservado_como_NA"
            confianza = "alto"
            revision = "no"
        elif pd.notna(orig_display) and pd.isna(norm_val):
            accion = "convertido_a_NA"
            confianza = "medio"
            revision = "sí"
        elif str(orig_display) == str(norm_val):
            accion = "sin_cambio"
            confianza = "alto"
            revision = "no"
        else:
            orig_norm_key = normalize_for_comparison(str(orig_display))
            norm_norm_key = (
                normalize_for_comparison(str(norm_val))
                if pd.notna(norm_val)
                else ""
            )
            if orig_norm_key == norm_norm_key:
                accion = "correccion_casing_tilde"
                confianza = "alto"
                revision = "no"
            else:
                accion = "normalizacion_aplicada"
                confianza = "alto"
                revision = "no"

        rows.append(
            {
                "valor_original": orig_display,
                "valor_normalizado": norm_val,
                "frecuencia_original": freq_orig,
                "pct_original": pct_orig,
                "accion_aplicada": accion,
                "nivel_confianza": confianza,
                "requiere_revision_manual": revision,
            }
        )

    return pd.DataFrame(rows).sort_values("frecuencia_original", ascending=False)


def run_granularity_check(df_before, df_after, col_list, report_path):
    summary_rows = []
    all_reports = {}

    for col in col_list:
        if col not in df_before.columns or col not in df_after.columns:
            continue

        uniq_before = df_before[col].nunique(dropna=True)
        uniq_after = df_after[col].nunique(dropna=True)
        pct_red = round((1 - uniq_after / uniq_before) * 100, 1) if uniq_before > 0 else 0
        alerta = pct_red > 20

        if alerta and col in ("Canal de Contacto", "Canal de Anuencia", "campania"):
            logging.warning(
                f"ALERTA GRANULARIDAD: '{col}' perdió {pct_red:.1f}% de categorías "
                f"({uniq_before} → {uniq_after}). Revisar normalización."
            )

        report_df = build_normalization_report(df_before, df_after, col)
        all_reports[col] = report_df
        n_revision = int((report_df["requiere_revision_manual"] == "sí").sum())

        summary_rows.append(
            {
                "columna": col,
                "unicos_antes": uniq_before,
                "unicos_despues": uniq_after,
                "pct_reduccion": pct_red,
                "alerta_granularidad": "SÍ" if alerta else "no",
                "valores_requieren_revision": n_revision,
            }
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(
            writer,
            sheet_name="resumen_granularidad",
            index=False,
        )
        for col_name, rep in all_reports.items():
            sheet_name = col_name[:31]
            rep.to_excel(writer, sheet_name=sheet_name, index=False)

    logging.info(f"Reporte de granularidad guardado: {report_path}")
    return pd.DataFrame(summary_rows)


def standardize_column_names(df) -> tuple[pd.DataFrame, dict]:
    column_mapping = {}

    for column in df.columns:
        clean_column = str(column).strip()
        clean_column = unicodedata.normalize("NFKD", clean_column)
        clean_column = "".join(
            char for char in clean_column if not unicodedata.combining(char)
        )
        clean_column = clean_column.lower()
        clean_column = re.sub(r"\s+", "_", clean_column)
        clean_column = re.sub(r"[^a-z0-9_]+", "_", clean_column)
        clean_column = re.sub(r"_+", "_", clean_column)
        clean_column = clean_column.strip("_")
        column_mapping[column] = clean_column

    return df.rename(columns=column_mapping), column_mapping


def remove_test_records(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    # Search every text column for internal/test markers.
    # If any text field contains a marker word, the whole row is removed.
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    if len(text_columns) == 0:
        return df, 0

    # Build one row-level mask so a match in any text column removes the row.
    test_mask = pd.Series(False, index=df.index)
    for column in text_columns:
        column_matches = df[column].astype("string").str.contains(
            TEST_RECORD_PATTERN,
            na=False,
            regex=True,
        )
        test_mask = test_mask | column_matches

    removed = int(test_mask.sum())
    return df.loc[~test_mask].copy(), removed


def add_normalized_contact_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # Create normalized email/phone helper columns only when matching columns exist.
    # These helper columns make duplicate detection more consistent.
    helper_columns = []

    for column in find_email_columns(df):
        # Emails are compared with the same normalized text logic used elsewhere.
        helper_column = f"normalized_{column}"
        df[helper_column] = df[column].apply(normalize_for_comparison)
        helper_columns.append(helper_column)

    for column in find_phone_columns(df):
        # Phones are compared by digits only, so spaces, dashes, and parentheses do not matter.
        helper_column = f"normalized_{column}"
        df[helper_column] = (
            df[column]
            .astype("string")
            .fillna("")
            .str.replace(r"\D+", "", regex=True)
        )
        helper_columns.append(helper_column)

    return df, helper_columns


def add_priority_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Add temporary columns used only to choose the best duplicate record.
    # They are dropped before the final cleaned CSV is saved.
    if "etapa_trato" in df.columns:
        etapa_normalized = df["etapa_trato"].apply(normalize_for_comparison)
        df["_stage_priority"] = etapa_normalized.map(STAGE_PRIORITY).fillna(0)
    else:
        df["_stage_priority"] = 0

    # Parse dates safely; invalid dates become missing and sort last.
    # This date is not used to delete rows; it only breaks ties between duplicates.
    if "fecha_creacion_trato" in df.columns:
        df["_created_at_priority"] = pd.to_datetime(
            df["fecha_creacion_trato"],
            errors="coerce",
        )
    else:
        df["_created_at_priority"] = pd.NaT

    return df


def deduplicate_with_priority(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    # Add normalized contact columns if email or phone columns exist.
    # The returned helper column names tell us which contact fields can be deduped.
    before = len(df)
    df, contact_helper_columns = add_normalized_contact_columns(df)

    # Sort once so every dedupe pass keeps the best available record.
    # Priority order: won rows first, then most advanced stage, then newest creation date.
    df = add_priority_columns(df)
    sort_columns = ["target_won", "_stage_priority", "_created_at_priority"]
    df = df.sort_values(sort_columns, ascending=[False, False, False])

    # Deduplicate by id_trato first using priority logic.
    # This is the strongest unique key, so it runs before email/phone dedupe.
    id_mask = df["id_trato"].notna() & (df["id_trato"].astype("string").str.strip() != "")
    df_with_id = df.loc[id_mask].drop_duplicates(subset=["id_trato"], keep="first")
    df_without_id = df.loc[~id_mask]
    df = pd.concat([df_with_id, df_without_id], ignore_index=False)
    df = df.sort_values(sort_columns, ascending=[False, False, False])

    # Deduplicate by normalized email/phone helper columns using the same priority.
    # Blank contact values are excluded so unrelated blank rows are not merged together.
    for helper_column in contact_helper_columns:
        has_value = df[helper_column].notna() & (df[helper_column] != "")
        df_with_value = df.loc[has_value].drop_duplicates(
            subset=[helper_column],
            keep="first",
        )
        df_without_value = df.loc[~has_value]
        df = pd.concat([df_with_value, df_without_value], ignore_index=False)
        df = df.sort_values(sort_columns, ascending=[False, False, False])

    # Remove temporary priority columns, but keep normalized email/phone columns.
    # Keeping normalized contact columns helps audit how contact dedupe was done.
    df = df.drop(columns=["_stage_priority", "_created_at_priority"])
    removed = before - len(df)
    return df.copy(), removed


def build_diagnostic_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        missing = s.isna().sum()
        n_total = len(s)
        n_unique = s.nunique(dropna=True)
        sample = [str(v) for v in s.dropna().unique()[:5]]
        row = {
            "nombre_original": col,
            "dtype": str(s.dtype),
            "count_no_null": n_total - missing,
            "missing_count": missing,
            "missing_pct": round(missing / n_total * 100, 2),
            "unique_values": n_unique,
            "sample_values": " | ".join(sample),
        }
        if pd.api.types.is_numeric_dtype(s):
            row["min_value"] = s.min()
            row["max_value"] = s.max()
            row["mean_value"] = round(s.mean(), 2) if n_total - missing > 0 else None
        else:
            row["min_value"] = None
            row["max_value"] = None
            row["mean_value"] = None
        rows.append(row)

    return pd.DataFrame(rows)


def save_final_diagnostic(df, report_path):
    final_diag = build_diagnostic_table(df)
    mode = "a" if report_path.exists() else "w"
    writer_kwargs = {"engine": "openpyxl", "mode": mode}
    if mode == "a":
        writer_kwargs["if_sheet_exists"] = "replace"

    with pd.ExcelWriter(report_path, **writer_kwargs) as writer:
        final_diag.to_excel(writer, sheet_name="diagnostico_final", index=False)

    logging.info(f"Reporte de calidad actualizado con diagnóstico final: {report_path}")


def snake_case_name(value) -> str:
    clean_value = str(value).strip()
    clean_value = unicodedata.normalize("NFKD", clean_value)
    clean_value = "".join(
        char for char in clean_value if not unicodedata.combining(char)
    )
    clean_value = clean_value.lower()
    clean_value = re.sub(r"\s+", "_", clean_value)
    clean_value = re.sub(r"[^a-z0-9_]+", "_", clean_value)
    clean_value = re.sub(r"_+", "_", clean_value)
    return clean_value.strip("_")


def save_variable_dictionary(df, column_mapping, removal_log, output_path):
    reverse_mapping = {snake: original for original, snake in column_mapping.items()}
    rows = []

    for column in df.columns:
        if column in {"id_trato", "person_id", "user_id"}:
            role = "identificador"
        elif column == "target_won":
            role = "target"
        else:
            role = "feature"

        rows.append(
            {
                "nombre_original": reverse_mapping.get(column, column),
                "nombre_snake_case": column,
                "rol": role,
                "descripcion": "",
            }
        )

    removed_columns_log = [
        entry
        for entry in removal_log
        if entry.get("tipo") == "columna" or "tipo" not in entry
    ]
    for entry in removed_columns_log:
        original_name = entry.get("nombre_original", "")
        rows.append(
            {
                "nombre_original": original_name,
                "nombre_snake_case": snake_case_name(original_name),
                "rol": "eliminada",
                "descripcion": "",
            }
        )

    df_features = df.drop(columns=["target_won"], errors="ignore")
    features_rows = pd.DataFrame(
        {
            "nombre_snake_case": df_features.columns,
            "descripcion": "",
        }
    )
    removed_df = pd.DataFrame(removed_columns_log)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="column_mapping", index=False)
        features_rows.to_excel(writer, sheet_name="features_finales", index=False)
        removed_df.to_excel(writer, sheet_name="columnas_eliminadas", index=False)

    logging.info(f"Diccionario de variables guardado: {output_path}")


def save_cleaning_summary(
    filas_iniciales,
    cols_iniciales,
    df,
    df_features,
    removal_log,
    output_path,
    n_test,
    n_dup,
    n_oos,
):
    removed_columns_log = [
        entry
        for entry in removal_log
        if entry.get("tipo") == "columna" or "tipo" not in entry
    ]
    removed_rows_log = [
        entry
        for entry in removal_log
        if entry.get("tipo") == "filas"
    ]

    removed_by_reason = (
        pd.DataFrame(removed_columns_log)
        .groupby("razon")
        .size()
        .reset_index(name="n_columnas")
        if removed_columns_log
        else pd.DataFrame(columns=["razon", "n_columnas"])
    )

    row_removals = {
        entry.get("razon"): entry.get("cantidad", 0)
        for entry in removed_rows_log
    }
    target_undefined = row_removals.get("target_won_indefinido", 0)
    target_counts = df["target_won"].value_counts(dropna=False).to_dict()
    target_one = int(target_counts.get(1, 0))
    target_zero = int(target_counts.get(0, 0))
    total_target = len(df)
    pct_one = round(target_one / total_target * 100, 2) if total_target else 0
    pct_zero = round(target_zero / total_target * 100, 2) if total_target else 0

    project_counts = (
        df["proyecto"].value_counts(dropna=False)
        if "proyecto" in df.columns
        else pd.Series(dtype="int64")
    )

    lines = [
        f"# Resumen de Limpieza — {INPUT_CSV.name}",
        f"**Fecha:** {datetime.today().strftime('%Y-%m-%d')}",
        "",
        "## Dimensiones",
        "| | Filas | Columnas |",
        "|---|---:|---:|",
        f"| Dataset original | {filas_iniciales:,} | {cols_iniciales} |",
        f"| Dataset limpio | {len(df):,} | {len(df.columns)} |",
        f"| Features | {len(df):,} | {len(df_features.columns)} |",
        "",
        "## Columnas eliminadas por categoría",
        "| Razón | n_columnas |",
        "|---|---:|",
    ]

    if len(removed_by_reason):
        for _, row in removed_by_reason.iterrows():
            lines.append(f"| {row['razon']} | {int(row['n_columnas'])} |")
    else:
        lines.append("| ninguna | 0 |")

    lines.extend(
        [
            "",
            "## Filas eliminadas",
            "| Razón | N |",
            "|---|---:|",
            f"| target_won indefinido | {target_undefined:,} |",
            f"| Registros de prueba | {n_test:,} |",
            f"| Duplicados | {n_dup:,} |",
            "",
            "## Variable objetivo (target_won)",
            "| Valor | N | % |",
            "|---|---:|---:|",
            f"| 1 (ganado) | {target_one:,} | {pct_one}% |",
            f"| 0 (no ganado) | {target_zero:,} | {pct_zero}% |",
            "",
            "## Proyectos",
            "| Proyecto | N |",
            "|---|---:|",
        ]
    )

    if len(project_counts):
        for project, count in project_counts.items():
            project_label = "" if pd.isna(project) else str(project)
            lines.append(f"| {project_label} | {int(count):,} |")
    else:
        lines.append("| columna proyecto no disponible | 0 |")

    lines.extend(
        [
            "",
            "## Proyectos fuera del alcance principal",
            f"Marcados con proyecto_fuera_alcance=1: {n_oos:,} registros",
            "",
            "## Columnas features finales",
        ]
    )

    for column in df_features.columns:
        lines.append(f"- {column}")

    age_column = "edad" if "edad" in df.columns else "rango_de_edad"
    lines.extend(
        [
            "",
            "## Decisiones metodológicas",
            f"- Variable de edad conservada: {age_column}",
            f"- Columnas de leakage preservadas en: {LEAKAGE_REF_CSV}",
        ]
    )

    lines.extend(
        [
            "- status='open' no define target por sí solo; la etapa gobierna salvo status='won'.",
            "",
            "## Integración de exports nuevos",
            "",
            "Se integraron registros de proyecto03, proyecto05 y proyecto04 a partir de exports con estructura distinta. Las columnas label fueron utilizadas como versión interpretable de las variables codificadas.",
            "",
            "## Clasificación de etapas posteriores",
            "",
            "Las etapas posteriores a reserva fueron clasificadas como target_won = 1. Esto incluye: Conformidad final, Entrega Fallida, Gestión Minutas, Gestión RGP, Liberación, Liquidación, Minutas Aprobadas, Orden Escrituración, Programación, Habitado, Operaciones, Reparaciones, Resguardo, Sin Garantías y Obra gris.",
            "",
            "## Advertencia sobre fecha_creacion_trato",
            "",
            "Para los exports nuevos, date_insert se conserva temporalmente como proxy de fecha_creacion_trato. Esta fecha corresponde a la exportación y no necesariamente a la fecha real de creación del lead.",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info(f"Resumen de limpieza guardado: {output_path}")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Archivo no encontrado: {INPUT_CSV}. Ejecuta 02_build_final_raw_dataset.py primero."
        )
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df, removed_code_columns = keep_human_readable_labels(df)
    for column in removed_code_columns:
        removal_log.append(
            {
                "tipo": "columna",
                "nombre_original": column,
                "razon": "codigo_reemplazado_por_label",
            }
        )
    logging.info(
        "Columnas homologadas al nombre label con fallback al valor original: "
        f"{removed_code_columns}"
    )
    filas_iniciales, cols_iniciales = df.shape
    logging.info(f"Dataset cargado: {filas_iniciales:,} filas × {cols_iniciales} columnas")
    run_initial_diagnosis(df, REPORT_QUALITY)
    logging.info("FASE 1 COMPLETADA: diagnóstico guardado")
    df = standardize_etapa_trato(df)
    df = create_target_won(df)
    df = filter_and_log_target(df)
    logging.info(
        f"target_won creada | won: {(df['target_won'] == 1).sum():,} | "
        f"not-won: {(df['target_won'] == 0).sum():,}"
    )
    logging.info("FASE 2 COMPLETADA: target_won lista")

    df_with_leakage_source = df.copy()

    # A) Guardar referencia de leakage ANTES de eliminar
    save_leakage_reference(df, LEAKAGE_COLS, LEAKAGE_REF_CSV)

    # B) Eliminar leakage
    df = drop_columns_with_log(df, LEAKAGE_COLS, "leakage")

    # C) Eliminar por missingness extremo / no disponibilidad temprana
    df = drop_columns_with_log(
        df,
        HIGH_MISSING_DROP_COLS,
        "missing_extremo_o_no_disponible_temprano",
    )

    # D) Eliminar por decisión del investigador
    df = drop_columns_with_log(df, USER_DECISION_DROP_COLS, "decision_investigador")

    # E) Eliminar Proyecto.1 y visible_to
    df = drop_columns_with_log(
        df,
        STRUCTURAL_DROP_COLS,
        "redundante_o_sin_valor_predictivo",
    )

    # F) Guardar log en CSV
    removed_columns_log = [entry for entry in removal_log if "nombre_original" in entry]
    pd.DataFrame(removed_columns_log).to_csv(
        REPORT_REMOVED,
        index=False,
        encoding="utf-8-sig",
    )

    # G) Loggear estado
    logging.info(f"Después de drops: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    logging.info(f"Columnas restantes: {list(df.columns)}")
    logging.info("FASE 3 COMPLETADA: columnas eliminadas y documentadas")

    df_before_norm = df.copy()

    df["tiene_hijos"] = df["tiene_hijos"].apply(normalize_tiene_hijos)
    df = apply_children_signal_from_marital_status(df)
    vc = df["tiene_hijos"].value_counts(dropna=False)
    logging.info(f"tiene_hijos normalizada:\n{vc}")

    df = choose_age_column(df)

    # ── LIMPIEZA DE CAMPANIA ──────────────────────────────────────
    logging.info("Limpiando campania...")
    n_na_antes = df["campania"].isna().sum()

    # PASO 1 — Strip básico
    df["campania"] = df["campania"].str.strip()

    # PASO 2 — Separar concatenaciones campaña+canal
    # Para filas donde el campo mezcla nombre de campaña y canal de contacto,
    # extrae ambos valores y los coloca en su columna correcta.
    n_concat = 0
    for concat_val, (camp_val, canal_val) in CAMPANIA_CONCAT_CANAL.items():
        mask = df["campania"] == concat_val
        if mask.any():
            df.loc[mask, "campania"] = camp_val
            # Actualizar Canal de Contacto si está vacío
            mask_cc = mask & df["Canal de Contacto"].isna()
            df.loc[mask_cc, "Canal de Contacto"] = canal_val
            # Actualizar Canal de Anuencia si está vacío
            mask_ca = mask & df["Canal de Anuencia"].isna()
            df.loc[mask_ca, "Canal de Anuencia"] = canal_val
            n_concat += int(mask.sum())
            logging.info(
                f"  Concat separada: '{concat_val}' → "
                f"campania='{camp_val}', canal='{canal_val}' "
                f"({mask.sum()} filas)"
            )
    logging.info(f"  Total filas con concatenación corregida: {n_concat}")

    # PASO 3 — Convertir valores inválidos a NA
    n_invalid = 0
    for bad_val in CAMPANIA_TO_NA:
        mask = df["campania"] == bad_val
        if mask.any():
            df.loc[mask, "campania"] = pd.NA
            n_invalid += int(mask.sum())
            logging.info(f"  Campaña inválida → NA: '{bad_val}' ({mask.sum()} filas)")
    logging.info(f"  Total filas convertidas a NA por valor inválido: {n_invalid}")

    # PASO 4 — Corregir auto-duplicación de texto sin separador
    # (ej: 'LEADS proyecto01 NOVLEADS proyecto01 NOV' → 'LEADS proyecto01 NOV')
    df["campania"] = df["campania"].apply(fix_duplicated_campaign_text)
    logging.info("  Auto-duplicación sin separador corregida")

    # PASO 5 — Corregir espaciado y puntuación
    # (doble espacio, punto final, NOV- DIC → NOV-DIC)
    df["campania"] = df["campania"].apply(clean_campaign_spacing)
    logging.info("  Espaciado y puntuación corregidos")

    # PASO 6 — Corregir typo CAMAPAÑA → CAMPAÑA
    df["campania"] = df["campania"].str.replace(
        "CAMAPAÑA", "CAMPAÑA", regex=False
    ).str.replace(
        "Camapaña", "Campaña", regex=False
    )
    logging.info("  Typo CAMAPAÑA corregido")

    # PASO 7 — Aplicar mapa de correcciones puntuales aprobadas
    # (WhatsApp NOV 24→2024, guión→espacio, versiones ABR-26)
    def _apply_specific_map(value):
        if pd.isna(value):
            return pd.NA
        return CAMPANIA_SPECIFIC_MAP.get(str(value), str(value))

    df["campania"] = df["campania"].apply(_apply_specific_map)
    logging.info("  Mapa de correcciones puntuales aplicado")

    # PASO 8 — Eliminar sufijo '- Copia' (fusionar con campaña original)
    df["campania"] = df["campania"].apply(remove_copia_suffix)
    logging.info("  Sufijos '- Copia' eliminados")

    # PASO 9 — Limpieza base de texto (clean_repeated_campaign_text existente)
    # Aplica title case y limpieza residual de texto repetido con separador.
    df["campania"] = df["campania"].apply(clean_repeated_campaign_text)
    logging.info("  Limpieza base de texto aplicada")

    # Log de resultado
    n_na_despues = df["campania"].isna().sum()
    n_unicos = df["campania"].nunique()
    logging.info(
        f"campania limpia: {n_unicos} valores únicos | "
        f"NA antes: {n_na_antes:,} → después: {n_na_despues:,}"
    )
    # ── FIN LIMPIEZA DE CAMPANIA ──────────────────────────────────

    df = clean_categorical_columns(df)

    granularity_summary = run_granularity_check(
        df_before_norm,
        df,
        COLUMNAS_CATEGORICAS_REPORTAR,
        REPORT_NORM,
    )
    print("\n=== RESUMEN DE GRANULARIDAD ===")
    print(granularity_summary.to_string(index=False))
    logging.info("FASE 4B COMPLETADA: reporte de granularidad generado")

    df["Fecha_Creación_Trato"] = pd.to_datetime(
        df["Fecha_Creación_Trato"],
        errors="coerce",
    )
    nat_count = df["Fecha_Creación_Trato"].isna().sum()
    logging.info(f"Fecha_Creación_Trato parseada | NaT: {nat_count}")

    out_of_scope = {"proyecto03", "proyecto04", "proyecto04", "proyecto05", "proyecto06"}
    df["proyecto_fuera_alcance"] = df["Proyecto"].apply(
        lambda value: 1
        if normalize_for_comparison(str(value)) in out_of_scope
        else 0
    )
    n_oos_pre_snake = int((df["proyecto_fuera_alcance"] == 1).sum())
    logging.info(
        f"Registros con proyecto fuera del alcance principal: {n_oos_pre_snake:,}"
    )

    allowed_tiene_hijos = {"sin hijos", "con hijos"}
    invalid_tiene_hijos = set(df["tiene_hijos"].dropna().unique()) - allowed_tiene_hijos
    assert not invalid_tiene_hijos, f"Valores inválidos en tiene_hijos: {invalid_tiene_hijos}"
    assert_no_no_registrado(df)

    removed_columns_log = [entry for entry in removal_log if "nombre_original" in entry]
    pd.DataFrame(removed_columns_log).to_csv(
        REPORT_REMOVED,
        index=False,
        encoding="utf-8-sig",
    )

    logging.info(
        f"Estado después de limpieza de variables: "
        f"{df.shape[0]:,} filas × {df.shape[1]} columnas"
    )
    logging.info("FASE 4 COMPLETADA: variables limpiadas")

    df, column_mapping = standardize_column_names(df)
    logging.info("Columnas renombradas a snake_case")
    for orig, snake in column_mapping.items():
        logging.info(f"  {orig!r} → {snake!r}")

    # Test records
    df, n_test = remove_test_records(df)
    logging.info(f"Registros de prueba eliminados: {n_test}")
    removal_log.append({"tipo": "filas", "razon": "test_record", "cantidad": n_test})

    # Deduplicación
    df, n_dup = deduplicate_with_priority(df)
    logging.info(f"Duplicados eliminados: {n_dup}")
    removal_log.append({"tipo": "filas", "razon": "duplicado", "cantidad": n_dup})

    # Verificar fecha_creacion_trato (ya parseada a datetime)
    if "fecha_creacion_trato" in df.columns:
        fecha_min = pd.Timestamp("2017-01-01")
        fecha_max = pd.Timestamp("2026-05-01")
        fuera = (
            (df["fecha_creacion_trato"] < fecha_min)
            | (df["fecha_creacion_trato"] > fecha_max)
        ).sum()
        logging.info(f"Fechas de creación fuera del rango 2017-2026: {fuera}")

    removed_columns_log = [entry for entry in removal_log if "nombre_original" in entry]
    pd.DataFrame(removed_columns_log).to_csv(
        REPORT_REMOVED,
        index=False,
        encoding="utf-8-sig",
    )

    logging.info(
        f"Estado final pre-guardado: {df.shape[0]:,} filas × {df.shape[1]} columnas"
    )
    logging.info("FASE 5 COMPLETADA: snake_case, dedup, consistencia")

    # Dataset modelable final con target_won.
    validate_analytical_label_schema(df, CLEANED_CSV.name)
    df.to_csv(CLEANED_CSV, index=False, encoding="utf-8-sig")
    logging.info(
        f"Dataset limpio guardado: {CLEANED_CSV} | "
        f"{df.shape[0]:,} × {df.shape[1]}"
    )

    df_with_leakage = build_cleaned_with_leakage_output(
        df,
        df_with_leakage_source,
        LEAKAGE_COLS,
    )
    validate_final_outputs(df, df_with_leakage)
    validate_analytical_label_schema(
        df_with_leakage,
        CLEANED_WITH_LEAKAGE_CSV.name,
    )
    df_with_leakage.to_csv(
        CLEANED_WITH_LEAKAGE_CSV,
        index=False,
        encoding="utf-8-sig",
    )
    logging.info(
        f"Dataset con leakage guardado: {CLEANED_WITH_LEAKAGE_CSV} | "
        f"{df_with_leakage.shape[0]:,} × {df_with_leakage.shape[1]}"
    )

    # Dataset final sin target_won.
    df_features = df.drop(columns=["target_won"], errors="ignore")
    df_features.to_csv(FEATURES_CSV, index=False, encoding="utf-8-sig")
    logging.info(f"Features guardadas: {FEATURES_CSV} | {df_features.shape[1]} columnas")
    logging.info(f"Columnas en features: {list(df_features.columns)}")

    save_final_diagnostic(df, REPORT_QUALITY)
    save_variable_dictionary(df, column_mapping, removal_log, REPORT_DICT)

    if "proyecto_fuera_alcance" in df.columns:
        n_oos = int((df["proyecto_fuera_alcance"] == 1).sum())
    elif "proyecto" in df.columns:
        out_of_scope = {"proyecto03", "proyecto04", "proyecto04", "proyecto05", "proyecto06"}
        n_oos = int(
            df["proyecto"]
            .apply(lambda value: normalize_for_comparison(str(value)) in out_of_scope)
            .sum()
        )
    else:
        n_oos = 0

    save_cleaning_summary(
        filas_iniciales,
        cols_iniciales,
        df,
        df_features,
        removal_log,
        REPORT_SUMMARY,
        n_test,
        n_dup,
        n_oos,
    )
    logging.info("FASE 6 COMPLETADA: todos los archivos guardados")


if __name__ == "__main__":
    main()
