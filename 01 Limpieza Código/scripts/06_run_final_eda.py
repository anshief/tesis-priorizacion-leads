import hashlib
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from schema_contract import validate_analytical_label_schema


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
warnings.filterwarnings("ignore")

INPUT_CSV = Path("data/features/leads_enriched_final.csv")
REPORT_DIR = Path("reports/eda")
REPORT_XLSX = REPORT_DIR / "eda_technical_report.xlsx"
REPORT_MD = REPORT_DIR / "eda_summary.md"

EXPECTED_SHEETS = [
    "dataset_overview",
    "target_balance",
    "missingness",
    "categorical_cardinality",
    "numeric_summary",
    "categorical_vs_target",
    "numeric_vs_target",
    "category_support_flags",
    "eda_recommendations",
]


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_validate_input(path):
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    input_hash = file_md5(path)
    df = pd.read_csv(path, low_memory=False)
    if len(df) == 0:
        raise AssertionError("leads_enriched_final.csv no tiene filas")
    if "target_won" not in df.columns:
        raise AssertionError("target_won no existe")
    if df["target_won"].isna().any():
        raise AssertionError("target_won tiene nulos")

    validate_analytical_label_schema(df, path.name)

    target_values = set(df["target_won"].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise AssertionError(f"target_won tiene valores inesperados: {target_values}")

    logging.info(
        f"leads_enriched_final.csv cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas"
    )
    return df, input_hash


def detect_datetime_columns(df):
    datetime_cols = []
    for col in df.columns:
        if "fecha" in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col]):
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > 0:
                datetime_cols.append(col)
    return datetime_cols


def classify_column_role(col, series, datetime_cols):
    col_lower = col.lower()
    if col == "target_won":
        return "target"
    if "id" in col_lower or col in {"id_trato", "person_id", "user_id"}:
        return "id"
    if "fecha" in col_lower or col in datetime_cols:
        return "fecha"
    if pd.api.types.is_numeric_dtype(series):
        return "numerica"
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    ):
        return "categorica"
    return "otro"


def build_dataset_overview(df):
    datetime_cols = detect_datetime_columns(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    total_missing = int(df.isna().sum().sum())
    total_cells = int(df.shape[0] * df.shape[1])
    pct_missing = round(total_missing / total_cells * 100, 3) if total_cells else 0

    id_duplicates = pd.NA
    if "id_trato" in df.columns:
        id_duplicates = int(df["id_trato"].duplicated(keep=False).sum())

    n_projects = int(df["proyecto"].nunique(dropna=True)) if "proyecto" in df.columns else pd.NA

    fecha_min = pd.NA
    fecha_max = pd.NA
    if "fecha_creacion_trato" in df.columns:
        fechas = pd.to_datetime(df["fecha_creacion_trato"], errors="coerce")
        fecha_min = fechas.min()
        fecha_max = fechas.max()

    return pd.DataFrame(
        [
            {"metrica": "filas", "valor": len(df)},
            {"metrica": "columnas", "valor": len(df.columns)},
            {"metrica": "columnas_numericas", "valor": len(numeric_cols)},
            {"metrica": "columnas_categoricas", "valor": len(categorical_cols)},
            {"metrica": "columnas_datetime_detectables", "valor": len(datetime_cols)},
            {"metrica": "total_valores_faltantes", "valor": total_missing},
            {"metrica": "pct_global_valores_faltantes", "valor": pct_missing},
            {"metrica": "duplicados_por_id_trato", "valor": id_duplicates},
            {"metrica": "proyectos_unicos", "valor": n_projects},
            {"metrica": "fecha_creacion_trato_min", "valor": fecha_min},
            {"metrica": "fecha_creacion_trato_max", "valor": fecha_max},
        ]
    )


def build_target_balance(df):
    counts = df["target_won"].value_counts(dropna=False).sort_index()
    total = len(df)
    balance = counts.reset_index()
    balance.columns = ["target_won", "n"]
    balance["porcentaje"] = round(balance["n"] / total * 100, 3)

    positive_rate = float((df["target_won"] == 1).sum() / total)
    minority = counts.min()
    majority = counts.max()
    imbalance_ratio = float(majority / minority) if minority else np.inf

    metrics = pd.DataFrame(
        [
            {"target_won": "positive_rate", "n": positive_rate, "porcentaje": ""},
            {"target_won": "imbalance_ratio", "n": imbalance_ratio, "porcentaje": ""},
        ]
    )
    return pd.concat([balance, metrics], ignore_index=True)


def build_missingness(df):
    datetime_cols = detect_datetime_columns(df)
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "columna": col,
                "tipo_dato": str(s.dtype),
                "n_missing": int(s.isna().sum()),
                "pct_missing": round(s.isna().mean() * 100, 3),
                "n_unique": int(s.nunique(dropna=True)),
                "rol_sugerido": classify_column_role(col, s, datetime_cols),
            }
        )
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False)


def categorical_columns(df):
    cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return [col for col in cols if col != "target_won"]


def numeric_columns(df):
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in cols if col != "target_won"]


def build_categorical_cardinality(df):
    rows = []
    for col in categorical_columns(df):
        s = df[col]
        vc = s.value_counts(dropna=False)
        top_1_valor = vc.index[0] if len(vc) else pd.NA
        top_1_n = int(vc.iloc[0]) if len(vc) else 0
        top_1_pct = top_1_n / len(df) if len(df) else 0
        top_5_cobertura_pct = vc.head(5).sum() / len(df) if len(df) else 0
        n_unique = int(s.nunique(dropna=True))
        pct_missing = float(s.isna().mean())

        risks = []
        if n_unique > 50:
            risks.append("alta_cardinalidad")
        if top_1_pct >= 0.95:
            risks.append("dominada_por_una_categoria")
        if pct_missing >= 0.80:
            risks.append("missing_alto")
        if not risks:
            risks.append("ok")

        rows.append(
            {
                "columna": col,
                "n_unique": n_unique,
                "pct_unique_sobre_filas": round(n_unique / len(df) * 100, 3),
                "n_missing": int(s.isna().sum()),
                "pct_missing": round(pct_missing * 100, 3),
                "top_1_valor": "NA" if pd.isna(top_1_valor) else top_1_valor,
                "top_1_n": top_1_n,
                "top_1_pct": round(top_1_pct, 3),
                "top_5_cobertura_pct": round(top_5_cobertura_pct, 3),
                "riesgo_modelado": ";".join(risks),
            }
        )
    return pd.DataFrame(rows).sort_values(["riesgo_modelado", "n_unique"])


def build_numeric_summary(df):
    rows = []
    for col in numeric_columns(df):
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "columna": col,
                "count": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "pct_missing": round(s.isna().mean() * 100, 3),
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(),
                "min": s.min(),
                "p01": s.quantile(0.01),
                "p05": s.quantile(0.05),
                "p25": s.quantile(0.25),
                "p75": s.quantile(0.75),
                "p95": s.quantile(0.95),
                "p99": s.quantile(0.99),
                "max": s.max(),
                "n_unique": int(s.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def build_categorical_vs_target(df):
    rows = []
    total = len(df)
    for col in categorical_columns(df):
        grouped = (
            df.groupby(col, dropna=False)["target_won"]
            .agg(["count", "sum"])
            .reset_index()
        )
        for _, row in grouped.iterrows():
            n = int(row["count"])
            won = int(row["sum"])
            category = row[col]
            rows.append(
                {
                    "variable": col,
                    "categoria": "NA" if pd.isna(category) else category,
                    "n": n,
                    "won": won,
                    "conversion_rate": won / n if n else 0,
                    "porcentaje_del_total": n / total if total else 0,
                    "soporte_suficiente": n >= 100,
                }
            )
    return pd.DataFrame(rows).sort_values(["variable", "n"], ascending=[True, False])


def build_numeric_vs_target(df):
    rows = []
    for col in numeric_columns(df):
        for target_value, group in df.groupby("target_won"):
            s = pd.to_numeric(group[col], errors="coerce")
            rows.append(
                {
                    "variable": col,
                    "target_won": target_value,
                    "n": int(s.notna().sum()),
                    "mean": s.mean(),
                    "median": s.median(),
                    "std": s.std(),
                    "min": s.min(),
                    "p25": s.quantile(0.25),
                    "p75": s.quantile(0.75),
                    "max": s.max(),
                }
            )
    return pd.DataFrame(rows)


def build_category_support_flags(categorical_vs_target):
    rows = []
    for variable, group in categorical_vs_target.groupby("variable"):
        low_support = group[group["n"] < 100]
        n_categories = len(group)
        n_low = len(low_support)
        pct_low_categories = n_low / n_categories if n_categories else 0
        n_low_records = int(low_support["n"].sum())
        n_total_records = int(group["n"].sum())
        pct_low_records = n_low_records / n_total_records if n_total_records else 0

        if pct_low_categories > 0.50:
            recommendation = "revisar_agrupacion"
        elif pct_low_categories >= 0.20:
            recommendation = "usable_con_cuidado"
        else:
            recommendation = "ok"

        rows.append(
            {
                "variable": variable,
                "n_categorias_total": n_categories,
                "n_categorias_bajo_soporte": n_low,
                "pct_categorias_bajo_soporte": round(pct_low_categories * 100, 3),
                "n_registros_en_bajo_soporte": n_low_records,
                "pct_registros_en_bajo_soporte": round(pct_low_records * 100, 3),
                "recomendacion": recommendation,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["recomendacion", "pct_categorias_bajo_soporte"],
        ascending=[True, False],
    )


def build_eda_recommendations(
    target_balance,
    missingness,
    categorical_cardinality,
    support_flags,
):
    rows = []

    imbalance_ratio = float(
        target_balance.loc[target_balance["target_won"] == "imbalance_ratio", "n"].iloc[0]
    )
    rows.append(
        {
            "tema": "desbalance de target_won",
            "hallazgo": f"imbalance_ratio={imbalance_ratio:.1f}",
            "recomendacion": "Documentar el desbalance antes de modelado; no aplicar técnicas aún en EDA.",
            "prioridad": "alta",
        }
    )

    high_missing = missingness[missingness["pct_missing"] > 80]
    rows.append(
        {
            "tema": "missingness",
            "hallazgo": f"{len(high_missing)} columnas tienen más de 80% missing",
            "recomendacion": "Revisar utilidad y estrategia de imputación o exclusión en etapa de modelado.",
            "prioridad": "alta" if len(high_missing) else "media",
        }
    )

    high_card = categorical_cardinality[
        categorical_cardinality["riesgo_modelado"].str.contains("alta_cardinalidad", na=False)
    ]
    rows.append(
        {
            "tema": "alta cardinalidad",
            "hallazgo": f"{len(high_card)} variables categóricas tienen más de 50 categorías",
            "recomendacion": "Evaluar agrupación, hashing o encoding adecuado en etapa posterior.",
            "prioridad": "media" if len(high_card) else "baja",
        }
    )

    dominated = categorical_cardinality[
        categorical_cardinality["riesgo_modelado"].str.contains(
            "dominada_por_una_categoria",
            na=False,
        )
    ]
    rows.append(
        {
            "tema": "categorías dominantes",
            "hallazgo": f"{len(dominated)} variables están dominadas por una categoría",
            "recomendacion": "Revisar baja variabilidad antes de incluirlas en modelado.",
            "prioridad": "media" if len(dominated) else "baja",
        }
    )

    low_support_vars = support_flags[
        support_flags["recomendacion"].isin(["revisar_agrupacion", "usable_con_cuidado"])
    ]
    rows.append(
        {
            "tema": "bajo soporte categórico",
            "hallazgo": f"{len(low_support_vars)} variables tienen muchas categorías con n < 100",
            "recomendacion": "Interpretar conversion_rate con soporte mínimo; evitar conclusiones por categorías pequeñas.",
            "prioridad": "media" if len(low_support_vars) else "baja",
        }
    )

    rows.append(
        {
            "tema": "Power BI vs Python",
            "hallazgo": "Power BI será usado para visualización comercial; Python para soporte técnico del modelado.",
            "recomendacion": "Mantener EDA técnico reproducible en Python y narrativa visual en Power BI.",
            "prioridad": "media",
        }
    )

    return pd.DataFrame(rows)


def markdown_table(df, columns=None):
    if columns is not None:
        df = df[columns]
    if df.empty:
        return "| |\n|---|\n| sin datos |"

    lines = [
        "| " + " | ".join(df.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = ["NA" if pd.isna(value) else str(value) for value in row.tolist()]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_markdown_summary(
    df,
    overview,
    target_balance,
    missingness,
    categorical_cardinality,
    categorical_vs_target,
    numeric_vs_target,
):
    target_table = target_balance[
        target_balance["target_won"].isin([0, 1])
    ].copy()
    positive_rate = float((df["target_won"] == 1).sum() / len(df) * 100)
    top_missing = missingness.head(10)[["columna", "pct_missing", "n_missing"]]
    risk_categoricals = categorical_cardinality[
        categorical_cardinality["riesgo_modelado"] != "ok"
    ][["columna", "n_unique", "top_1_pct", "pct_missing", "riesgo_modelado"]]
    top_conversion = categorical_vs_target[
        categorical_vs_target["soporte_suficiente"]
    ].sort_values("conversion_rate", ascending=False).head(10)
    top_conversion = top_conversion[
        ["variable", "categoria", "n", "won", "conversion_rate"]
    ].copy()
    top_conversion["conversion_rate"] = top_conversion["conversion_rate"].round(4)

    numeric_brief = numeric_vs_target[
        numeric_vs_target["variable"].isin(
            ["edad", "m2_minimo", "m2_maximo", "precio_minimo", "precio_promedio"]
        )
    ].copy()
    numeric_brief = numeric_brief[
        ["variable", "target_won", "n", "mean", "median"]
    ].round(3)

    lines = [
        "# Resumen EDA Técnico — Leads Enriched",
        f"**Input:** {INPUT_CSV}  ",
        f"**Fecha de ejecución:** {datetime.today().strftime('%Y-%m-%d')}  ",
        f"**Filas:** {len(df):,}  ",
        f"**Columnas:** {len(df.columns)}  ",
        "",
        "## 1. Balance del target",
        markdown_table(target_table[["target_won", "n", "porcentaje"]]),
        "",
        f"La tasa positiva de target_won es {positive_rate:.3f}%, por lo que el dataset está fuertemente desbalanceado.",
        "",
        "## 2. Missingness",
        "Top 10 columnas con mayor porcentaje de missing:",
        markdown_table(top_missing),
        "",
        "## 3. Cardinalidad categórica",
        "Variables con alta cardinalidad o dominadas por una categoría:",
        markdown_table(risk_categoricals),
        "",
        "## 4. Variables categóricas vs target",
        "Las tasas de conversión por categoría deben interpretarse junto con el soporte mínimo.",
        "Top 10 categorías con mayor conversion_rate entre categorías con n >= 100:",
        markdown_table(top_conversion),
        "",
        "## 5. Variables numéricas vs target",
        "Resumen breve de variables numéricas comparadas por target:",
        markdown_table(numeric_brief),
        "",
        "## 6. Recomendaciones para Power BI",
        "Power BI puede usarse para:",
        "- distribución de leads;",
        "- tasa de conversión por proyecto;",
        "- tasa de conversión por campaña;",
        "- tasa de conversión por canal;",
        "- evolución temporal;",
        "- filtros por proyecto, campaña, edad y atributos de producto.",
        "",
        "## 7. Recomendaciones para Python/modelado posterior",
        "Python debe usarse después para:",
        "- train-test split;",
        "- encoding;",
        "- manejo de desbalance;",
        "- validación;",
        "- baseline;",
        "- métricas de clasificación;",
        "- evaluación por top-k si aplica.",
        "",
        "## 8. Limitaciones",
        "- el EDA no implica causalidad;",
        "- tasas con bajo soporte pueden ser inestables;",
        "- variables derivadas de campaña reflejan estrategia comercial y pueden dominar patrones;",
        "- atributos de proyecto son contextuales, no atributos individuales del lead;",
        "- precios provienen de fuentes 2024 o externas documentadas.",
    ]
    return "\n".join(lines) + "\n"


def write_reports(df):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    overview = build_dataset_overview(df)
    target_balance = build_target_balance(df)
    missingness = build_missingness(df)
    cat_cardinality = build_categorical_cardinality(df)
    num_summary = build_numeric_summary(df)
    cat_vs_target = build_categorical_vs_target(df)
    num_vs_target = build_numeric_vs_target(df)
    support_flags = build_category_support_flags(cat_vs_target)
    recommendations = build_eda_recommendations(
        target_balance,
        missingness,
        cat_cardinality,
        support_flags,
    )

    with pd.ExcelWriter(REPORT_XLSX, engine="openpyxl") as writer:
        overview.to_excel(writer, sheet_name="dataset_overview", index=False)
        target_balance.to_excel(writer, sheet_name="target_balance", index=False)
        missingness.to_excel(writer, sheet_name="missingness", index=False)
        cat_cardinality.to_excel(
            writer,
            sheet_name="categorical_cardinality",
            index=False,
        )
        num_summary.to_excel(writer, sheet_name="numeric_summary", index=False)
        cat_vs_target.to_excel(writer, sheet_name="categorical_vs_target", index=False)
        num_vs_target.to_excel(writer, sheet_name="numeric_vs_target", index=False)
        support_flags.to_excel(writer, sheet_name="category_support_flags", index=False)
        recommendations.to_excel(writer, sheet_name="eda_recommendations", index=False)

    REPORT_MD.write_text(
        build_markdown_summary(
            df,
            overview,
            target_balance,
            missingness,
            cat_cardinality,
            cat_vs_target,
            num_vs_target,
        ),
        encoding="utf-8",
    )

    return {
        "overview": overview,
        "target_balance": target_balance,
        "missingness": missingness,
        "categorical_cardinality": cat_cardinality,
        "numeric_summary": num_summary,
        "categorical_vs_target": cat_vs_target,
        "numeric_vs_target": num_vs_target,
        "support_flags": support_flags,
        "recommendations": recommendations,
    }


def validate_outputs(input_hash_before):
    assert REPORT_XLSX.exists() and REPORT_XLSX.stat().st_size > 0
    assert REPORT_MD.exists() and REPORT_MD.stat().st_size > 0

    sheet_names = pd.ExcelFile(REPORT_XLSX).sheet_names
    missing_sheets = [sheet for sheet in EXPECTED_SHEETS if sheet not in sheet_names]
    assert not missing_sheets, f"Hojas faltantes: {missing_sheets}"

    input_hash_after = file_md5(INPUT_CSV)
    assert input_hash_before == input_hash_after, "leads_enriched_final.csv fue modificado"

    training_outputs = list(Path("data").glob("**/*train*")) + list(
        Path("data").glob("**/*training*")
    )
    assert not training_outputs, (
        "Se detectaron posibles datasets de entrenamiento: "
        f"{[str(path) for path in training_outputs]}"
    )


def main():
    df, input_hash_before = load_and_validate_input(INPUT_CSV)
    outputs = write_reports(df)
    validate_outputs(input_hash_before)

    print("EDA TÉCNICO COMPLETADO")
    print(f"Input: {INPUT_CSV}")
    print(f"Output Excel: {REPORT_XLSX}")
    print(f"Output Markdown: {REPORT_MD}")
    print(f"Filas analizadas: {len(df):,}")
    print(f"Columnas analizadas: {len(df.columns)}")
    print("Hojas generadas: 9")
    print("STOP")


if __name__ == "__main__":
    main()
