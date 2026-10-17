"""Contrato central de columnas para los datasets analíticos finales."""

from __future__ import annotations


ANALYTICAL_LABEL_REPLACEMENTS = {
    "apartamento_no": "apartamento_no_label",
    "apto_referidor": "apto_referidor_label",
    "bodega_asignada": "bodega_asignada_label",
    "cantidad_parqueos": "cantidad_parqueos_label",
    "csi_calificacion_amenidades": "csi_calificacion_amenidades_label",
    "csi_calificacion_apartamento": "csi_calificacion_apartamento_label",
    "csi_calificacion_proyecto": "csi_calificacion_proyecto_label",
    "csi_comentarios_adicionales": "csi_comentarios_adicionales_label",
    "csi_conocimiento_asesor": "csi_conocimiento_asesor_label",
    "csi_tiempo_seguimiento": "csi_tiempo_seguimiento_label",
    "cuota_deseada": "cuota_deseada_label",
    "modelo": "modelo_label",
    "nps_recomendacion_proyecto": "nps_recomendacion_proyecto_label",
    "parqueo_1": "parqueo_1_label",
    "parqueo_2": "parqueo_2_label",
    "parqueo_3": "parqueo_3_label",
    "proyecto_referidor": "proyref_label",
    "razon_inversion": "razon_inversion_label",
    "tipo_modelo": "tipo_modelo_label",
    "tipo_visita": "tipovisita_label",
    "torre_referidora": "torre_referidora_label",
    "ubicacion_bodega": "ubicacion_bodega_label",
    "ubicacion_parqueos": "ubicacion_parqueos_label",
}

REMOVED_LABEL_COLUMNS = {"tipomodelo_label"}
REQUIRED_ANALYTICAL_LABEL_COLUMNS = set(ANALYTICAL_LABEL_REPLACEMENTS.values())
FORBIDDEN_ANALYTICAL_COLUMNS = (
    set(ANALYTICAL_LABEL_REPLACEMENTS) | REMOVED_LABEL_COLUMNS
)


def keep_human_readable_labels(df):
    """Homologa cada par bajo el nombre label, con fallback al valor original.

    Reglas por variable:
    - Si existen ambas columnas, prioriza el valor label y completa sus nulos
      con la columna original.
    - Si solo existe la columna original, la renombra al nombre label canónico.
    - Si solo existe label, la conserva sin cambios.
    - ``tipomodelo_label`` se elimina siempre.
    """
    result = df.copy()
    processed_columns = []

    for original, label in ANALYTICAL_LABEL_REPLACEMENTS.items():
        original_exists = original in result.columns
        label_exists = label in result.columns

        if original_exists and label_exists:
            result[label] = result[label].where(
                result[label].notna(),
                result[original],
            )
            result = result.drop(columns=[original])
            processed_columns.append(original)
        elif original_exists:
            result = result.rename(columns={original: label})
            processed_columns.append(original)

    removed_duplicates = sorted(REMOVED_LABEL_COLUMNS & set(result.columns))
    if removed_duplicates:
        result = result.drop(columns=removed_duplicates)
        processed_columns.extend(removed_duplicates)

    return result, sorted(processed_columns)


def validate_analytical_label_schema(df, dataset_name, require_all_labels=True):
    """Valida que no existan códigos duplicados y que se conserven los labels."""
    forbidden = sorted(FORBIDDEN_ANALYTICAL_COLUMNS & set(df.columns))
    if forbidden:
        raise AssertionError(
            f"{dataset_name} contiene columnas prohibidas de código/duplicadas: "
            f"{forbidden}"
        )

    if require_all_labels:
        missing_labels = sorted(
            REQUIRED_ANALYTICAL_LABEL_COLUMNS - set(df.columns)
        )
        if missing_labels:
            raise AssertionError(
                f"{dataset_name} no contiene columnas label requeridas: "
                f"{missing_labels}"
            )
