"""Edición pública: utiliza artefactos sintéticos generados en examples/."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .columnas import obtener_serie
from .normalizacion import (
    classify_campania_audiencia,
    classify_campania_tipo,
    clean_channel_value_conservative,
    clean_text_value,
    normalize_marital_status,
    normalize_sex,
)

# Rango de edad considerado válido en la limpieza original. En entrenamiento las
# filas fuera de rango se eliminaban; en inferencia no se puede perder ningún
# lead, así que el valor pasa a nulo y lo resuelven los imputadores del modelo.
EDAD_MIN, EDAD_MAX = 18, 90

# Punto medio de cada rango declarado de ingreso familiar, en miles de quetzales.
# Copiado literal de la celda de features del notebook final.
INCOME_MID = {
    "Menos De Q10k": 8,
    "Q 10k A Q13.9k": 12,
    "Q 14k A Q17.9k": 16,
    "Q18k A Q21.9k": 20,
    "Q22k A Q25.9k": 24,
    "Q26k A Q29.9k": 28,
    "Q30k A 33.9k": 32,
    "Q34k A 37.9k": 36,
    "Q38k A 41.9k": 40,
    "Q42k A 45.9k": 44,
    "46k A 49.9k": 48,
    "Q50k A Más": 55,
}

# Agrupación de canales en familias. Copiado literal de §21.4 del notebook final.
CANAL_FAMILIA = {
    "Facebook": "pauta_digital",
    "Instagram": "pauta_digital",
    "Fanpage Líderes": "pauta_digital",
    "Inbox": "pauta_digital",
    "portal_demo": "portal",
    "Página Web": "portal",
    "WhatsApp": "directo",
    "Llamada": "directo",
    "Referencia": "referido",
    "Friends/Family": "referido",
    "Shareholders Sale": "referido",
    "Sede": "presencial",
    "ExpoCasa": "presencial",
    "ExpoCasa Virtual": "presencial",
    "Visita Inédita": "presencial",
    "Open House": "presencial",
    "OpenSale": "presencial",
    "Ferias": "presencial",
    "Evento": "presencial",
    "Actividades MKT": "presencial",
    "Actividad Posicionamiento": "presencial",
    "Lanzamiento": "presencial",
    "Vallas": "exterior",
    "Corporación": "otro",
    "usa": "otro",
    "Prensa": "otro",
    "Radio/tv": "otro",
}

# Columnas categóricas que consume el pipeline del modelo predictivo.
COLUMNAS_CATEGORICAS = (
    "sexo",
    "estado_civil",
    "ingreso_familiar",
    "razon_de_compra",
    "canal_de_anuencia",
    "canal_de_contacto",
    "canal_familia",
    "tipo_unidad",
    "opciones_visitadas",
    "participantes_en_credito",
    "lugar_trabajo",
    "lugar_vivienda",
    "campania_tipo",
    "campania_audiencia",
)

# Variables categóricas que en el entrenamiento sólo recibieron clean_text_value.
_SOLO_TEXTO = (
    "razon_de_compra",
    "tipo_unidad",
    "ingreso_familiar",
    "lugar_vivienda",
    "lugar_trabajo",
    "participantes_en_credito",
    "opciones_visitadas",
)


def _texto_limpio(serie: pd.Series) -> pd.Series:
    return serie.map(clean_text_value)


def preparar(dataset: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla de trabajo para los dos modelos.

    Devuelve un DataFrame **nuevo**, con exactamente las mismas filas y en el
    mismo orden que ``dataset`` (índice reseteado a ``RangeIndex``). El
    ``dataset`` de entrada no se modifica en ningún momento.
    """
    df = pd.DataFrame(index=pd.RangeIndex(len(dataset)))

    # ── Limpieza de valores, idéntica a la del entrenamiento ────────────────
    edad = pd.to_numeric(obtener_serie(dataset, "edad"), errors="coerce").to_numpy()
    edad = pd.Series(edad, index=df.index)
    df["edad"] = edad.where(edad.between(EDAD_MIN, EDAD_MAX))

    df["sexo"] = pd.Series(
        obtener_serie(dataset, "sexo").to_numpy(), index=df.index
    ).map(normalize_sex)
    df["estado_civil"] = pd.Series(
        obtener_serie(dataset, "estado_civil").to_numpy(), index=df.index
    ).map(normalize_marital_status)

    for variable in ("canal_de_anuencia", "canal_de_contacto"):
        df[variable] = pd.Series(
            obtener_serie(dataset, variable).to_numpy(), index=df.index
        ).map(clean_channel_value_conservative)

    for variable in _SOLO_TEXTO:
        df[variable] = _texto_limpio(
            pd.Series(obtener_serie(dataset, variable).to_numpy(), index=df.index)
        )

    # ── Variables de campaña ────────────────────────────────────────────────
    campania_tipo = pd.Series(
        obtener_serie(dataset, "campania_tipo").to_numpy(), index=df.index
    )
    campania_audiencia = pd.Series(
        obtener_serie(dataset, "campania_audiencia").to_numpy(), index=df.index
    )
    if campania_tipo.isna().all() or campania_audiencia.isna().all():
        campania = pd.Series(
            obtener_serie(dataset, "campania").to_numpy(), index=df.index
        )
        if campania_tipo.isna().all():
            campania_tipo = campania.map(classify_campania_tipo)
        if campania_audiencia.isna().all():
            campania_audiencia = campania.map(classify_campania_audiencia)
    df["campania_tipo"] = campania_tipo
    df["campania_audiencia"] = campania_audiencia

    # ── Ingeniería de variables del modelo predictivo (§21.4) ───────────────
    df["ingreso_mid"] = df["ingreso_familiar"].map(INCOME_MID)
    df["edad_n"] = df["edad"]
    df["es_inversion"] = (df["razon_de_compra"] == "Inversión").astype(int)
    df["canal_familia"] = df["canal_de_anuencia"].map(CANAL_FAMILIA).fillna("otro")
    df["es_referido"] = (df["canal_familia"] == "referido").astype(int)
    df["edad_x_inversion"] = df["edad_n"] * df["es_inversion"]
    df["edad_x_ingreso"] = df["edad_n"] * df["ingreso_mid"] / 100.0
    df["edad_x_referido"] = df["edad_n"] * df["es_referido"]
    df["mayor_45"] = (df["edad_n"] >= 45).astype(int)
    df["ingreso_x_inversion"] = df["ingreso_mid"] * df["es_inversion"]
    df["ingreso_alto"] = (df["ingreso_mid"] >= 32).astype(int)

    # El pipeline del modelo aprendió sus categorías como texto (el entrenamiento
    # leyó los datos desde CSV). Se homologa el tipo para que un valor que Power BI
    # entregue como número siga cayendo en la categoría correcta —por ejemplo
    # ``participantes_en_credito``—. Los nulos los resuelve el propio
    # SimpleImputer(fill_value='_missing_') del pipeline.
    for columna in COLUMNAS_CATEGORICAS:
        valores = df[columna]
        df[columna] = pd.Series(
            [np.nan if pd.isna(v) else str(v) for v in valores],
            index=df.index,
            dtype=object,
        )

    return df
