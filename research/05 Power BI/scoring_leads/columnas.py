"""Edición pública: utiliza artefactos sintéticos generados en examples/."""

from __future__ import annotations

import unicodedata

import pandas as pd

# Para cada variable canónica, los nombres aceptados en orden de preferencia.
# El primer alias presente en la tabla es el que se usa.
ALIAS_COLUMNAS: dict[str, tuple[str, ...]] = {
    "edad": ("edad", "Edad"),
    "sexo": ("sexo", "Sexo", "sexo_label"),
    "estado_civil": ("estado_civil", "Estado Cívil", "Estado Civil", "estado_civil_label"),
    "razon_de_compra": ("razon_de_compra", "Razón de Compra", "Razon de Compra", "razoncompra_label"),
    "tipo_unidad": ("tipo_unidad", "Tipo Unidad", "tipo_de_apartamento", "tipoapa_label"),
    "ingreso_familiar": ("ingreso_familiar", "Ingreso Familiar", "ingfam_label"),
    "canal_de_anuencia": ("canal_de_anuencia", "Canal de Anuencia", "anuencia_label"),
    "canal_de_contacto": ("canal_de_contacto", "Canal de Contacto", "contacto_label"),
    "lugar_trabajo": ("lugar_trabajo", "Lugar Trabajo", "lugtrab_label"),
    "lugar_vivienda": ("lugar_vivienda", "Lugar Vivienda", "lugvivi_label"),
    "opciones_visitadas": ("opciones_visitadas", "Opciones Visitadas", "opciones_visitadas_label"),
    "participantes_en_credito": (
        "participantes_en_credito",
        "Participantes en Crédito",
        "Participantes en Credito",
        "partcredi_label",
    ),
    "campania": ("campania", "Campania", "Campaña", "campaña"),
    # Derivadas: si la tabla ya las trae calculadas se reutilizan tal cual.
    "campania_tipo": ("campania_tipo",),
    "campania_audiencia": ("campania_audiencia",),
}

# Variables sin las cuales el scoring pierde todo su valor informativo.
VARIABLES_CLAVE = (
    "edad",
    "sexo",
    "estado_civil",
    "razon_de_compra",
    "tipo_unidad",
    "ingreso_familiar",
    "canal_de_anuencia",
    "canal_de_contacto",
)


def _clave(nombre: str) -> str:
    """Forma comparable de un nombre de columna: sin acentos, minúsculas y sin separadores."""
    texto = unicodedata.normalize("NFKD", str(nombre).strip().lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return "".join(c for c in texto if c.isalnum())


def construir_indice(dataset: pd.DataFrame) -> dict[str, str]:
    """Mapea nombre comparable -> nombre real de la columna en la tabla."""
    indice: dict[str, str] = {}
    for columna in dataset.columns:
        indice.setdefault(_clave(columna), columna)
    return indice


def obtener_serie(dataset: pd.DataFrame, variable: str) -> pd.Series:
    """Devuelve la columna de ``dataset`` que corresponde a ``variable``.

    Si ninguno de los alias está presente devuelve una Serie de nulos con el
    mismo índice, para que la inferencia siga funcionando sin perder filas.
    """
    indice = construir_indice(dataset)
    for alias in ALIAS_COLUMNAS.get(variable, (variable,)):
        columna = indice.get(_clave(alias))
        if columna is not None:
            return dataset[columna]
    return pd.Series(pd.NA, index=dataset.index, dtype="object", name=variable)


def variables_ausentes(dataset: pd.DataFrame) -> list[str]:
    """Lista las variables clave que no se encontraron bajo ningún alias."""
    indice = construir_indice(dataset)
    faltantes = []
    for variable in VARIABLES_CLAVE:
        alias = ALIAS_COLUMNAS.get(variable, (variable,))
        if not any(_clave(a) in indice for a in alias):
            faltantes.append(variable)
    return faltantes
