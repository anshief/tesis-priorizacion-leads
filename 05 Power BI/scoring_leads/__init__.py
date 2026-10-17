"""Edición pública: utiliza artefactos sintéticos generados en examples/."""

from __future__ import annotations

import pandas as pd

from .clustering import CLUSTER_SIN_SEGMENTO, asignar_cluster
from .columnas import variables_ausentes
from .prediccion import calcular_percentil, probabilidad_conversion
from .preparacion import preparar

__all__ = [
    "agregar_cluster_y_percentil",
    "variables_ausentes",
    "CLUSTER_SIN_SEGMENTO",
    "COLUMNA_CLUSTER",
    "COLUMNA_PERCENTIL",
]

COLUMNA_CLUSTER = "cluster"
COLUMNA_PERCENTIL = "percentil"


def agregar_cluster_y_percentil(
    dataset: pd.DataFrame,
    decimales_percentil: int = 2,
    asignar_cluster_fuera_de_universo: bool = False,
) -> pd.DataFrame:
    """Devuelve ``dataset`` con las columnas ``cluster`` y ``percentil`` añadidas.

    Garantías, verificadas en tiempo de ejecución antes de devolver el resultado:

    * mismas filas, en el mismo orden, sin altas ni bajas;
    * mismas columnas originales, con el mismo contenido y la misma posición;
    * las dos columnas nuevas se añaden al final y no hay ninguna más;
    * la probabilidad de conversión nunca se expone como columna.

    Parameters
    ----------
    dataset:
        Tabla del CRM tal como la entrega Power BI.
    decimales_percentil:
        Redondeo del percentil. Por defecto 2.
    asignar_cluster_fuera_de_universo:
        Por defecto ``False``: los leads sin las cinco variables demográficas
        completas reciben ``cluster = -1`` ("sin segmento"). Poner ``True``
        fuerza a asignarles el centroide más cercano. Ver
        :func:`scoring_leads.clustering.asignar_cluster`.
    """
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("Se esperaba un DataFrame de pandas en 'dataset'.")

    columnas_originales = list(dataset.columns)
    n_filas = len(dataset)

    for nombre in (COLUMNA_CLUSTER, COLUMNA_PERCENTIL):
        if nombre in columnas_originales:
            raise ValueError(
                f"La tabla ya trae una columna llamada '{nombre}'. "
                "Elimínela en Power Query antes de ejecutar el script."
            )

    # Copia de trabajo: la tabla que entrega Power BI no se muta.
    resultado = dataset.copy()

    if n_filas == 0:
        resultado[COLUMNA_CLUSTER] = pd.Series(dtype="int64")
        resultado[COLUMNA_PERCENTIL] = pd.Series(dtype="float64")
        return resultado

    preparado = preparar(dataset)

    clusters = asignar_cluster(
        preparado, asignar_fuera_de_universo=asignar_cluster_fuera_de_universo
    )
    probabilidades = probabilidad_conversion(preparado)
    percentiles = calcular_percentil(probabilidades, decimales=decimales_percentil)

    # Se asignan por posición para no depender del índice que traiga Power BI.
    resultado[COLUMNA_CLUSTER] = pd.Series(clusters, index=resultado.index)
    resultado[COLUMNA_PERCENTIL] = pd.Series(
        percentiles.to_numpy(), index=resultado.index
    )

    _validar_contrato(dataset, resultado, columnas_originales, n_filas)
    return resultado


def _validar_contrato(original, resultado, columnas_originales, n_filas) -> None:
    """Comprueba que sólo se agregaron las dos columnas acordadas."""
    if len(resultado) != n_filas:
        raise AssertionError(
            f"El número de filas cambió: {n_filas} -> {len(resultado)}."
        )

    esperado = columnas_originales + [COLUMNA_CLUSTER, COLUMNA_PERCENTIL]
    if list(resultado.columns) != esperado:
        raise AssertionError(
            "Las columnas del resultado no coinciden con las esperadas.\n"
            f"  esperadas: {esperado}\n  obtenidas: {list(resultado.columns)}"
        )

    for columna in columnas_originales:
        izquierda = original[columna].to_numpy()
        derecha = resultado[columna].to_numpy()
        iguales = (izquierda == derecha) | (
            pd.isna(pd.Series(izquierda)) & pd.isna(pd.Series(derecha))
        ).to_numpy()
        if not iguales.all():
            raise AssertionError(f"La columna original '{columna}' fue modificada.")

    if resultado[COLUMNA_CLUSTER].isna().any():
        raise AssertionError("Se generaron clusters nulos.")
    if resultado[COLUMNA_PERCENTIL].isna().any():
        raise AssertionError("Se generaron percentiles nulos.")
