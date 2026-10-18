"""Edición pública: utiliza artefactos sintéticos generados en examples/."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .artefactos import cargar_kmeans, cargar_preprocesamiento_kmeans

__all__ = [
    "CLUSTER_SIN_SEGMENTO",
    "asignar_cluster",
    "construir_matriz",
    "mascara_universo",
]


def construir_matriz(df: pd.DataFrame) -> np.ndarray:
    """Matriz R1 para inferencia, con manejo explícito de nulos y categorías nuevas.

    * ``edad`` nula o fuera de rango -> se estandariza a 0, es decir la edad
      media del entrenamiento: es el valor neutro de ese bloque.
    * categoría nula o no vista en entrenamiento -> bloque one-hot en ceros,
      exactamente lo que hace ``OneHotEncoder(handle_unknown='ignore')``.
    """
    prep = cargar_preprocesamiento_kmeans()
    media = prep["edad_scaler"]["mean"]
    escala = prep["edad_scaler"]["scale"]
    n_filas = len(df)

    edad = pd.to_numeric(df.get("edad"), errors="coerce").to_numpy(dtype=float)
    z_edad = np.where(np.isnan(edad), 0.0, (edad - media) / escala)
    bloques = [(z_edad * prep["peso_edad"]).reshape(-1, 1)]

    for variable in prep["orden_bloques"][1:]:
        categorias = prep["categorias"][variable]
        divisor = np.sqrt(prep["n_categorias_entrenamiento"][variable])
        peso = prep["pesos_categoricas"][variable]
        posicion = {categoria: i for i, categoria in enumerate(categorias)}

        bloque = np.zeros((n_filas, len(categorias)), dtype=float)
        valores = df[variable] if variable in df.columns else pd.Series([np.nan] * n_filas)
        for fila, valor in enumerate(valores.to_numpy()):
            if valor is None or (isinstance(valor, float) and np.isnan(valor)):
                continue
            columna = posicion.get(str(valor))
            if columna is not None:
                bloque[fila, columna] = 1.0
        bloques.append(bloque / divisor * peso)

    return np.hstack(bloques)


# Valor que reciben los leads que no pertenecen al universo del modelo.
CLUSTER_SIN_SEGMENTO = -1


def mascara_universo(df: pd.DataFrame) -> np.ndarray:
    """``True`` en los leads que sí pertenecen al universo del modelo.

    El universo de clustering se definió con las cinco variables demográficas
    completas: ``edad``, ``sexo``, ``estado_civil``, ``razon_de_compra`` y
    ``tipo_unidad``. Un lead al que le falta alguna no tiene un segmento
    interpretable: su vector se acerca al origen y el centroide más cercano
    queda determinado por el azar, no por su perfil.
    """
    prep = cargar_preprocesamiento_kmeans()
    variables = prep["orden_bloques"]
    dentro = np.ones(len(df), dtype=bool)
    for variable in variables:
        if variable not in df.columns:
            return np.zeros(len(df), dtype=bool)
        dentro &= df[variable].notna().to_numpy()
    return dentro


def asignar_cluster(df: pd.DataFrame, asignar_fuera_de_universo: bool = False) -> np.ndarray:
    """Cluster de cada fila de ``df``, en el mismo orden recibido.

    Parameters
    ----------
    asignar_fuera_de_universo:
        ``False`` (por defecto): los leads sin las cinco variables demográficas
        completas reciben ``CLUSTER_SIN_SEGMENTO`` (-1). Así el segmento se lee
        como "sin información suficiente" y no contamina los seis clusters con
        leads a los que el modelo no puede perfilar.

        ``True``: se les asigna igualmente el centroide más cercano, tratando
        los faltantes como valores neutros (edad media, bloque one-hot en
        ceros). Útil sólo si el tablero necesita que todo lead tenga un
        segmento numérico entre 0 y 5.
    """
    modelo = cargar_kmeans()
    matriz = construir_matriz(df)
    if matriz.shape[1] != modelo.n_features_in_:
        raise ValueError(
            f"La matriz de clustering tiene {matriz.shape[1]} columnas y el modelo "
            f"espera {modelo.n_features_in_}."
        )

    clusters = modelo.predict(matriz).astype("int64")
    if not asignar_fuera_de_universo:
        clusters = np.where(mascara_universo(df), clusters, CLUSTER_SIN_SEGMENTO)
    return clusters.astype("int64")
