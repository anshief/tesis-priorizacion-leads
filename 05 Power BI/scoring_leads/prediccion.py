"""Edición pública: utiliza artefactos sintéticos generados en examples/."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .artefactos import cargar_modelo_conversion


def variables_requeridas() -> list[str]:
    """Las 24 variables de entrada del modelo, en su orden original."""
    return list(cargar_modelo_conversion()["features"])


def probabilidad_conversion(df: pd.DataFrame) -> pd.Series:
    """Probabilidad de conversión de cada fila, en el orden recibido."""
    bundle = cargar_modelo_conversion()
    pipeline = bundle["pipeline"]
    features = list(bundle["features"])

    entrada = pd.DataFrame(index=pd.RangeIndex(len(df)))
    for variable in features:
        entrada[variable] = (
            df[variable].to_numpy() if variable in df.columns else np.nan
        )

    probabilidades = pipeline.predict_proba(entrada)[:, 1]
    return pd.Series(probabilidades, index=pd.RangeIndex(len(df)), name="probabilidad")


def calcular_percentil(probabilidades: pd.Series, decimales: int = 2) -> pd.Series:
    """Posición relativa del lead, de 0 a 100.

    Ranking porcentual sobre la probabilidad: mayor probabilidad -> mayor
    percentil. Los empates reciben el promedio de sus posiciones.
    """
    return (probabilidades.rank(method="average", pct=True) * 100).round(decimales)
