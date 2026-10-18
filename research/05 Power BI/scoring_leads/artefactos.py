"""Edición pública: utiliza artefactos sintéticos generados en examples/."""

from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path

import joblib

DIRECTORIO_MODELOS = Path(__file__).resolve().parent / "modelos"

RUTA_KMEANS = DIRECTORIO_MODELOS / "modelo_kmeans_R1.joblib"
RUTA_KMEANS_PREP = DIRECTORIO_MODELOS / "preprocesamiento_kmeans_R1.json"
RUTA_CONVERSION = DIRECTORIO_MODELOS / "modelo_conversion_lr.pkl"


def _verificar(ruta: Path) -> Path:
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el artefacto '{ruta.name}' en {DIRECTORIO_MODELOS}. "
            "Copie la carpeta 'modelos' completa junto al paquete scoring_leads."
        )
    return ruta


@lru_cache(maxsize=1)
def cargar_kmeans():
    """Modelo K-Means final (configuración R1, k=6)."""
    return joblib.load(_verificar(RUTA_KMEANS))


@lru_cache(maxsize=1)
def cargar_preprocesamiento_kmeans() -> dict:
    """Estadísticos y categorías congelados del preprocesamiento R1."""
    with _verificar(RUTA_KMEANS_PREP).open(encoding="utf-8") as archivo:
        return json.load(archivo)


@lru_cache(maxsize=1)
def cargar_modelo_conversion() -> dict:
    """Bundle del modelo predictivo final: pipeline + lista de variables."""
    with _verificar(RUTA_CONVERSION).open("rb") as archivo:
        return pickle.load(archivo)
