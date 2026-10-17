"""Pruebas del contrato con Power BI.

Ejecución:
    python -m pytest "pruebas/test_contrato_dataset.py" -v
o sin pytest:
    python "pruebas/test_contrato_dataset.py"

Verifica que `agregar_cluster_y_percentil` conserva la tabla intacta y sólo
agrega `cluster` y `percentil`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scoring_leads import (  # noqa: E402
    CLUSTER_SIN_SEGMENTO,
    COLUMNA_CLUSTER,
    COLUMNA_PERCENTIL,
    agregar_cluster_y_percentil,
)
from scoring_leads.clustering import mascara_universo  # noqa: E402
from scoring_leads.prediccion import (  # noqa: E402
    calcular_percentil,
    probabilidad_conversion,
)
from scoring_leads.preparacion import preparar  # noqa: E402


def tabla_de_ejemplo() -> pd.DataFrame:
    """Tabla que imita una exportación del CRM, con casos borde deliberados."""
    return pd.DataFrame(
        {
            "id_trato": [101, 102, 103, 104, 105, 106, 107],
            "proyecto": ["proyecto01", "proyecto05", "proyecto03", "proyecto04", "proyecto01", "proyecto05", None],
            # fila 3: edad fuera del rango válido; fila 5: edad ausente
            "edad": [34, 52, 27, 150, 41, None, 63],
            "sexo": ["Femenino", "Masculino", None, "MASCULINO", "F", "Otro", "Femenino"],
            "estado_civil": ["Casad@", "Solter@", "Futuros Conyuges", None, "casado", "X", "Solter@"],
            "razon_de_compra": ["Vivienda", "Inversión", None, "Vivienda", "INVERSIÓN", "Otra Razón", None],
            "tipo_unidad": ["2 Habs", "3 Habs", "1 Hab", None, "5 Habs", "Bodega", "2 Habs"],
            "ingreso_familiar": [
                "Q30k A 33.9k", "Q50k A Más", None, "Q22k A Q25.9k",
                "Rango Inventado", "Menos De Q10k", "Q18k A Q21.9k",
            ],
            "canal_de_anuencia": [
                "Facebook", "Referencia", "portal_demo", "Canal Nuevo 2026",
                None, "Sede", "Instagram",
            ],
            "canal_de_contacto": [
                "Facebook", "Referencia", "Página Web", "WhatsApp",
                "Visita Inédita", None, "Instagram",
            ],
            "lugar_trabajo": ["Zona 10", None, "Mixco", "Marte", "Zona 9", "Usa", "Zona 4"],
            "lugar_vivienda": ["Zona 15", "Mixco", None, "Zona 11", "Júpiter", "Ces", "Zona 1"],
            "opciones_visitadas": ["Ninguna", "proyecto_externo_demo", None, "Proyecto X", "proyecto_externo_demo", "Ninguna", "proyecto_externo_demo"],
            # enteros y texto mezclados a propósito
            "participantes_en_credito": [1, 2, None, 3, "+ De 4", 1, 2],
            "campania": [
                "proyecto01 PRINCIPAL MAYO 25", "proyecto04 LEADS EEUU LA", None,
                "proyecto05 MESSENGER", "proyecto03 FOOTER USA CA", "portal_demo EXPOCASA", "Campaña Rara",
            ],
            "importe_estimado": [850000.0, 1200000.0, None, 640000.0, 990000.0, 720000.0, 1100000.0],
        }
    )


def _ejecutar():
    entrada = tabla_de_ejemplo()
    copia_previa = entrada.copy(deep=True)
    salida = agregar_cluster_y_percentil(entrada)
    return entrada, copia_previa, salida


def test_filas_intactas():
    entrada, _, salida = _ejecutar()
    assert len(salida) == len(entrada), "cambió la cantidad de filas"
    assert list(salida.index) == list(entrada.index), "cambió el índice o el orden"
    assert salida["id_trato"].tolist() == entrada["id_trato"].tolist(), "se reordenaron los leads"


def test_solo_dos_columnas_nuevas():
    entrada, _, salida = _ejecutar()
    nuevas = [c for c in salida.columns if c not in entrada.columns]
    assert nuevas == [COLUMNA_CLUSTER, COLUMNA_PERCENTIL], f"columnas nuevas inesperadas: {nuevas}"
    assert list(salida.columns) == list(entrada.columns) + nuevas, "cambió el orden de columnas"


def test_columnas_originales_sin_modificar():
    _, copia_previa, salida = _ejecutar()
    for columna in copia_previa.columns:
        izq, der = copia_previa[columna], salida[columna]
        assert izq.equals(der), f"la columna original '{columna}' cambió"


def test_dataset_de_entrada_no_se_muta():
    entrada, copia_previa, _ = _ejecutar()
    assert list(entrada.columns) == list(copia_previa.columns), "se mutó el dataset de entrada"
    assert entrada.equals(copia_previa), "se mutó el contenido del dataset de entrada"


def test_no_se_expone_la_probabilidad():
    _, _, salida = _ejecutar()
    sospechosas = [c for c in salida.columns if "prob" in str(c).lower() or "score" in str(c).lower()]
    assert not sospechosas, f"columnas de probabilidad expuestas: {sospechosas}"


def test_cluster_valido_para_todas_las_filas():
    _, _, salida = _ejecutar()
    assert salida[COLUMNA_CLUSTER].notna().all(), "hay clusters nulos"
    validos = set(range(6)) | {CLUSTER_SIN_SEGMENTO}
    fuera = set(salida[COLUMNA_CLUSTER].unique()) - validos
    assert not fuera, f"clusters con valores inesperados: {fuera}"


def test_cluster_respeta_el_universo_del_modelo():
    """Sólo los leads con las 5 variables demográficas completas reciben segmento."""
    entrada, _, salida = _ejecutar()
    preparado = preparar(entrada)
    dentro = mascara_universo(preparado)

    asignados = salida[COLUMNA_CLUSTER].to_numpy()
    assert np.all(asignados[dentro] >= 0), "un lead del universo quedó sin segmento"
    assert np.all(
        asignados[~dentro] == CLUSTER_SIN_SEGMENTO
    ), "se asignó segmento a un lead fuera del universo"


def test_lead_sin_informacion_no_recibe_segmento():
    """Un lead sin ningún dato demográfico no debe caer en un cluster real."""
    vacio = pd.DataFrame(
        {
            "id_trato": [1],
            "edad": [np.nan],
            "sexo": [None],
            "estado_civil": [None],
            "razon_de_compra": [None],
            "tipo_unidad": [None],
        }
    )
    salida = agregar_cluster_y_percentil(vacio)
    assert salida[COLUMNA_CLUSTER].iloc[0] == CLUSTER_SIN_SEGMENTO

    # Con la opción explícita sí se le asigna el centroide más cercano.
    forzado = agregar_cluster_y_percentil(vacio, asignar_cluster_fuera_de_universo=True)
    assert 0 <= forzado[COLUMNA_CLUSTER].iloc[0] <= 5


def test_percentil_en_rango_y_monotono():
    entrada, _, salida = _ejecutar()
    percentil = salida[COLUMNA_PERCENTIL]
    assert percentil.notna().all(), "hay percentiles nulos"
    assert percentil.between(0, 100).all(), "hay percentiles fuera de 0..100"

    # Mayor probabilidad debe implicar mayor percentil.
    probabilidad = probabilidad_conversion(preparar(entrada))
    esperado = calcular_percentil(probabilidad)
    assert np.allclose(percentil.to_numpy(), esperado.to_numpy()), "percentil no coincide con el ranking"
    orden_prob = np.argsort(probabilidad.to_numpy(), kind="stable")
    assert np.all(np.diff(percentil.to_numpy()[orden_prob]) >= 0), "el percentil no es monótono"


def test_tabla_vacia():
    vacia = tabla_de_ejemplo().iloc[0:0]
    salida = agregar_cluster_y_percentil(vacia)
    assert len(salida) == 0
    assert list(salida.columns) == list(vacia.columns) + [COLUMNA_CLUSTER, COLUMNA_PERCENTIL]


def test_columnas_faltantes_no_pierden_filas():
    minima = tabla_de_ejemplo()[["id_trato", "edad", "sexo"]].copy()
    salida = agregar_cluster_y_percentil(minima)
    assert len(salida) == len(minima)
    assert salida[COLUMNA_CLUSTER].notna().all()
    assert salida[COLUMNA_PERCENTIL].notna().all()


def test_alias_de_columnas_del_crm():
    """La exportación cruda del CRM (columnas *_label) debe dar el mismo resultado."""
    canonica = tabla_de_ejemplo()
    cruda = canonica.rename(
        columns={
            "sexo": "sexo_label",
            "estado_civil": "estado_civil_label",
            "razon_de_compra": "razoncompra_label",
            "tipo_unidad": "tipoapa_label",
            "ingreso_familiar": "ingfam_label",
            "canal_de_anuencia": "anuencia_label",
            "canal_de_contacto": "contacto_label",
            "lugar_trabajo": "lugtrab_label",
            "lugar_vivienda": "lugvivi_label",
            "opciones_visitadas": "opciones_visitadas_label",
            "participantes_en_credito": "partcredi_label",
        }
    )
    salida_canonica = agregar_cluster_y_percentil(canonica)
    salida_cruda = agregar_cluster_y_percentil(cruda)
    assert salida_canonica[COLUMNA_CLUSTER].tolist() == salida_cruda[COLUMNA_CLUSTER].tolist()
    assert salida_canonica[COLUMNA_PERCENTIL].tolist() == salida_cruda[COLUMNA_PERCENTIL].tolist()


if __name__ == "__main__":
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fallos = 0
    for prueba in pruebas:
        try:
            prueba()
            print(f"  OK    {prueba.__name__}")
        except AssertionError as error:
            fallos += 1
            print(f"  FALLA {prueba.__name__}: {error}")
    print(f"\n{len(pruebas) - fallos}/{len(pruebas)} pruebas superadas")
    sys.exit(1 if fallos else 0)
