"""Preparación de datos y utilidades compartidas para la exploración exhaustiva.

Universo: leads calificados con fuente, canal de anuencia/contacto, sexo,
razón de compra, estado civil, ingreso familiar, edad, tiene hijos y núcleo
familiar completos (13,038 leads / 563 conversiones).

Variables excluidas por fuga confirmada por el usuario:
  sector_referidor(+__miss), campania_plataforma(+__miss),
  tipo_unidad__miss, opciones_visitadas__miss.
"""

from pathlib import Path
import os, sys

# Find the repository independently of the computer or clone location.
REPO_ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                  if (p / ".tesis-portafolio").is_file()), None)
if REPO_ROOT is None:
    raise RuntimeError("Abre Jupyter desde la carpeta de este repositorio.")

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
DATA = Path(f'{REPO_ROOT}/01 Limpieza Código/'
            'data/features/leads_enriched_qualified_final.csv')

UNIVERSE_VARS = ['fuente_pt2', 'canal_de_anuencia', 'canal_de_contacto', 'sexo',
                 'razon_de_compra', 'estado_civil', 'ingreso_familiar', 'edad',
                 'tiene_hijos', 'nucleo_familiar']

INCOME_MID = {'Menos De Q10k': 8, 'Q 10k A Q13.9k': 12, 'Q 14k A Q17.9k': 16,
              'Q18k A Q21.9k': 20, 'Q22k A Q25.9k': 24, 'Q26k A Q29.9k': 28,
              'Q30k A 33.9k': 32, 'Q34k A 37.9k': 36, 'Q38k A 41.9k': 40,
              'Q42k A 45.9k': 44, '46k A 49.9k': 48, 'Q50k A Más': 55}

# Agrupación comercial de canal de anuencia (de §16/§17: referencia >> pauta)
CANAL_FAMILIA = {
    'Facebook': 'pauta_digital', 'Instagram': 'pauta_digital',
    'Fanpage Líderes': 'pauta_digital', 'portal_demo': 'portal',
    'Página Web': 'portal', 'Inbox': 'pauta_digital', 'WhatsApp': 'directo',
    'Llamada': 'directo', 'Referencia': 'referido', 'Friends/Family': 'referido',
    'Shareholders Sale': 'referido', 'Sede': 'presencial',
    'ExpoCasa': 'presencial', 'ExpoCasa Virtual': 'presencial',
    'Visita Inédita': 'presencial', 'Open House': 'presencial',
    'OpenSale': 'presencial', 'Ferias': 'presencial', 'Evento': 'presencial',
    'Actividades MKT': 'presencial', 'Actividad Posicionamiento': 'presencial',
    'Lanzamiento': 'presencial', 'Vallas': 'exterior', 'Corporación': 'otro',
    'usa': 'otro', 'Prensa': 'otro', 'Radio/tv': 'otro',
}


def _clean_nucleo(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if '+' in s or 'De' in s:
        return 6.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_universe():
    df = pd.read_csv(DATA, low_memory=False)
    mask = df[UNIVERSE_VARS].notna().all(axis=1)
    d = df.loc[mask].copy().reset_index(drop=True)

    # ── derivadas numéricas ──
    d['edad_n'] = pd.to_numeric(d['edad'], errors='coerce')
    d['ingreso_mid'] = d['ingreso_familiar'].map(INCOME_MID)
    d['nucleo_n'] = d['nucleo_familiar'].apply(_clean_nucleo)
    d['ingreso_per_capita'] = d['ingreso_mid'] / d['nucleo_n'].clip(lower=1)
    d['tiene_hijos_bin'] = (d['tiene_hijos'].astype(str).str.strip().str.lower()
                            .isin(['si', 'sí', 'yes', '1', 'true'])).astype(int)

    # ── flags e interacciones comerciales ──
    d['es_inversion'] = (d['razon_de_compra'] == 'Inversión').astype(int)
    d['canal_familia'] = d['canal_de_anuencia'].map(CANAL_FAMILIA).fillna('otro')
    d['es_referido'] = (d['canal_familia'] == 'referido').astype(int)
    d['edad_x_inversion'] = d['edad_n'] * d['es_inversion']
    d['ingreso_x_inversion'] = d['ingreso_mid'] * d['es_inversion']
    d['edad_x_ingreso'] = d['edad_n'] * d['ingreso_mid'] / 100.0
    d['edad_x_referido'] = d['edad_n'] * d['es_referido']
    d['etapa_familiar'] = (d['estado_civil'].astype(str) + '|' +
                           np.where(d['tiene_hijos_bin'] == 1, 'hijos', 'sin_hijos'))
    d['mayor_45'] = (d['edad_n'] >= 45).astype(int)
    d['ingreso_alto'] = (d['ingreso_mid'] >= 32).astype(int)  # >= Q30k

    return d


# ── pools de variables ────────────────────────────────────────────────────
# Base limpia (valores, sin __miss con fuga; sin identidad de proyecto)
POOL_CLEAN = ['edad_n', 'sexo', 'estado_civil', 'tiene_hijos_bin', 'nucleo_n',
              'ingreso_mid', 'ingreso_familiar', 'razon_de_compra',
              'canal_de_anuencia', 'canal_de_contacto', 'canal_familia',
              'tipo_unidad', 'opciones_visitadas', 'participantes_en_credito',
              'lugar_trabajo', 'lugar_vivienda', 'campania_tipo',
              'campania_audiencia']
# + identidad de proyecto/fuente (no-fuga temporal, pero no generaliza a proyectos nuevos)
POOL_PROJ = POOL_CLEAN + ['fuente_pt2', 'proyecto']
# + derivadas e interacciones
POOL_ENG = POOL_CLEAN + ['ingreso_per_capita', 'es_inversion', 'es_referido',
                         'edad_x_inversion', 'ingreso_x_inversion',
                         'edad_x_ingreso', 'edad_x_referido', 'etapa_familiar',
                         'mayor_45', 'ingreso_alto']
POOL_ENG_PROJ = POOL_ENG + ['fuente_pt2', 'proyecto']
# mínimos
POOL_DEMO = ['edad_n', 'sexo', 'estado_civil', 'tiene_hijos_bin', 'nucleo_n',
             'ingreso_mid', 'razon_de_compra']
POOL_EDAD = ['edad_n']

POOLS = {
    'P0_SoloEdad': POOL_EDAD,
    'P1_Demo': POOL_DEMO,
    'P2_Clean': POOL_CLEAN,
    'P3_Clean+Proj': POOL_PROJ,
    'P4_Eng': POOL_ENG,
    'P5_Eng+Proj': POOL_ENG_PROJ,
}


def oof_metrics(y, prob, target_recall=0.70):
    """Métricas a partir de predicciones out-of-fold."""
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 precision_recall_curve)
    out = {'pr_auc': average_precision_score(y, prob),
           'roc_auc': roc_auc_score(y, prob)}
    n = len(y)
    order = np.argsort(prob)[::-1]
    cum = np.cumsum(np.asarray(y)[order])
    for k in (0.10, 0.20):
        tn = max(1, int(np.ceil(n * k)))
        tp = cum[tn - 1]
        out[f'lift_{int(k*100)}'] = (tp / tn) / (y.mean())
        out[f'rec_{int(k*100)}'] = tp / y.sum()
    prc, rcc, thr = precision_recall_curve(y, prob)
    idx = np.where(rcc[:-1] >= target_recall)[0]
    if len(idx):
        i = idx[-1]
        sel = prob >= thr[i]
        out['pct_leads_r70'] = sel.mean() * 100
        out['prec_r70'] = prc[i]
    else:
        out['pct_leads_r70'] = np.nan
        out['prec_r70'] = np.nan
    return out
