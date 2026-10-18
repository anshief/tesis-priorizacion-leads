# =============================================================================
#  Scoring de leads para Power BI
#  -----------------------------------------------------------------------------
#  Power BI entrega la tabla del CRM en el DataFrame `dataset`.
#  Este script le agrega dos columnas y devuelve la MISMA tabla:
#
#     dataset["cluster"]    segmento del modelo K-Means final (R1, k=6)
#     dataset["percentil"]  posición relativa 0-100 del lead según la
#                           probabilidad del modelo predictivo final
#
#  No se conecta al CRM, no escribe nada, no reentrena, no filtra ni reordena
#  filas y no modifica ninguna columna original.
#
#  Copie y pegue este bloque completo en:
#     Power BI Desktop -> Transformar datos -> Ejecutar script de Python
# =============================================================================

import sys

# -----------------------------------------------------------------------------
# Ubicación del paquete `scoring_leads`.
#
# Deje la cadena vacía si instaló el paquete con pip (opción recomendada, ver el
# README). Si prefiere no instalar nada, escriba aquí la ruta de la carpeta que
# contiene `scoring_leads`, por ejemplo:  CARPETA_SCORING = r"C:\Modelos\PowerBI"
# -----------------------------------------------------------------------------
CARPETA_SCORING = ""

if CARPETA_SCORING and CARPETA_SCORING not in sys.path:
    sys.path.insert(0, CARPETA_SCORING)

from scoring_leads import agregar_cluster_y_percentil

# -----------------------------------------------------------------------------
# Única transformación: la tabla sigue llamándose `dataset` y sólo gana las dos
# columnas nuevas. La probabilidad de conversión se usa internamente para el
# ranking y nunca se expone.
# -----------------------------------------------------------------------------
dataset = agregar_cluster_y_percentil(dataset)
