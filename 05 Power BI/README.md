# Integración de scoring: edición pública

El paquete conserva la interfaz de integración desarrollada para Power BI: recibe una tabla y agrega `cluster` y `percentil` sin modificar las columnas originales.

**En esta edición, los modelos son sintéticos. No se distribuyen los modelos entrenados con datos del CRM.** Antes de ejecutar, genera los artefactos siguiendo la [guía de reproducción](../docs/REPRODUCIBILIDAD.md).

```python
from scoring_leads import agregar_cluster_y_percentil
salida = agregar_cluster_y_percentil(dataset)
```

`cluster` identifica uno de seis grupos de demostración; `-1` indica información insuficiente para segmentar. `percentil` es la posición relativa de la probabilidad dentro del lote; no es un porcentaje de compra. Las pruebas usan exclusivamente registros ficticios.

Los módulos incluyen reconocimiento de columnas, normalización, preparación, clustering, predicción y validación del contrato de tabla. Los nombres de empresa y proyectos fueron sustituidos por etiquetas genéricas.
