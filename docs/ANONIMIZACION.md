[English](en/ANONYMIZATION.md) | **Español** · [Portafolio](../README.es.md)

# Alcance de la anonimización

Esta edición pública utiliza una exportación independiente, sin el historial de trabajo privado.

## Medidas aplicadas

- Sustitución de nombres de empresa, proyectos, campañas, portales comerciales y fuentes internas por etiquetas genéricas.
- Exclusión completa de bases originales, bases procesadas, identificadores y predicciones por registro.
- Exclusión de todos los modelos originales. Los artefactos de demostración se generan desde cero con datos sintéticos y semilla fija.
- Retirada de salidas, adjuntos y metadatos de ejecución de los notebooks.
- Exclusión de figuras originales confidenciales, capturas sin anonimizar del CRM y Power BI, manuales internos y anexos PDF. Se publican dos vistas agregadas del dashboard como PNG renderizados después de retirar la cabecera y el logo de la empresa. Se excluyen la pantalla de predicción, las puntuaciones individuales, los nombres comerciales y los archivos adjuntos originales.
- Eliminación de ejemplos de valores observados del diccionario; conservación de definiciones y porcentajes agregados.
- Sustitución del enriquecimiento con precios, direcciones y catálogos comerciales por una adaptación genérica explícita.
- Exclusión de las correspondencias entre etiquetas genéricas y entidades originales.

## Resultados conservados

Se conservan métricas y cantidades agregadas de la tesis con autorización de la autora. No se publican filas reales bajo seudónimos. La demostración utiliza registros ficticios independientes.

## Diferencias frente a la tesis original

El PDF público es una edición adaptada y lo indica en su portada. Omite material visual y anexos operativos; no sustituye el documento académico original. Los notebooks conservan la lógica de investigación con referencias genéricas y requieren datos privados para reproducir el estudio histórico. No deben interpretarse como un entrenamiento histórico ejecutable con los datos sintéticos de la demostración.
