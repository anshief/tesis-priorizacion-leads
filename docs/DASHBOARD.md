[English](en/DASHBOARD.md) | **Español** · [← Portafolio](../README.es.md)

# Dentro del dashboard

Dos vistas de la interfaz analítica de Power BI, seleccionadas para este portafolio público. Ambas muestran resultados históricos agregados. Se retiró la cabecera con el logo de la empresa de las imágenes renderizadas; los valores de los gráficos se conservaron.

## 01 / El embudo comercial

Desde el ingreso hasta la conversión, el embudo permite ver el tamaño de la cartera y la caída entre etapas. Los porcentajes de progresión corresponden a etapas sucesivas; la tasa de conversión global utiliza toda la cartera. La tabla de razones de pérdida muestra categorías agregadas, no notas individuales de clientes.

[![Embudo comercial, progresión entre etapas y razones de pérdida agregadas](assets/dashboard-funnel.png)](assets/dashboard-funnel.png)

Los valores abreviados con K representan miles redondeados.

## 02 / Exploración demográfica

La segunda vista compara distribuciones y tasas de conversión agregadas según los atributos demográficos registrados. Documenta el análisis exploratorio, sin registros de clientes ni predicciones individuales.

[![Distribuciones agregadas por sexo, edad, estado civil, hijos y tamaño del núcleo familiar](assets/dashboard-demographics.png)](assets/dashboard-demographics.png)

**Alcance:** estos gráficos utilizan subconjuntos de registros completos. La disponibilidad varía según el campo, por lo que pueden tener distintos denominadores; sus cantidades no deben sumarse ni interpretarse como la totalidad del CRM. Las asociaciones observadas no demuestran causalidad ni resultados futuros individuales.

## Qué muestra esta galería

El dashboard conecta la exploración descriptiva con el flujo de la tesis. La integración de scoring descrita en el [caso de estudio](CASO_DE_ESTUDIO.md) agrega un perfil de cliente (`cluster`) y una prioridad relativa (`percentil`). La galería pública omite la pantalla de predicción porque contiene identificadores y puntuaciones por registro.

Son vistas históricas estáticas, no un dashboard en vivo ni evidencia de implementación en producción. La presentación se usó como contexto de investigación; no se distribuyen el archivo original, el PDF de capturas, sus recursos incrustados ni sus notas. La [demostración sintética](REPRODUCIBILIDAD.md) permite explorar el contrato de scoring sin datos reales de clientes.
