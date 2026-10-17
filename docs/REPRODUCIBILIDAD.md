# Ejecutar la demostración sintética

Usa Python 3.11. Desde la raíz del repositorio:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r "05 Power BI/requirements.txt"
python examples/entrenar_demo_sintetica.py
python examples/demo_scoring.py
python "05 Power BI/pruebas/test_contrato_dataset.py"
```

En Windows, activa el entorno con `.venv\Scripts\activate`.

El primer script genera datos inventados con una semilla fija y entrena K-Means y regresión logística. Guarda los artefactos en `05 Power BI/scoring_leads/modelos/`. La demostración puntúa cinco registros ficticios y las pruebas verifican que se conserva la tabla original y se agregan `cluster` y `percentil`.

Los modelos de demostración no son los de la tesis y sus etiquetas de cluster no representan los seis perfiles históricos. No deben usarse para decisiones comerciales.

Para explorar los notebooks: `python -m pip install -r requirements.txt` y `jupyter lab`. Los notebooks de investigación requieren bases y artefactos privados que no se incluyen. Los nombres comerciales están generalizados y el enriquecimiento con catálogos privados se retiró.

La tesis pública se compila desde `tesis/latex/` con `latexmk -pdf -interaction=nonstopmode main.tex`. No utiliza imágenes ni anexos originales.
