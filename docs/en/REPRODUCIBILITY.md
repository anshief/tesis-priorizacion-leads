[← Portfolio](../../README.md) · **English** | [Español](../REPRODUCIBILIDAD.md)

# Run the synthetic demonstration

Use Python 3.11. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r "05 Power BI/requirements.txt"
python examples/entrenar_demo_sintetica.py
python examples/demo_scoring.py
python "05 Power BI/pruebas/test_contrato_dataset.py"
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The first script creates fictional data with a fixed seed and trains K-Means and logistic regression. It saves artifacts under `05 Power BI/scoring_leads/modelos/`. The demonstration scores five fictional records, and the checks verify that the original table is preserved while `cluster` and `percentil` are added.

Demonstration models are not the thesis models, and their cluster labels do not represent the six historical profiles. They must not be used for sales decisions.

To explore the notebooks, run `python -m pip install -r requirements.txt` and `jupyter lab`. Research notebooks require private databases and artifacts that are not included. Business names are generalized, and enrichment with private catalogs has been removed.

Compile the public thesis from `tesis/latex/` with `latexmk -pdf -interaction=nonstopmode main.tex`. It uses no original images or confidential appendices. The PDF and source commentary remain in Spanish.
