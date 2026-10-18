"""Adaptación pública de ingeniería de variables.

Se retiraron catálogos, precios, direcciones y reglas particulares de la empresa.
Las transformaciones genéricas se muestran en el paquete de scoring y notebooks.
Este archivo no pretende reproducir el enriquecimiento comercial privado.
"""
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "05 Power BI"))
from scoring_leads.preparacion import preparar

def main():
    folder=Path(__file__).resolve().parents[1]
    source=folder / "data/processed/leads_cleaned_final.csv"
    if not source.exists():
        raise FileNotFoundError("La base histórica es privada y no forma parte del portafolio.")
    tabla=pd.read_csv(source,low_memory=False)
    features=preparar(tabla)
    output=folder / "data/features"
    output.mkdir(parents=True,exist_ok=True)
    features.to_csv(output / "leads_enriched_final.csv",index=False)

if __name__ == "__main__":
    main()
