"""Demostración pública: todos los registros son ficticios."""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/05 Power BI"))
from scoring_leads import agregar_cluster_y_percentil

def main():
    entrada = pd.DataFrame({
        "id_ejemplo": ["FICTICIO-01", "FICTICIO-02", "FICTICIO-03", "FICTICIO-04", "FICTICIO-05"],
        "edad": [28, 42, 55, 36, None],
        "sexo": ["Femenino", "Masculino", "Femenino", "Masculino", None],
        "estado_civil": ["Soltero", "Casado", "Casada", "Soltero", None],
        "razon_de_compra": ["Vivienda", "Inversión", "Vivienda", "Inversión", None],
        "tipo_unidad": ["2 Habs", "3 Habs", "2 Habs", "1 Hab", None],
        "canal_de_anuencia": ["Facebook", "Referencia", "Instagram", "Página Web", None],
        "canal_de_contacto": ["WhatsApp", "Referencia", "Facebook", "Página Web", None],
    })
    salida = agregar_cluster_y_percentil(entrada)
    assert len(salida) == len(entrada)
    assert salida[entrada.columns].equals(entrada)
    print("Ejemplo ficticio: el percentil expresa posición dentro de estas cinco filas.")
    print(salida[["id_ejemplo", "cluster", "percentil"]].to_string(index=False))

if __name__ == "__main__":
    main()
