"""Entrena artefactos de demostración sin leer ningún dato real."""
from pathlib import Path
import json, pickle, sys
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "research/05 Power BI"))
from scoring_leads.preparacion import preparar

def main():
    rng=np.random.default_rng(2026)
    n=1200
    raw=pd.DataFrame({
      "edad":rng.integers(22,71,n),
      "sexo":rng.choice(["Femenino","Masculino"],n),
      "estado_civil":rng.choice(["Casado","Soltero","Futuros Conyuges"],n),
      "razon_de_compra":rng.choice(["Vivienda","Inversión"],n),
      "tipo_unidad":rng.choice(["1 Hab","2 Habs","3 Habs","4 Habs","Bodega"],n),
      "ingreso_familiar":rng.choice(["Q18k A Q21.9k","Q30k A 33.9k","Q50k A Más"],n),
      "canal_de_anuencia":rng.choice(["Facebook","Referencia","Instagram","Página Web"],n),
      "canal_de_contacto":rng.choice(["Facebook","WhatsApp","Página Web"],n),
      "lugar_trabajo":rng.choice(["Zona Demo 1","Zona Demo 2"],n),
      "lugar_vivienda":rng.choice(["Zona Demo 1","Zona Demo 2"],n),
      "opciones_visitadas":rng.choice(["Ninguna","proyecto_demo"],n),
      "participantes_en_credito":rng.choice(["1","2"],n),
      "campania":rng.choice(["proyecto01 LEADS","proyecto02 PRINCIPAL","proyecto03 WHATSAPP"],n),
    })
    df=preparar(raw)
    folder=ROOT / "research/05 Power BI/scoring_leads/modelos"
    folder.mkdir(parents=True,exist_ok=True)
    cats={c:sorted(df[c].dropna().unique().tolist()) for c in ["sexo","estado_civil","razon_de_compra","tipo_unidad"]}
    prep={"descripcion":"DEMO SINTETICA: no corresponde al entrenamiento histórico", "origen_datos":"generador independiente con semilla 2026", "random_state":2026,"k":6,
      "n_observaciones_entrenamiento":n,"peso_edad":0.25,
      "pesos_categoricas":{"sexo":1.2,"estado_civil":1.5,"razon_de_compra":1.4,"tipo_unidad":1.0},
      "orden_bloques":["edad",*cats], "edad_scaler":{"mean":float(df.edad.mean()),"scale":float(df.edad.std(ddof=0))},
      "categorias":cats,"n_categorias_entrenamiento":{c:len(v) for c,v in cats.items()},"n_features":1+sum(map(len,cats.values()))}
    (folder / "preprocesamiento_kmeans_R1.json").write_text(json.dumps(prep,ensure_ascii=False,indent=2))
    from scoring_leads.artefactos import cargar_preprocesamiento_kmeans
    cargar_preprocesamiento_kmeans.cache_clear()
    from scoring_leads.clustering import construir_matriz
    km=KMeans(n_clusters=6,random_state=2026,n_init=10).fit(construir_matriz(df))
    joblib.dump(km,folder / "modelo_kmeans_R1.joblib")
    numeric=["edad_n","ingreso_mid","es_inversion","es_referido","edad_x_inversion","ingreso_x_inversion","edad_x_ingreso","edad_x_referido","mayor_45","ingreso_alto"]
    categorical=["sexo","estado_civil","ingreso_familiar","razon_de_compra","canal_de_anuencia","canal_de_contacto","canal_familia","tipo_unidad","opciones_visitadas","participantes_en_credito","lugar_trabajo","lugar_vivienda","campania_tipo","campania_audiencia"]
    preprocess=ColumnTransformer([
      ("numeric",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),
      ("categorical",Pipeline([("impute",SimpleImputer(strategy="constant",fill_value="_missing_")),("encode",OneHotEncoder(handle_unknown="ignore"))]),categorical)])
    # This artificial target is a demonstration rule, unrelated to actual buyers.
    probability=1/(1+np.exp(-(-2.5+1.2*df.es_referido+0.8*df.es_inversion+0.015*(df.edad_n-40))))
    target=rng.binomial(1,probability)
    features=numeric+categorical
    model=Pipeline([("preprocess",preprocess),("model",LogisticRegression(max_iter=1000,random_state=2026))]).fit(df[features],target)
    with (folder / "modelo_conversion_lr.pkl").open("wb") as f:pickle.dump({"pipeline":model,"features":features,"synthetic":True,"seed":2026},f)
    print("Artefactos sintéticos generados. Ningún archivo de datos reales fue leído.")

if __name__ == "__main__":main()
