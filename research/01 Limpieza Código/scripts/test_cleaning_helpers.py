import pandas as pd

from cleaning_helpers import (
    clean_channel_value_conservative,
    normalize_marital_status,
    normalize_tiene_hijos,
)
from schema_contract import (
    ANALYTICAL_LABEL_REPLACEMENTS,
    FORBIDDEN_ANALYTICAL_COLUMNS,
    keep_human_readable_labels,
    validate_analytical_label_schema,
)


# Contrato de columnas analíticas: conservar labels y eliminar códigos duplicados.
schema_input = pd.DataFrame(
    {
        **{
            code: ["codigo_no_usado", "respaldo_original"]
            for code in ANALYTICAL_LABEL_REPLACEMENTS
        },
        **{
            label: ["Etiqueta", pd.NA]
            for label in ANALYTICAL_LABEL_REPLACEMENTS.values()
        },
        "tipomodelo_label": ["duplicada", "duplicada"],
    }
)
schema_output, removed_columns = keep_human_readable_labels(schema_input)
assert set(removed_columns) == FORBIDDEN_ANALYTICAL_COLUMNS
assert not (FORBIDDEN_ANALYTICAL_COLUMNS & set(schema_output.columns))
assert set(ANALYTICAL_LABEL_REPLACEMENTS.values()).issubset(schema_output.columns)
for label in ANALYTICAL_LABEL_REPLACEMENTS.values():
    assert schema_output[label].tolist() == ["Etiqueta", "respaldo_original"]
validate_analytical_label_schema(schema_output, "test_schema_output")

original_only, _ = keep_human_readable_labels(
    pd.DataFrame({"apartamento_no": ["A-101"]})
)
assert "apartamento_no" not in original_only.columns
assert original_only["apartamento_no_label"].tolist() == ["A-101"]
validate_analytical_label_schema(
    original_only,
    "test_original_only",
    require_all_labels=False,
)

label_only, _ = keep_human_readable_labels(
    pd.DataFrame({"apartamento_no_label": ["A-102"]})
)
assert label_only["apartamento_no_label"].tolist() == ["A-102"]


# clean_channel_value_conservative
assert clean_channel_value_conservative("FaceBook") == "Facebook"
assert clean_channel_value_conservative("Whatsapp") == "WhatsApp"
assert clean_channel_value_conservative("FB empresa_demo") == "Facebook"
assert clean_channel_value_conservative("FB EXPECTATIVA") == "Facebook"
assert clean_channel_value_conservative("IG empresa_demo") == "Instagram"
assert clean_channel_value_conservative("Messenger") == "Facebook"
assert clean_channel_value_conservative("INBOX") == "Inbox"
assert clean_channel_value_conservative("ExpoCasa") == "ExpoCasa"
assert clean_channel_value_conservative("Lanzamiento") == "Lanzamiento"
assert clean_channel_value_conservative("Lanzamientos") == "Lanzamiento"
assert clean_channel_value_conservative("Fanpage Líderes") == "Fanpage Líderes"
assert clean_channel_value_conservative("Friends&Family") == "Friends/Family"
assert clean_channel_value_conservative("Expocasa Virtual") == "ExpoCasa Virtual"
assert clean_channel_value_conservative("Mailing") == "Correo"
assert clean_channel_value_conservative("Mupi Expocasa") == "ExpoCasa"
assert clean_channel_value_conservative("OPEN HOUSE") == "Open House"
assert clean_channel_value_conservative("Referencia Personal") == "Referencia"
assert clean_channel_value_conservative("Referencia Externa") == "Referencia"
assert clean_channel_value_conservative("Referencia Terceros") == "Referencia"
assert clean_channel_value_conservative("USA LA") == "usa"
assert clean_channel_value_conservative("USA") == "usa"
assert clean_channel_value_conservative("USA CA") == "usa"
assert clean_channel_value_conservative("EEUU LA") == "usa"
# Verificar que los siguientes NO colapsan entre sí:
assert clean_channel_value_conservative("INBOX") != "WhatsApp"
assert clean_channel_value_conservative("Fanpage Líderes") != "Facebook"
assert clean_channel_value_conservative("ExpoCasa") != "Evento"
# Missing -> pd.NA
assert pd.isna(clean_channel_value_conservative(None))
assert pd.isna(clean_channel_value_conservative(""))
assert pd.isna(clean_channel_value_conservative("sin dato"))

# normalize_tiene_hijos
assert normalize_tiene_hijos("0") == "sin hijos"
assert normalize_tiene_hijos("no") == "sin hijos"
assert normalize_tiene_hijos("1") == "con hijos"
assert normalize_tiene_hijos("si") == "con hijos"
assert normalize_tiene_hijos("1 HIJO") == "con hijos"
assert normalize_tiene_hijos("2 Hijas") == "con hijos"
assert normalize_tiene_hijos("En camino") == "con hijos"
assert pd.isna(normalize_tiene_hijos("34"))
assert pd.isna(normalize_tiene_hijos("150000"))
assert pd.isna(normalize_tiene_hijos(None))
assert pd.isna(normalize_tiene_hijos("no quiso responder"))
assert normalize_tiene_hijos("aun no") == "sin hijos"
assert normalize_tiene_hijos("noi") == "sin hijos"
assert normalize_tiene_hijos("np") == "sin hijos"
assert pd.isna(normalize_tiene_hijos("ces"))
assert normalize_tiene_hijos("mo") == "sin hijos"

tiene_hijos_expected = {
    "dos": "con hijos",
    "mayores": "con hijos",
    "tres": "con hijos",
    "o": pd.NA,
    "7": "con hijos",
    "dos mayores": "con hijos",
    "si 2": "con hijos",
    "S/N": pd.NA,
    "00": "sin hijos",
    "34": pd.NA,
    "25": pd.NA,
    "21": pd.NA,
    "11 años": "con hijos",
    "nieto": "con hijos",
    ".0": "sin hijos",
    "tres adultos": "con hijos",
    "SOLTERA": "sin hijos",
    "uno de 14 años": "con hijos",
    "ADULTOS": "con hijos",
    "MAYORES": "con hijos",
    "una mayor": "con hijos",
    "n": pd.NA,
    "uno mayor": "con hijos",
    "un bebe": "con hijos",
    "tres mayores": "con hijos",
    "una bebe": "con hijos",
    "2 MAYORES": "con hijos",
    "NO": "sin hijos",
    "-": "sin hijos",
    "30": pd.NA,
    "150000": pd.NA,
    "8000": pd.NA,
    "Si (No quiso decir cuantos)": "con hijos",
    "28": pd.NA,
    "12": pd.NA,
    "ces": pd.NA,
    "no quiso responder": pd.NA,
    "SI, NO A SU CARGO": "con hijos",
    "8": pd.NA,
    "69": pd.NA,
    "31": pd.NA,
    "nietos": "con hijos",
    "2, 19 y 12": "con hijos",
    "si uno": "con hijos",
    "dos": "con hijos",
    "si, adultos": "con hijos",
    "}2": "con hijos",
    "1|": "con hijos",
    ",2": "con hijos",
    ' "no': "sin hijos",
    "10": pd.NA,
}

for original, expected in tiene_hijos_expected.items():
    normalized = normalize_tiene_hijos(original)
    if pd.isna(expected):
        assert pd.isna(normalized), f"{original!r}: esperado NA, obtuvo {normalized!r}"
    else:
        assert not pd.isna(normalized), f"{original!r}: esperado {expected!r}, obtuvo NA"
        assert normalized == expected, f"{original!r}: esperado {expected!r}, obtuvo {normalized!r}"

# normalize_marital_status
assert normalize_marital_status("Casad@") == "Casad@"
assert normalize_marital_status("Solter@") == "Solter@"
assert normalize_marital_status("Futuros Conyuges") == "Futuros Conyuges"
assert normalize_marital_status("Solter@ + hijos") == "Solter@"

print(f'Messenger -> "{clean_channel_value_conservative("Messenger")}"')
print(f'FB empresa_demo -> "{clean_channel_value_conservative("FB empresa_demo")}"')
print(f'IG empresa_demo -> "{clean_channel_value_conservative("IG empresa_demo")}"')
print(f'Referencia Personal -> "{clean_channel_value_conservative("Referencia Personal")}"')
print(f'USA LA -> "{clean_channel_value_conservative("USA LA")}"')
print(f'INBOX -> "{clean_channel_value_conservative("INBOX")}"')
print(f'normalize_tiene_hijos("34") is NA: {pd.isna(normalize_tiene_hijos("34"))}')
print("TODAS LAS PRUEBAS PASARON")
