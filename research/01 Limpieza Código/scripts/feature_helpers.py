import re

import pandas as pd


def classify_campania_plataforma(campania):
    if pd.isna(campania):
        return pd.NA

    c = str(campania).upper().strip()

    # portal_demo seguido de espacio = portal_demo
    if re.match(r"^portal_demo\s", c):
        return "portal_demo"

    # Eventos, brokers y otros no-campaña
    patron_otros = [
        r"\bEXPOCASA\b",
        r"\bEXPOSITUA\b",
        r"\bFIESTA\b",
        r"\bFERIA\b",
        r"\bMUPI\b",
        r"OPEN\s+HOUSE",
        r"OPEN\s+SMART",
        r"BIENES\s+RAI",
        r"GRAJEDA",
        r"HERNANDEZ",
        r"HERNÁNDEZ",
        r"^\s*VALLAS\s*$",
        r"^\s*PÁGINA\s+WEB\s*$",
    ]
    for patron in patron_otros:
        if re.search(patron, c):
            return "otro"

    # Por defecto: campaña directa de empresa_demo
    return "directo"


def classify_campania_tipo(campania):
    if pd.isna(campania):
        return pd.NA

    c = str(campania).upper().strip()

    reglas = [
        ("messenger", [r"\bMESSENGER\b"]),
        ("empresa_demo", [r"\bEMPRESA_DEMO\b"]),
        ("top", [r"\bTOP\b"]),
        ("footer", [r"\bFOOTER\b"]),
        ("inbox", [r"\bINBOX\b"]),
        ("fan_page", [r"FAN\s+PAGE", r"\bFANPAGE\b"]),
        ("open_house", [r"OPEN\s+HOUSE", r"OPEN\s+SMART"]),
        ("evento", [r"\bEXPOCASA\b", r"\bEXPOSITUA\b", r"\bFIESTA\b", r"\bFERIA\b", r"\bMUPI\b"]),
        ("segmentada", [r"\bSEGMENTADA\b"]),
        ("presupuesto_diario", [r"\bPPTO\b"]),
        ("whatsapp", [r"\bWHATSAPP\b"]),
        ("principal", [r"\bPRINCIPAL\d*\b", r"\bALA\s+OCT(?:UBRE)?\s+\d{2}\b"]),
        ("leads", [r"\bLEADS\b"]),
        ("cotizador", [r"\bCOTIZADOR\b"]),
    ]

    for tipo, patrones in reglas:
        for patron in patrones:
            if re.search(patron, c):
                return tipo

    return "otro"


def classify_campania_audiencia(campania):
    if pd.isna(campania):
        return pd.NA

    c = str(campania).upper().strip()

    if re.search(r"\bUSA\s+LA\b", c) or re.search(r"\bEEUU\s+LA\b", c):
        return "la"
    if re.search(r"\bUSA\s+CA\b", c) or re.search(r"\bEEUU\s+CA\b", c):
        return "ca"
    if re.search(r"\bEEUU\b", c) or re.search(r"\bUSA\b", c):
        return "eeuu"

    return "guatemala"


def create_rango_edad_modelo(edad_value):
    """
    Reconstituye los rangos del CRM desde la edad numérica.
    Rangos idénticos a los originales del campo 'Rango de Edad'.
    """
    if pd.isna(edad_value):
        return pd.NA

    try:
        edad = float(edad_value)
    except (ValueError, TypeError):
        return pd.NA

    if edad < 18 or edad > 90:
        return pd.NA
    if edad <= 27:
        return "menor_28"
    if edad <= 34:
        return "28_34"
    if edad <= 45:
        return "35_45"
    if edad <= 52:
        return "46_52"
    if edad <= 60:
        return "53_60"
    return "mayor_60"


def build_campania_mapping_report(df, reports_dir):
    mapping = df["campania"].value_counts(dropna=False).reset_index()
    mapping.columns = ["campania_valor", "frecuencia"]
    mapping["campania_plataforma"] = mapping["campania_valor"].apply(
        classify_campania_plataforma
    )
    mapping["campania_tipo"] = mapping["campania_valor"].apply(classify_campania_tipo)
    mapping["campania_audiencia"] = mapping["campania_valor"].apply(
        classify_campania_audiencia
    )

    mapping = mapping[
        [
            "campania_valor",
            "campania_plataforma",
            "campania_tipo",
            "campania_audiencia",
            "frecuencia",
        ]
    ]

    output_path = reports_dir / "campania_mapping.xlsx"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        mapping.to_excel(writer, sheet_name="mapeo_campania", index=False)

    return mapping
