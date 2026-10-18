"""Edición pública: utiliza artefactos sintéticos generados en examples/."""

import re
import unicodedata

import pandas as pd


EMPTY_TEXT_VALUES = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "-",
    "--",
    "sin dato",
    "sin datos",
    "no aplica",
    "no registrado",
    "n.a.",
}

# Acronyms that should stay uppercase when applying consistent text formatting.


COMMON_ACRONYMS = {"CRM", "NPS", "CSI", "FHA", "USD", "GTQ"}


CHANNEL_CANONICAL_MAP = {
    # Facebook - solo casing
    "facebook": "Facebook",
    "facebook ads": "Facebook Ads",
    # WhatsApp - casing y typo
    "whatsapp": "WhatsApp",
    "whats app": "WhatsApp",
    "wsp": "WhatsApp",
    # Instagram - solo casing
    "instagram": "Instagram",
    "instagram ads": "Instagram Ads",
    # Google
    "google": "Google",
    "google ads": "Google Ads",
    "adwords": "Google Ads",
    # TikTok
    "tiktok": "TikTok",
    "tik tok": "TikTok",
    # FB empresa_demo - agrupar como Facebook
    "fb empresa_demo": "Facebook",
    "fb expectativa": "Facebook",
    # IG empresa_demo - agrupar como Instagram
    "ig empresa_demo": "Instagram",
    # Inbox - solo casing; NO es WhatsApp
    "inbox": "Inbox",
    # ExpoCasa - preservar camelCase
    "expocasa": "ExpoCasa",
    # ExpoCasa Virtual - casing (Expocasa Virtual -> ExpoCasa Virtual)
    "expocasa virtual": "ExpoCasa Virtual",
    # Mupi ExpoCasa - agrupar como ExpoCasa
    "mupi expocasa": "ExpoCasa",
    # Open House - solo casing
    "open house": "Open House",
    # OpenSale
    "opensale": "OpenSale",
    "open sale": "OpenSale",
    # Correo - solo casing
    "correo": "Correo",
    "mailing": "Correo",
    # Friends/Family - estandarizar separador (Canal Anuencia usa /)
    "friends&family": "Friends/Family",
    "friends & family": "Friends/Family",
    "friends / family": "Friends/Family",
    "friends/family": "Friends/Family",
    # Lanzamiento - singular/plural mismo canal
    "lanzamientos": "Lanzamiento",
    "lanzamiento": "Lanzamiento",
    # Pag WEB - abreviacion obvia
    "pag web": "Página Web",
    "pagina web": "Página Web",
    # Corporacion - con tilde
    "corporacion": "Corporación",
    # portal_demo - preservar camelCase
    "portal_demo": "portal_demo",
    "portal_demo": "portal_demo",
    # Fanpage Lideres - preservar tilde
    "fanpage lideres": "Fanpage Líderes",
    # Actividades MKT - preservar acronimo en mayusculas
    "actividades mkt": "Actividades MKT",
    "actividades marketing": "Actividades MKT",
    # Visita Inedita - preservar tilde
    "visita inedita": "Visita Inédita",
    # Shareholders Sale
    "shareholders sale": "Shareholders Sale",
    # Ferias, LinkedIn, Vallas, Prensa - identity mappings para consistencia
    "ferias": "Ferias",
    "linkedin": "LinkedIn",
    "vallas": "Vallas",
    "prensa": "Prensa",
    "revistas": "Revistas",
    "sede": "Sede",
    "llamada": "Llamada",
    "messenger": "Facebook",
    "activaciones": "Activaciones",
    # Referencia - agrupar variantes
    "referencia personal": "Referencia",
    "referencia externa": "Referencia",
    "referencia terceros": "Referencia",
    "referencia de terceros": "Referencia",
    # USA/EEUU - agrupar variantes
    "usa la": "usa",
    "usa ca": "usa",
    "eeuu la": "usa",
    "usa": "usa",
}


MARITAL_CANONICAL_MAP = {
    "casad@": "Casad@",
    "casado": "Casad@",
    "casada": "Casad@",
    "solter@": "Solter@",
    "soltero": "Solter@",
    "soltera": "Solter@",
    "futuros conyuges": "Futuros Conyuges",
    "solter@ + hijos": "Solter@",
    "solter@ +hijos": "Solter@",
    "soltero + hijos": "Solter@",
    "soltera + hijos": "Solter@",
    "solter@ con hijos": "Solter@",
}


def normalize_for_comparison(value) -> str:
    # Convert values into a stable text form for comparisons.
    # This avoids bugs from differences like " Ganó ", "gano", and "GANÓ".
    if pd.isna(value):
        return ""

    # Strip spaces, lowercase, collapse repeated spaces, and remove accents.
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def clean_text_value(value):
    # Clean one categorical text value without touching numeric columns.
    # Missing-like strings become pd.NA so they can later be filled consistently.
    if pd.isna(value):
        return pd.NA

    # Do not alter real numeric values if this helper is accidentally called on one.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value

    # Strip leading/trailing spaces and collapse repeated spaces.
    text = re.sub(r"\s+", " ", str(value).strip())
    text_key = normalize_for_comparison(text)
    if text_key in EMPTY_TEXT_VALUES:
        return pd.NA

    # Apply readable title case while preserving common business acronyms.
    words = []
    for word in text.split(" "):
        bare_word_key = normalize_for_comparison(word).upper()
        if bare_word_key in COMMON_ACRONYMS:
            words.append(bare_word_key)
        else:
            words.append(word.lower().capitalize())

    return " ".join(words)


def is_missing_or_not_registered(value) -> bool:
    # Treat both real missing values and the standard label as missing for rules.
    if pd.isna(value):
        return True

    value_key = normalize_for_comparison(value)
    return value_key in EMPTY_TEXT_VALUES


def normalize_sex(value) -> str:
    # Homologate Sexo in the original column as requested.
    if is_missing_or_not_registered(value):
        return pd.NA

    value_key = normalize_for_comparison(value)
    # Single-letter codes: some exports abbreviate to "F" / "M". The fragment
    # rules below cannot catch them, and without this an "F" would fall through
    # to "Otro", a category the models never saw.
    if value_key in ("f", "m"):
        return "Femenino" if value_key == "f" else "Masculino"
    if any(term in value_key for term in ("fem", "mujer", "femenino")):
        return "Femenino"
    if any(term in value_key for term in ("masc", "hombre", "masculino")):
        return "Masculino"
    return "Otro"


def clean_channel_value_conservative(value):
    """
    Normalización conservadora de canal de contacto o anuencia.

    Solo corrige: casing, typos evidentes, separador Friends (& -> /),
    singular/plural del mismo canal (Lanzamientos -> Lanzamiento),
    abreviaciones obvias (Pag WEB -> Página Web), y tildes en display.

    NO agrupa:
    - Inbox != WhatsApp
    - Fanpage Líderes != Facebook
    - ExpoCasa != Evento

    Retorna pd.NA para missing. Para valores no mapeados, devuelve el valor
    con limpieza mínima de texto (strip + clean_text_value), sin cambiar
    su semántica.
    """
    if is_missing_or_not_registered(value):
        return pd.NA
    lookup_key = normalize_for_comparison(value)
    if lookup_key in CHANNEL_CANONICAL_MAP:
        return CHANNEL_CANONICAL_MAP[lookup_key]
    # Valor no mapeado: devolver con limpieza mínima, preservar significado
    cleaned = clean_text_value(value)
    if pd.isna(cleaned):
        return pd.NA
    return str(cleaned)


def normalize_marital_status(value):
    """
    Normaliza Estado Civil a los 4 valores reales del dataset.
    Para valores no reconocidos, devuelve el valor limpio (nunca "Otro").
    """
    if is_missing_or_not_registered(value):
        return pd.NA
    lookup_key = normalize_for_comparison(value)
    if lookup_key in MARITAL_CANONICAL_MAP:
        return MARITAL_CANONICAL_MAP[lookup_key]
    # Fallbacks por fragmento (variantes no previstas)
    if "casad" in lookup_key:
        return "Casad@"
    if "futuros" in lookup_key or "conyuge" in lookup_key:
        return "Futuros Conyuges"
    if "solter" in lookup_key and any(t in lookup_key for t in ("hijo", "hija", "+ h")):
        return "Solter@"
    if "solter" in lookup_key:
        return "Solter@"
    cleaned = clean_text_value(value)
    return str(cleaned) if not pd.isna(cleaned) else pd.NA


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
