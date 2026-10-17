import re
import unicodedata

import pandas as pd


# This order is used only to decide which duplicate record is the best one.
# Higher numbers mean a more advanced stage, so those rows win during dedupe.
STAGE_PRIORITY = {
    "contactados": 1,
    "ingresados": 2,
    "interesados": 3,
    "visita programada": 4,
    "visita modelo": 5,
    "cortos plazos": 6,
    "corto plazos": 6,
    "reserva": 7,
    "suspension": 8,
    "armado expediente": 9,
    "analisis credito": 10,
    "aprobacion": 11,
    "resolucion": 12,
    "expediente tecnico": 13,
    "formalizacion": 14,
}

# These words identify rows that look like internal/test records.
# The regex is case-insensitive and checks full words like "test" or "demo".
TEST_RECORD_PATTERN = re.compile(
    r"\b(?:test|prueba|testing|demo|dummy|example)\b",
    flags=re.IGNORECASE,
)

# If a categorical column has too many options, the file stays readable by showing
# a short preview instead of placing hundreds or thousands of values in one cell.
MAX_DISTINCT_VALUES_TO_LIST = 100

# These words help detect columns that usually contain free text written by users.
OPEN_RESPONSE_COLUMN_KEYWORDS = (
    "comentario",
    "comment",
    "nota",
    "observacion",
    "descripcion",
    "description",
)

# Values that should be treated as missing when cleaning categorical text.
# They are normalized before comparison, so " N/A ", "n.a.", and "NA" are caught.
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

PROJECT_CANONICAL_MAP = {
    "proyecto01": "proyecto01",
    "proyecto01": "proyecto01",
    "proyecto02": "proyecto02",
    "proyecto02": "proyecto02",
    "mariscal": "proyecto02",
    "proyecto03": "proyecto03",
    "proyecto05": "proyecto05",
    "proyecto04": "proyecto04",
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

TIENE_HIJOS_ZERO_KEYS = {
    "0",
    "0.0",
    "0.00",
    "0.",
    "no",
    "no tiene",
    "no tienen",
    "ninguno",
    "ninguna",
    "no hijos",
    "no tiene hijos",
    "sin hijos",
    "no tiene hijo",
    "aun no",
    "noi",
    "np",
    "mo",
}

TIENE_HIJOS_EXPLICIT_MAP = {
    "dos": "con hijos",
    "mayores": "con hijos",
    "tres": "con hijos",
    "o": pd.NA,
    "7": "con hijos",
    "dos mayores": "con hijos",
    "si 2": "con hijos",
    "s/n": pd.NA,
    "00": "sin hijos",
    "34": pd.NA,
    "25": pd.NA,
    "21": pd.NA,
    "11 anos": "con hijos",
    "nieto": "con hijos",
    ".0": "sin hijos",
    "tres adultos": "con hijos",
    "soltera": "sin hijos",
    "uno de 14 anos": "con hijos",
    "adultos": "con hijos",
    "una mayor": "con hijos",
    "n": pd.NA,
    "uno mayor": "con hijos",
    "un bebe": "con hijos",
    "tres mayores": "con hijos",
    "una bebe": "con hijos",
    "2 mayores": "con hijos",
    "no": "sin hijos",
    "-": "sin hijos",
    "30": pd.NA,
    "150000": pd.NA,
    "8000": pd.NA,
    "si (no quiso decir cuantos)": "con hijos",
    "28": pd.NA,
    "12": pd.NA,
    "ces": pd.NA,
    "no quiso responder": pd.NA,
    "si, no a su cargo": "con hijos",
    "8": pd.NA,
    "69": pd.NA,
    "31": pd.NA,
    "nietos": "con hijos",
    "2, 19 y 12": "con hijos",
    "si uno": "con hijos",
    "si, adultos": "con hijos",
    "}2": "con hijos",
    "1|": "con hijos",
    ",2": "con hijos",
    '"no': "sin hijos",
    "10": pd.NA,
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


def readable_title(value):
    # Keep the value readable after spacing cleanup.
    # This is used for display/output, not for comparison logic.
    if pd.isna(value):
        return value

    # Title-case most text while preserving common small words.
    cleaned_value = clean_text_value(value)
    if pd.isna(cleaned_value):
        return cleaned_value

    words = str(cleaned_value).lower().split(" ")
    small_words = {"de", "del", "la", "las", "los", "y", "en"}
    titled_words = [
        word if word in small_words else word.capitalize()
        for word in words
    ]
    return " ".join(titled_words)


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


def clean_repeated_campaign_text(value):
    # Clean campaign text and collapse exact duplicated strings such as
    # "proyecto01 Footerproyecto01 Footer" into "proyecto01 Footer".
    if is_missing_or_not_registered(value):
        return pd.NA

    text = clean_text_value(value)
    if pd.isna(text):
        return pd.NA

    text = str(text).strip()
    halfway = len(text) // 2
    if len(text) % 2 == 0 and text[:halfway].strip() == text[halfway:].strip():
        text = text[:halfway].strip()

    return text or pd.NA


def fix_duplicated_campaign_text(value):
    """
    Detecta y corrige campañas donde el nombre fue concatenado con
    sí mismo sin separador.
    """
    if is_missing_or_not_registered(value):
        return pd.NA

    s = str(value).strip()
    n = len(s)

    # Caso 1: duplicación exacta por mitades
    if n >= 6 and n % 2 == 0:
        half = n // 2
        if s[:half] == s[half:]:
            return s[:half].strip()

    # Caso 2: prefijo repetido (con texto adicional después)
    for split in range(5, n // 2 + 1):
        prefix = s[:split]
        if s.startswith(prefix, split):
            canonical = prefix + s[split + len(prefix):]
            return canonical.strip()

    return s


def clean_campaign_spacing(value):
    """
    Corrige problemas de espaciado y puntuación en campania.
    """
    if is_missing_or_not_registered(value):
        return pd.NA

    s = str(value).strip()
    s = re.sub(r"  +", " ", s)
    s = re.sub(r"(\w)-\s+(\w)", r"\1-\2", s)
    s = s.rstrip(".")
    return s.strip() if s else pd.NA


def remove_copia_suffix(value):
    """
    Fusiona campañas marcadas como 'Copia' con su versión original.
    """
    if is_missing_or_not_registered(value):
        return pd.NA

    s = str(value).strip()
    s = re.sub(r"\s*-\s*copia\s*$", "", s, flags=re.IGNORECASE).strip()
    return s if s else pd.NA


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


def normalize_project(value):
    """
    Mapea a los 5 proyectos conocidos. Valores no reconocidos se devuelven
    limpios (nunca "Otro").
    """
    if is_missing_or_not_registered(value):
        return pd.NA
    lookup_key = normalize_for_comparison(value)
    if lookup_key in PROJECT_CANONICAL_MAP:
        return PROJECT_CANONICAL_MAP[lookup_key]
    cleaned = clean_text_value(value)
    return str(cleaned) if not pd.isna(cleaned) else pd.NA


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


def normalize_tiene_hijos(value):
    """
    Normaliza 'tiene_hijos' (52 valores únicos originales) a:
    'sin hijos', 'con hijos', o pd.NA.

    Valores corruptos (edades, montos: 34, 28, 150000, 8000, 21, 12,
    25, 30) se convierten a NA. Valores ambiguos definidos por negocio
    pueden convertirse a NA o a sin hijos.
    """
    if pd.isna(value):
        return pd.NA
    key = normalize_for_comparison(str(value)).strip()

    if key in TIENE_HIJOS_EXPLICIT_MAP:
        return TIENE_HIJOS_EXPLICIT_MAP[key]

    if is_missing_or_not_registered(value):
        return pd.NA

    # Cero explícito
    if key in TIENE_HIJOS_ZERO_KEYS:
        return "sin hijos"

    # Uno explícito (palabras afirmativas)
    if key in {"si", "s", "si no a su cargo"}:
        return "con hijos"

    # Palabras relacionadas con hijos (sin prefijo negativo)
    CHILD_WORDS = ("hijo", "hija", "nino", "nina", "nena")
    if any(w in key for w in CHILD_WORDS) and not key.startswith("no "):
        return "con hijos"

    # "En camino" = embarazada
    if "camino" in key:
        return "con hijos"

    # "uno", "una"
    if key in {"uno", "una"}:
        return "con hijos"

    # Numérico >= 7 o multi-dígito -> dato corrupto (edad, monto) -> NA
    try:
        first_token = key.split()[0]
        n = int(float(first_token))
        if 1 <= n <= 6:
            return "con hijos"
        else:
            return pd.NA  # 7, 8, 12, 21, 28, 34, 150000, etc.
    except (ValueError, IndexError):
        pass

    # Ambiguos definidos por negocio como NA.
    if any(t in key for t in ("quiso", "ces")):
        return pd.NA

    return pd.NA  # Cualquier otro valor ininterpretable


def find_email_columns(df: pd.DataFrame) -> list[str]:
    # Detect email columns by common Spanish and English column names.
    # This keeps the script flexible if a future export uses "correo" instead of "email".
    email_keywords = ("email", "e-mail", "correo", "mail")
    return [
        column
        for column in df.columns
        if any(keyword in normalize_for_comparison(column) for keyword in email_keywords)
    ]


def find_phone_columns(df: pd.DataFrame) -> list[str]:
    # Detect phone columns by common Spanish and English column names.
    # Phone-like columns are normalized by keeping only digits.
    phone_keywords = ("phone", "telefono", "celular", "movil", "whatsapp")
    return [
        column
        for column in df.columns
        if any(keyword in normalize_for_comparison(column) for keyword in phone_keywords)
    ]


def calculate_column_fullness(df: pd.DataFrame) -> pd.DataFrame:
    # Count both real nulls and blank strings as empty values.
    # This answers: "What percent of this column is actually filled?"
    filled_mask = df.notna()

    # Treat text cells with only spaces as empty values.
    # A cell like "   " should not count as real data.
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        filled_mask[column] = df[column].astype("string").str.strip().ne("").fillna(False)

    # Build one row per column with filled counts, empty counts, and fullness percent.
    # The final report is easy to open in Excel or review in the terminal.
    total_rows = len(df)
    filled_count = filled_mask.sum()
    fullness_report = pd.DataFrame(
        {
            "column": df.columns,
            "filled_rows": filled_count.reindex(df.columns).astype(int).to_numpy(),
            "empty_rows": (total_rows - filled_count.reindex(df.columns)).astype(int).to_numpy(),
            "fullness_percent": (
                filled_count.reindex(df.columns).div(total_rows).mul(100).round(2).to_numpy()
                if total_rows
                else 0
            ),
        }
    )

    # Show least complete columns first because they are usually the most useful to inspect.
    # This puts sparse fields like optional survey columns at the top.
    return fullness_report.sort_values(
        ["fullness_percent", "column"],
        ascending=[True, True],
    )


def print_column_fullness_report(fullness_report: pd.DataFrame) -> None:
    # Print the percent of fullness for each column.
    # The saved CSV has the same information in a sortable table.
    print("\nColumn fullness report")
    for _, row in fullness_report.iterrows():
        print(
            f"{row['column']}: {row['fullness_percent']}% "
            f"({row['filled_rows']} filled / {row['empty_rows']} empty)"
        )


def is_date_like_column(column: str, series: pd.Series) -> bool:
    # Detect date columns by name first because this export uses Spanish date labels.
    column_key = normalize_for_comparison(column)
    date_keywords = ("fecha", "date", "time")
    return any(keyword in column_key for keyword in date_keywords)


def format_distinct_values(values: pd.Series) -> str:
    # Convert distinct values into a readable semicolon-separated list.
    cleaned_values = values.dropna().astype("string").str.strip()
    cleaned_values = cleaned_values[cleaned_values.ne("")]
    distinct_values = sorted(cleaned_values.unique(), key=lambda value: str(value).lower())

    # Keep the dictionary readable when a categorical column has many options.
    if len(distinct_values) > MAX_DISTINCT_VALUES_TO_LIST:
        preview = "; ".join(distinct_values[:MAX_DISTINCT_VALUES_TO_LIST])
        return (
            f"{preview}; ... "
            f"({len(distinct_values)} valores distintos en total)"
        )

    return "; ".join(distinct_values)


def classify_numeric_column(column: str, series: pd.Series, distinct_count: int) -> str:
    # Numeric columns with few repeated values are usually coded categories.
    column_key = normalize_for_comparison(column)
    non_empty_count = int(series.notna().sum())
    unique_ratio = distinct_count / non_empty_count if non_empty_count else 0

    # ID/code fields are numeric categorical because their numbers are labels, not quantities.
    code_keywords = ("id", "stage", "pipeline", "visible")
    if any(keyword in column_key for keyword in code_keywords):
        return "numerica categorica"

    # Binary fields and small code fields are numeric categorical.
    if distinct_count <= 20 and unique_ratio <= 0.05:
        return "numerica categorica"

    # Integer-like columns with repeated values are discrete numeric variables.
    numeric_values = pd.to_numeric(series, errors="coerce").dropna()
    is_integer_like = bool((numeric_values % 1 == 0).all()) if len(numeric_values) else False
    if is_integer_like:
        return "numerica discreta"

    # Decimal-valued measurements or money amounts are continuous numeric variables.
    return "numerica continua"


def classify_text_column(column: str, series: pd.Series, distinct_count: int) -> str:
    # Text columns with long user-written content are treated as open response.
    column_key = normalize_for_comparison(column)
    non_empty = series.dropna().astype("string").str.strip()
    non_empty = non_empty[non_empty.ne("")]
    average_length = non_empty.str.len().mean() if len(non_empty) else 0
    unique_ratio = distinct_count / len(non_empty) if len(non_empty) else 0

    # Known note/comment columns are open response even if they have repeated values.
    if any(keyword in column_key for keyword in OPEN_RESPONSE_COLUMN_KEYWORDS):
        return "respuesta abierta"

    # High-cardinality long text usually means names, free notes, or manually entered text.
    if distinct_count > 50 and (average_length >= 35 or unique_ratio >= 0.5):
        return "respuesta abierta"

    # Otherwise, text columns are treated as categorical options.
    return "categorica"


def build_variable_info(df: pd.DataFrame, fullness_report: pd.DataFrame) -> pd.DataFrame:
    # Build a variable dictionary that explains each column in the cleaned dataset.
    fullness_lookup = fullness_report.set_index("column")["fullness_percent"].to_dict()
    variable_rows = []

    for column in df.columns:
        series = df[column]

        # Blank strings should not count as real distinct values.
        non_empty = series.dropna()
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            non_empty = non_empty.astype("string").str.strip()
            non_empty = non_empty[non_empty.ne("")]

        distinct_count = int(non_empty.nunique(dropna=True))
        missing_count = int(len(df) - len(non_empty))

        # Date columns get a date range instead of an options list.
        if is_date_like_column(column, series):
            parsed_dates = pd.to_datetime(series, errors="coerce").dropna()
            tipo_variable = "fecha"
            if len(parsed_dates):
                values_or_range = (
                    f"{parsed_dates.min().date()} a {parsed_dates.max().date()}"
                )
            else:
                values_or_range = "sin fechas validas"

        # Numeric columns get a min/max range and a numeric subtype.
        elif pd.api.types.is_numeric_dtype(series):
            numeric_values = pd.to_numeric(series, errors="coerce").dropna()
            tipo_variable = classify_numeric_column(column, series, distinct_count)
            if len(numeric_values):
                values_or_range = f"{numeric_values.min()} a {numeric_values.max()}"
            else:
                values_or_range = "sin valores numericos validos"

        # Text columns are either categorical options or open response.
        else:
            tipo_variable = classify_text_column(column, series, distinct_count)
            if tipo_variable == "respuesta abierta":
                values_or_range = "abierta"
            else:
                values_or_range = format_distinct_values(series)

        variable_rows.append(
            {
                "columna": column,
                "tipo_variable": tipo_variable,
                "valores_distintos_o_rango": values_or_range,
                "distinct_count": distinct_count,
                "missing_count": missing_count,
                "fullness_percent": fullness_lookup.get(column, 0),
            }
        )

    return pd.DataFrame(variable_rows)


def format_summary_value(value):
    # Convert values into CSV-friendly strings while keeping missing values blank.
    if pd.isna(value):
        return ""
    return str(value)
