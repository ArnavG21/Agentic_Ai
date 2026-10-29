"""The agent covers European cities only (the brief: restaurants for travellers in Europe).

EUROPE        - ISO country codes accepted, with names for friendly messages
CLOSED_WORDS  - how "closed for good" is usually written, by language, used to spot closure news
"""
from __future__ import annotations

EUROPE = {
    "AD": "Andorra", "AL": "Albania", "AT": "Austria", "AX": "Åland", "BA": "Bosnia and Herzegovina",
    "BE": "Belgium", "BG": "Bulgaria", "BY": "Belarus", "CH": "Switzerland", "CY": "Cyprus", "CZ": "Czechia",
    "DE": "Germany", "DK": "Denmark", "EE": "Estonia", "ES": "Spain", "FI": "Finland", "FO": "Faroe Islands",
    "FR": "France", "GB": "United Kingdom", "GG": "Guernsey", "GI": "Gibraltar", "GR": "Greece", "HR": "Croatia",
    "HU": "Hungary", "IE": "Ireland", "IM": "Isle of Man", "IS": "Iceland", "IT": "Italy", "JE": "Jersey",
    "LI": "Liechtenstein", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "MC": "Monaco", "MD": "Moldova",
    "ME": "Montenegro", "MK": "North Macedonia", "MT": "Malta", "NL": "Netherlands", "NO": "Norway", "PL": "Poland",
    "PT": "Portugal", "RO": "Romania", "RS": "Serbia", "RU": "Russia", "SE": "Sweden", "SI": "Slovenia",
    "SK": "Slovakia", "SM": "San Marino", "TR": "Türkiye", "UA": "Ukraine", "VA": "Vatican City", "XK": "Kosovo",
}

EXAMPLE_CITIES = "Barcelona, Lisbon, Rome, Paris, Berlin or Prague"


def is_european(country_code: str) -> bool:
    return (country_code or "").upper() in EUROPE


# Words that signal a restaurant has closed for good, by language (lower case).
# Used only to decide which search results are worth showing to the AI for a closer look.
CLOSED_WORDS = {
    "en": ["permanently closed", "closed permanently", "closed for good", "has closed", "have closed", "closes its doors",
           "closed its doors", "closing down", "last service", "shut down", "no longer open"],
    "es": ["cierra", "cerrado", "cierre", "último servicio", "ultimo servicio", "ha llegado a su fin", "baja la persiana"],
    "ca": ["tanca", "tancat", "tancament", "últim servei"],
    "fr": ["fermé définitivement", "ferme définitivement", "fermeture définitive", "ferme ses portes", "a fermé"],
    "it": ["chiuso definitivamente", "chiude", "chiusura definitiva", "ha chiuso"],
    "de": ["dauerhaft geschlossen", "geschlossen", "schließt", "schliesst", "hat geschlossen"],
    "pt": ["encerrado", "encerra", "fechado definitivamente", "fecha portas", "fechou"],
    "nl": ["definitief gesloten", "gesloten", "sluit de deuren"],
    "pl": ["zamknięta", "zamknięte", "zamknięty", "zamyka"],
    "cs": ["uzavřeno", "zavírá", "skončila"],
    "el": ["έκλεισε", "κλείνει"],
    "sv": ["stänger", "har stängt", "permanent stängt"],
    "da": ["lukker", "har lukket"],
    "no": ["legger ned", "har stengt"],
    "hu": ["bezár", "bezárt"],
    "ro": ["se închide", "s-a închis"],
    "hr": ["zatvoren", "zatvara", "zatvorio", "zatvorila"],          # also Bosnian, Montenegrin, Serbian (Latin)
    "sr": ["затворен", "затвара", "zatvoren", "zatvara"],
    "sl": ["zaprt", "zapira", "zaprla"],
    "bg": ["затвори", "затваря", "затворен"],                        # also used for North Macedonia
    "sk": ["zatvorené", "zatvára", "skončila"],
    "fi": ["suljettu", "sulkee", "lopettaa"],
    "et": ["suletud", "sulgeb"],
    "lv": ["slēgts", "slēdz"],
    "lt": ["uždarytas", "uždaro"],
    "is": ["lokað", "lokar"],
    "sq": ["mbyllet", "mbyll"],
    "tr": ["kapandı", "kapatıldı", "kapanıyor"],
    "uk": ["закрився", "закрито", "закривається"],
    "ru": ["закрылся", "закрыт", "закрывается"],
}
COUNTRY_LANGS = {"ES": ["es", "ca"], "AD": ["ca", "es"], "FR": ["fr"], "BE": ["fr", "nl"], "MC": ["fr"], "LU": ["fr", "de"],
                 "IT": ["it"], "SM": ["it"], "VA": ["it"], "CH": ["de", "fr", "it"], "DE": ["de"], "AT": ["de"],
                 "LI": ["de"], "PT": ["pt"], "NL": ["nl"], "PL": ["pl"], "CZ": ["cs"], "GR": ["el"], "CY": ["el"],
                 "SE": ["sv"], "DK": ["da"], "NO": ["no"], "HU": ["hu"], "RO": ["ro"], "MD": ["ro"],
                 "HR": ["hr"], "BA": ["hr", "sr"], "ME": ["hr", "sr"], "RS": ["sr"], "SI": ["sl"], "BG": ["bg"],
                 "MK": ["bg"], "SK": ["sk"], "FI": ["fi", "sv"], "AX": ["sv"], "EE": ["et"], "LV": ["lv"],
                 "LT": ["lt"], "IS": ["is"], "FO": ["da"], "AL": ["sq"], "XK": ["sq", "sr"], "TR": ["tr"],
                 "UA": ["uk"], "RU": ["ru"], "BY": ["ru"]}
# English-speaking places (GB, IE, MT, GI, IM, JE, GG) use the English words only.
ENGLISH_ONLY = {"GB", "IE", "MT", "GI", "IM", "JE", "GG"}


def closed_words(country_code: str) -> list[str]:
    words = list(CLOSED_WORDS["en"])
    for lang in COUNTRY_LANGS.get((country_code or "").upper(), []):
        words += CLOSED_WORDS.get(lang, [])
    return words
