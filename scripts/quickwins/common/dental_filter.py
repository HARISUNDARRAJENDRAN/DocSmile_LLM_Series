"""Regex-based dental relevance filter."""
from __future__ import annotations

import re

# Conservative dental-relevance regex. Hits anatomy, conditions, procedures, specialties.
DENTAL_TERMS = [
    r"\btooth\b", r"\bteeth\b", r"\bdent\w*\b", r"\bgingiv\w*\b", r"\bperiodont\w*\b",
    r"\bendodont\w*\b", r"\borthodont\w*\b", r"\bprosthodont\w*\b", r"\bpedodont\w*\b",
    r"\bperidont\w*\b", r"\bocclus\w*\b", r"\bmalocclus\w*\b",
    r"\bcari(?:es|ous)\b", r"\bcavit(?:y|ies)\b", r"\benamel\b", r"\bdentin\b",
    r"\bpulpitis\b", r"\bgingivitis\b", r"\bperiodontitis\b", r"\babscess\b",
    r"\bplaque\b", r"\btartar\b", r"\bcalculus\b",
    r"\bcrown\b", r"\bbridge\b", r"\bdenture\b", r"\bveneer\b", r"\bimplant\b",
    r"\bfilling\b", r"\bextraction\b", r"\broot canal\b", r"\bscaling\b",
    r"\bortho(?:dontic|brace|aligner)\w*\b", r"\bbraces\b", r"\baligner\b", r"\binvisalign\b",
    r"\bmolar\b", r"\bpremolar\b", r"\bincisor\b", r"\bcanine tooth\b", r"\bwisdom (?:tooth|teeth)\b",
    r"\bmandib\w*\b", r"\bmaxill\w*\b", r"\bocclusion\b", r"\bbite\b",
    r"\btmj\b", r"\btemporomandibular\b",
    r"\bdental (?:caries|implant|hygien\w*|surgery|crown|abscess|pain|exam|filling|cavity|plaque|prosthesis|fluorosis|trauma|emergency|sealant|x-ray|radiograph|cyst|fluoride)\b",
    r"\boral (?:health|hygiene|cancer|surgery|medicine|pathology|cavity|mucosa|lesion|ulcer|biopsy|examination|prosthesis|leukoplakia|lichen planus|candidiasis|thrush|appliance|disease)\b",
    r"\bmouth (?:ulcer|sore|cancer|disease|breath)\b",
    r"\bgum (?:disease|graft|recession|bleeding|infection|swelling)\b",
    r"\bjaw (?:pain|surgery|fracture|joint)\b",
    r"\bbruxism\b", r"\bxerostomia\b", r"\bhalitosis\b", r"\bsialadenitis\b",
    r"\bcleft (?:lip|palate)\b", r"\bdry mouth\b", r"\btoothache\b",
    r"\bfluorid\w*\b", r"\bdent(?:ist|istry|alschool)\b",
]
DENTAL_RE = re.compile("|".join(DENTAL_TERMS), re.IGNORECASE)


def is_dental(*texts: str) -> bool:
    for t in texts:
        if t and DENTAL_RE.search(t):
            return True
    return False


# PubMed MeSH dental-related descriptors. Matched as substrings in mesh_list strings.
DENTAL_MESH = {
    "dentistry", "tooth", "teeth", "dental", "gingiv", "periodont", "endodont",
    "orthodont", "prosthodont", "oral surg", "stomatognath", "mouth", "oral",
    "caries", "plaque", "calculus", "occlus", "malocclus", "pulpitis", "pulp dis",
    "periapic", "tmj", "temporomand", "implant", "denture", "crown", "veneer",
    "cleft lip", "cleft palate", "salivary", "xerostom", "leukoplakia",
    "lichen planus", "oral candid", "mouth neop", "tongue", "fluoros", "fluorid",
    "bruxism", "halitos", "gingival", "maxillofac", "jaw",
}


def mesh_is_dental(mesh_list) -> bool:
    if not mesh_list:
        return False
    if isinstance(mesh_list, list):
        text = " ".join(str(x).lower() for x in mesh_list)
    else:
        text = str(mesh_list).lower()
    return any(term in text for term in DENTAL_MESH)
