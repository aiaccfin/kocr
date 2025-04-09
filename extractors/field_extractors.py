import re
from typing import List, Dict, Any
from .nlp_utils import extract_org_name

def extract_fields(text: str) -> Dict[str, Any]:
    return {
        "vendor": extract_org_name(text),
        "dates": extract_dates(text),
        "amounts": extract_amounts(text),
        "line_items": extract_line_items(text)
    }

def extract_dates(text: str) -> List[str]:
    patterns = [r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b']
    return [match for p in patterns for match in re.findall(p, text)]

def extract_amounts(text: str) -> List[float]:
    matches = re.findall(r'\$\s*\d+[,.]?\d*', text)
    return [float(m.replace("$", "").replace(",", "")) for m in matches if m]

def extract_line_items(text: str) -> List[Dict[str, Any]]:
    return [{"description": line.strip()} for line in text.split('\n') if re.search(r'\$\s*\d+\.\d{2}', line)]
