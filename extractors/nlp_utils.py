import spacy
from core.config import logger

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    logger.warning("SpaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

def extract_org_name(text: str) -> str:
    if not nlp:
        return text.split('\n')[0].strip()
    doc = nlp(text[:1000])
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return orgs[0] if orgs else ""
