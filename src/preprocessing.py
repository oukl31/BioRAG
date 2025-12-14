import csv
import re
from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize
import spacy
import scispacy
import en_ner_bc5cdr_md

nltk.download("punkt")

DATA_DIR = Path("../data")
HGNC_FILE = DATA_DIR / "hgnc_complete_set.txt"

GENE_DICT = {}
nlp_ner = spacy.load("en_ner_bc5cdr_md")


def load_hgnc():
    with open(HGNC_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            symbol = row["symbol"]
            hgnc_id = row["hgnc_id"]
            entrez_id = row.get("entrez_id") or row.get("ncbi_gene_id")

            if not symbol:
                continue

            def _add_key(key):
                key_l = key.strip().lower()
                if not key_l:
                    return
                GENE_DICT[key_l] = {
                    "symbol": symbol,
                    "hgnc_id": hgnc_id,
                    "ncbi_gene_id": entrez_id,
                }

            _add_key(symbol)
            if row.get("alias_symbol"):
                for al in row["alias_symbol"].split("|"):
                    _add_key(al)
            if row.get("prev_symbol"):
                for al in row["prev_symbol"].split("|"):
                    _add_key(al)

load_hgnc()

def find_gene_mentions_with_dict(text: str):
    if not isinstance(text, str):
        return []

    tokens = re.findall(r"[A-Za-z0-9\-]+", text)
    mentions = set()

    for tok in tokens:
        key = tok.lower()
        if key in GENE_DICT:
            mentions.add(tok)

    return list(mentions)


def _clean_mention(text: str) -> str:
    if text is None:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower()


def normalize_gene(mention: str):
    cleaned = _clean_mention(mention)
    info = GENE_DICT.get(cleaned)
    if info is None:
        return {
            "mention": mention,
            "type": "gene",
            "symbol": None,
            "hgnc_id": None,
            "ncbi_gene_id": None,
        }
    return {
        "mention": mention,
        "type": "gene",
        "symbol": info.get("symbol"),
        "hgnc_id": info.get("hgnc_id"),
        "ncbi_gene_id": info.get("ncbi_gene_id"),
    }


def normalize_disease(mention: str):
    cleaned = _clean_mention(mention)
    return {
        "mention": mention,
        "type": "disease",
        "canonical_name": cleaned,
    }


def normalize_entity(ent_text, ent_label):
    if ent_label == "GENE":
        return normalize_gene(ent_text)
    elif ent_label == "DISEASE":
        return normalize_disease(ent_text)
    else:
        return {
            "mention": ent_text,
            "type": ent_label.lower(),
            "normalized": None,
        }


def process_article(article):
    text = article["abstract"]

    if not isinstance(text, str):
        if text is None:
            return []
        text = str(text)

    if not text.strip():
        return []

    sentences = sent_tokenize(text)

    processed_sentences = []
    for sent in sentences:
        doc = nlp_ner(sent)
        genes = []
        diseases = []
        for ent in doc.ents:
            if ent.label_ == "GENE":
                genes.append(normalize_entity(ent.text, ent.label_))
            elif ent.label_ == "DISEASE":
                diseases.append(normalize_entity(ent.text, ent.label_))

        processed_sentences.append(
            {
                "pmid": article["pmid"],
                "title": article["title"],
                "sentence": sent,
                "genes": genes,
                "diseases": diseases,
            }
        )
    return processed_sentences
