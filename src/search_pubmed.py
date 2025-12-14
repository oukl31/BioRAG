from Bio import Entrez
from xml.etree import ElementTree as ET

Entrez.email = "your_email@example.com"
Entrez.tool = "your_tool_name"


def search_pubmed(query, retmax=50):
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=retmax,
        sort="relevance",
        usehistory="n",
    )
    record = Entrez.read(handle)
    handle.close()
    pmids = record.get("IdList", [])
    return pmids


def fetch_pubmed_details(pmids):
    if not pmids:
        return []

    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        rettype="abstract",
        retmode="xml",
    )
    xml_data = handle.read()
    handle.close()

    root = ET.fromstring(xml_data)
    articles = []
    for article in root.findall(".//PubmedArticle"):
        pmid_elem = article.find(".//PMID")
        title_elem = article.find(".//ArticleTitle")
        abstract_elem = article.find(".//Abstract/AbstractText")

        pmid = pmid_elem.text if pmid_elem is not None else None
        title = title_elem.text if title_elem is not None else ""
        abstract = abstract_elem.text if abstract_elem is not None else ""

        if pmid:
            articles.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                }
            )
    return articles
