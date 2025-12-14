from Bio import Entrez  # biopython
from xml.etree import ElementTree as ET
from sentence_transformers import SentenceTransformer
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
import torch
import spacy
import scispacy
import en_ner_bc5cdr_md  # ensure package is installed
import faiss
import numpy as np
import re
import csv
from pathlib import Path
import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# ----- 설정 -----
Entrez.email = "your_email@example.com"  # NCBI 요구사항[web:12][web:15]
Entrez.tool = "your_tool_name"

# 임베딩 모델 & NER 모델 로드[web:7][web:10][web:16]
embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
nlp_ner = spacy.load("en_ner_bc5cdr_md")

# HuggingFace 모델 이름[web:197][web:190]
# BIOMISTRAL_MODEL_ID = "BioMistral/BioMistral-7B"
BIOMISTRAL_MODEL_ID = "medalpaca/medalpaca-7b"

def load_llm():
    tokenizer = LlamaTokenizer.from_pretrained(BIOMISTRAL_MODEL_ID)
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
        
    model = LlamaForCausalLM.from_pretrained(
        BIOMISTRAL_MODEL_ID,
        torch_dtype=dtype,      # 경고 메시지가 뜨면 dtype로 바꿔도 됨[web:197]
    ).to(device)

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device in ("cuda", "mps") else -1,
    )
    
    return gen_pipe

llm_pipe = load_llm()

def find_gene_mentions_with_dict(text: str):
    """
    질문 텍스트에서 HGNC 기준 gene 심볼을 찾아 리스트로 반환.
    너무 단순하지만 BRCA1, TP53 같은 심볼에는 잘 작동.
    """
    if not isinstance(text, str):
        return []

    tokens = re.findall(r"[A-Za-z0-9\-]+", text)
    mentions = set()

    for tok in tokens:
        key = tok.lower()
        if key in GENE_DICT:
            mentions.add(tok)

    return list(mentions)


# ----- Pubmed 쿼리 생성 -----
def build_pubmed_query_from_question(user_question: str):
    """
    user_question에서 disease는 bc5cdr NER로,
    gene은 HGNC 사전 기반으로 찾아 PubMed 검색 쿼리 생성.
    """
    doc = nlp_ner(user_question)
    gene_terms = []
    disease_terms = []

    # 1) disease: en_ner_bc5cdr_md가 DISEASE만 지원[web:324][web:325]
    for ent in doc.ents:
        if ent.label_ == "DISEASE":
            disease_terms.append(ent.text)

    # 2) gene: HGNC 사전 기반 매칭
    gene_terms = find_gene_mentions_with_dict(user_question)

    # 아무 것도 못 찾았을 때는 전체 질문을 그대로 term으로 사용
    if not gene_terms and not disease_terms:
        return user_question

    parts = []
    if gene_terms:
        # 예: ("BRCA1"[Title/Abstract] OR "TP53"[Title/Abstract])
        gene_query = " OR ".join(f'"{g}"[Title/Abstract]' for g in set(gene_terms))
        parts.append(f"({gene_query})")

    if disease_terms:
        disease_query = " OR ".join(f'"{d}"[Title/Abstract]' for d in set(disease_terms))
        parts.append(f"({disease_query})")

    # gene AND disease 형태로 결합, 하나만 있으면 그거만
    if len(parts) > 1:
        return " AND ".join(parts)
    else:
        return parts[0]


# ----- PubMed 실시간 호출 -----

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

# ----- 전처리 + NER + normalization (placeholder) -----

# 1) mapping tables 로드
DATA_DIR = Path("./data")

HGNC_FILE = DATA_DIR / "hgnc_complete_set.txt"
GENE_DICT = {}

def load_hgnc():
    """
    HGNC TSV(hgnc_complete_set.txt)를 읽어서
    심볼/alias -> {symbol, hgnc_id, ncbi_gene_id} 매핑 생성.
    """
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
            # alias_symbol, prev_symbol 등도 쉼표로 나눠서 추가 가능
            if row.get("alias_symbol"):
                for al in row["alias_symbol"].split("|"):
                    _add_key(al)
            if row.get("prev_symbol"):
                for al in row["prev_symbol"].split("|"):
                    _add_key(al)

# 프로그램 시작 시 한 번 호출
load_hgnc()

# 2) normalization
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
    """
    label에 따라 gene / disease normalizer 호출.
    gene: 추가 데이터 + bc5cdr
    disease: bc5cdr
    -> bc5cdr이 disease 특화 툴이기 때문
    """
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

# ----- 임베딩 + Faiss 인덱싱 -----

def build_faiss_index(sentences):
    texts = [s["sentence"] for s in sentences]
    embeddings = embed_model.encode(texts, convert_to_numpy=True)  # 768-d[web:7][web:10][web:16]
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return index, embeddings, texts

def search_similar_sentences(query, index, texts, k=5):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb.astype(np.float32), k)
    results = []
    for rank, idx in enumerate (I[0]):
        results.append(
            {
                "rank": rank + 1,
                "score": float(D[0][rank]),
                "text": texts[idx],
            }
        )
    return results

# ----- 전체 파이프라인: 실시간 PubMed → RAG용 컨텍스트 -----

def retrieve_top_k_sentences(user_question, retmax=50, k=5):
    # 1) PubMed 검색 (실시간)
    pubmed_query = build_pubmed_query_from_question(user_question)
    print("pubmed query: ", pubmed_query)
    pmids = search_pubmed(pubmed_query, retmax=retmax)

    # 2) 논문 메타데이터 & abstract 실시간 fetch
    articles = fetch_pubmed_details(pmids)

    # 3) 문장 단위 전처리 + NER + normalization
    all_sentences = []
    for art in articles:
        all_sentences.extend(process_article(art))

    if not all_sentences:
        return []

    # 4) 임베딩 + Faiss 인덱스
    index, _, texts = build_faiss_index(all_sentences)

    # 5) 사용자의 질문에 대해 semantic search
    results = search_similar_sentences(user_question, index, texts, k=k)

    # LLM에는 results와 대응되는 메타데이터(all_sentences[idx])를 합쳐서 전달하면 됨
    return results, all_sentences, pubmed_query

def format_context_for_llm(rag_results, max_sentences=10):
    """
    retrieve_top_k_sentences_from_question 의 results(meta 아님)를 받아서
    LLM에 넣을 컨텍스트 문자열로 변환.
    """
    lines = []
    for r in rag_results[:max_sentences]:
        # r["text"] 만 써도 되고, pmid/title 등을 같이 넣어도 됨
        lines.append(f"- {r['text']}")
    return "\n".join(lines)


def build_llm_prompt(user_question: str, context: str) -> str:
    return (
        "You are a biomedical expert.\n"
        "Use only the following PubMed sentences to answer the question.\n"
        "If the evidence is insufficient, say that the answer is uncertain.\n\n"
        f"Question:\n{user_question}\n\n"
        f"Relevant sentences:\n{context}\n\n"
        "Answer:"
    )

def answer_with_rag_llm(user_question: str, retmax=50, k=10, max_new_tokens=64):
    # 1) RAG: PubMed → NER/임베딩 → semantic search
    rag_results, meta, used_query = retrieve_top_k_sentences(
        user_question,
        retmax,
        k=k,
    )

    if not rag_results:
        return "No relevant evidence could be retrieved from PubMed for this question."

    # 2) 컨텍스트 포맷팅
    context = format_context_for_llm(rag_results, max_sentences=k)

    # 3) 프롬프트 구성
    prompt = build_llm_prompt(user_question, context)

    # 4) LLM 호출[web:197][web:190]
    outputs = llm_pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated = outputs[0]["generated_text"].strip()
    answer=generated
    return answer, used_query, rag_results

# 사용 예시
if __name__ == "__main__":
    # pubmed_query = "BRCA1"
    user_question = "What disease are associated with BRCA1?"
    answer, pubmed_query, rag_results = answer_with_rag_llm(user_question, retmax=20, k=5)
    # for r in results:
    #     print(r["rank"], r["score"], r["text"])
    print("PubMed query used:", pubmed_query)
    print("RAG-based answer:\n", answer)
