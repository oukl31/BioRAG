from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from .search_pubmed import search_pubmed, fetch_pubmed_details
from .preprocessing import nlp_ner, find_gene_mentions_with_dict, process_article

embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def build_pubmed_query_from_question(user_question: str):
    doc = nlp_ner(user_question)
    gene_terms = []
    disease_terms = []

    for ent in doc.ents:
        if ent.label_ == "DISEASE":
            disease_terms.append(ent.text)

    gene_terms = find_gene_mentions_with_dict(user_question)

    if not gene_terms and not disease_terms:
        return user_question

    parts = []
    if gene_terms:
        gene_query = " OR ".join(f'"{g}"[Title/Abstract]' for g in set(gene_terms))
        parts.append(f"({gene_query})")

    if disease_terms:
        disease_query = " OR ".join(
            f'"{d}"[Title/Abstract]' for d in set(disease_terms)
        )
        parts.append(f"({disease_query})")

    if len(parts) > 1:
        return " AND ".join(parts)
    else:
        return parts[0]


def build_faiss_index(sentences):
    texts = [s["sentence"] for s in sentences]
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return index, embeddings, texts


def search_similar_sentences(query, index, texts, k=5):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb.astype(np.float32), k)
    results = []
    for rank, idx in enumerate(I[0]):
        results.append(
            {
                "rank": rank + 1,
                "score": float(D[0][rank]),
                "text": texts[idx],
            }
        )
    return results


def retrieve_top_k_sentences(user_question, retmax=50, k=5):
    pubmed_query = build_pubmed_query_from_question(user_question)
    print("pubmed query:", pubmed_query)
    pmids = search_pubmed(pubmed_query, retmax=retmax)

    articles = fetch_pubmed_details(pmids)

    all_sentences = []
    for art in articles:
        all_sentences.extend(process_article(art))

    if not all_sentences:
        return [], [], pubmed_query

    index, _, texts = build_faiss_index(all_sentences)
    results = search_similar_sentences(user_question, index, texts, k=k)

    return results, all_sentences, pubmed_query


def format_context_for_llm(rag_results, max_sentences=10):
    lines = []
    for r in rag_results[:max_sentences]:
        lines.append(f"- {r['text']}")
    return "\n".join(lines)
