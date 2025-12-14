from .llm import answer_with_rag_llm


if __name__ == "__main__":
    user_question = "What disease are associated with BRCA1?"
    
    answer, pubmed_query, rag_results = answer_with_rag_llm(user_question, retmax=20, k=5,)
    print("PubMed query used:", pubmed_query)
    print("RAG-based answer:\n", answer)