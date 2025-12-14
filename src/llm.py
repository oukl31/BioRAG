import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline

from .retriever import retrieve_top_k_sentences, format_context_for_llm

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
        torch_dtype=dtype,
    ).to(device)

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device in ("cuda", "mps") else -1,
    )
    return gen_pipe

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
    rag_results, meta, used_query = retrieve_top_k_sentences(
        user_question,
        retmax=retmax,
        k=k,
    )

    if not rag_results:
        return "No relevant evidence could be retrieved from PubMed for this question.", used_query, rag_results

    context = format_context_for_llm(rag_results, max_sentences=k)
    prompt = build_llm_prompt(user_question, context)

    llm_pipe = load_llm()
    outputs = llm_pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated = outputs[0]["generated_text"].strip()
    answer = generated
    return answer, used_query, rag_results
