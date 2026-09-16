"""
Minimal RAG demo over the domain corpus used for fine-tuning.

Retrieves the most relevant Q&A passages from train.jsonl via embedding
similarity, then feeds them as grounding context to the LLM so answers are
sourced from retrievable text rather than relying solely on what the model
memorized during fine-tuning. Fine-tuning teaches the model the domain's
vocabulary and reasoning style; RAG keeps its answers current and citable.

Usage:
    pip install -U sentence-transformers transformers accelerate peft torch
    python rag_demo.py

Run this on a machine with a GPU (or adjust device="cpu" below) and, if you
want to query the fine-tuned adapter instead of the base model, set
MODEL_PATH to your saved LoRA output directory.
"""

import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

CORPUS_FILE = "train.jsonl"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"  # swap for your fine-tuned adapter/merged model dir
TOP_K = 3


def load_corpus(path: str):
    questions, answers = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj["messages"][0]["content"]
            a = obj["messages"][1]["content"]
            questions.append(q)
            answers.append(a)
    return questions, answers


def build_index(questions, embedder):
    embeddings = embedder.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def retrieve(query, questions, answers, embeddings, embedder, top_k=TOP_K):
    query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = embeddings @ query_vec
    top_idx = np.argsort(-scores)[:top_k]
    return [(questions[i], answers[i], float(scores[i])) for i in top_idx]


def build_prompt(query, retrieved):
    context = "\n\n".join(
        f"Source {i+1} (similarity {score:.2f}):\nQ: {q}\nA: {a}"
        for i, (q, a, score) in enumerate(retrieved)
    )
    return (
        "You are a compliance and sustainability domain assistant. "
        "Answer the user's question using ONLY the information in the sources below. "
        "If the sources don't contain enough information to answer confidently, say so "
        "explicitly rather than guessing.\n\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def generate(model, tokenizer, prompt, max_new_tokens=250):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)


def main():
    print(f"Loading corpus from {CORPUS_FILE} ...")
    questions, answers = load_corpus(CORPUS_FILE)
    print(f"Loaded {len(questions)} passages.")

    print(f"Loading embedding model {EMBEDDING_MODEL} ...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = build_index(questions, embedder)

    print(f"Loading generation model {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    demo_queries = [
        "What geolocation precision does EUDR require for small farm plots?",
        "How is a Digital Product Passport different from a certificate of origin?",
        "Why does mass balance chain of custody allow mixing of certified and non-certified material?",
    ]

    for query in demo_queries:
        retrieved = retrieve(query, questions, answers, embeddings, embedder)
        prompt = build_prompt(query, retrieved)
        answer = generate(model, tokenizer, prompt)

        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("-" * 80)
        for i, (q, a, score) in enumerate(retrieved):
            print(f"[Retrieved {i+1}, score={score:.2f}] {q}")
        print("-" * 80)
        print(f"ANSWER: {answer}")


if __name__ == "__main__":
    main()
