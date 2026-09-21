"""
Quick RAG test over my training corpus.

I wanted to see if grounding answers in the actual train.jsonl passages
(pulled by embedding similarity) helps vs just trusting whatever the
fine-tuned model memorized. Pulls the top-k closest Q&A pairs and stuffs
them into the prompt as context.

    pip install -U sentence-transformers transformers accelerate peft torch
    python rag_demo.py

Needs a GPU box, or change device="cpu" below. Point MODEL_PATH at my
saved LoRA dir if I want to test the fine-tuned adapter instead of base.
"""

import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

CORPUS_FILE = "train.jsonl"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"  # swap this for my fine-tuned adapter dir when testing that
TOP_K = 3

# reusing the training examples themselves as the retrievable passages
questions = []
answers = []
with open(CORPUS_FILE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        questions.append(obj["messages"][0]["content"])
        answers.append(obj["messages"][1]["content"])
print(f"Loaded {len(questions)} passages.")

print(f"Loading embedding model {EMBEDDING_MODEL} ...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
question_embeddings = embedder.encode(questions, convert_to_numpy=True, normalize_embeddings=True)

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
    query_embedding = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = question_embeddings @ query_embedding
    top_idx = np.argsort(-scores)[:TOP_K]

    context = ""
    for i in top_idx:
        context += f"Q: {questions[i]}\nA: {answers[i]}\n\n"

    prompt = (
        "You are a compliance and sustainability domain assistant. "
        "Answer the user's question using ONLY the information in the sources below. "
        "If the sources don't contain enough information to answer confidently, say no "
        "explicitly rather than guessing.\n\n"
        f"{context}"
        f"Question: {query}\n"
        "Answer:"
    )
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=250, do_sample=False)
    answer = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("-" * 80)
    for i in top_idx:
        print(f"[Retrieved, score={scores[i]:.2f}] {questions[i]}")
    print("-" * 80)
    print(f"ANSWER: {answer}")
