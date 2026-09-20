"""
Streamlit UI for the RAG demo (see rag_demo.py for the plain-script version).

Loads the base Qwen2.5-1.5B-Instruct model plus the fine-tuned LoRA adapter
from Hugging Face, and generates an answer from each side by side, so you
can see what fine-tuning actually changed.

Usage:
    pip install -r requirements.txt
    streamlit run app.py
"""

import json

import numpy as np
import streamlit as st
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

CORPUS_FILE = "train.jsonl"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_MODEL = "sajjan0001/qwen2.5-1.5b-trst01-compliance-v2"
TOP_K = 3

st.set_page_config(page_title="Compliance RAG Demo", page_icon="📋")
st.title("📋 Compliance & Sustainability Q&A")
st.caption("RAG over a fine-tuning dataset covering EUDR, DPP, carbon markets, CSRD/ESRS and more.")


@st.cache_resource(show_spinner="Loading models (first run only)...")
def load_everything():
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

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    question_embeddings = embedder.encode(questions, convert_to_numpy=True, normalize_embeddings=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    # Wraps base_model with the LoRA adapter. The base weights are shared,
    # so this does not double memory usage, and model.disable_adapter()
    # lets us fall back to the plain base model for comparison.
    finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
    return questions, answers, embedder, question_embeddings, tokenizer, finetuned_model


questions, answers, embedder, question_embeddings, tokenizer, model = load_everything()


def generate(prompt, use_adapter):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    if use_adapter:
        output_ids = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    else:
        with model.disable_adapter():
            output_ids = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


query = st.text_input("Ask a question", placeholder="e.g. What is EUDR and which companies does it apply to?")

if st.button("Get answer", type="primary") and query:
    with st.spinner("Retrieving sources and generating answers..."):
        # Retrieve the most similar questions from the corpus.
        query_embedding = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = question_embeddings @ query_embedding
        top_idx = np.argsort(-scores)[:TOP_K]

        # Build the grounding context from the retrieved passages.
        context = ""
        for i in top_idx:
            context += f"Q: {questions[i]}\nA: {answers[i]}\n\n"

        # Ask the model to answer using only that context.
        prompt = (
            "You are a compliance and sustainability domain assistant. "
            "Answer the user's question using ONLY the information in the sources below. "
            "If the sources don't contain enough information to answer confidently, say so "
            "explicitly rather than guessing.\n\n"
            f"{context}"
            f"Question: {query}\n"
            "Answer:"
        )

        base_answer = generate(prompt, use_adapter=False)
        finetuned_answer = generate(prompt, use_adapter=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Base model")
        st.write(base_answer)
    with col2:
        st.subheader("Fine-tuned model")
        st.write(finetuned_answer)

    st.subheader("Retrieved sources")
    for i in top_idx:
        with st.expander(f"{questions[i]}  (score={scores[i]:.2f})"):
            st.write(answers[i])
