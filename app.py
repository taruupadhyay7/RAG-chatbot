import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from llama_cpp import Llama
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load model
llm = Llama(
    model_path="llama-2-7b-chat.Q3_K_M.gguf",
    n_ctx=2048,
    n_threads=6
)

# Load embeddings and FAISS
embeddings = np.load("embeddings.npy")
with open("texts.json") as f:
    texts = json.load(f)
index = faiss.read_index("faiss.index")

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def retrieve_transactions(query, top_k=3):
    q_emb = model.encode([query]).astype("float32")
    dist, idx = index.search(q_emb, top_k)
    return [texts[i] for i in idx[0]]

def generate_answer(question):
    retrieved = retrieve_transactions(question)

    # Detect customer
    customers = ["Amit", "Riya", "Karan"]
    target = next((c for c in customers if c.lower() in question.lower()), None)

    if target:
        retrieved = [t for t in retrieved if target in t]

    context = "\n".join(retrieved)

    prompt = f"""
Use ONLY this context:

{context}

Question: {question}

Answer:
    """

    output = llm(prompt, max_tokens=200)
    return output["choices"][0]["text"].strip()

# STREAMLIT UI
st.title("🛒 RAG-Powered Retail Chatbot (Offline LLaMA)")
st.write("Ask anything about customer transactions")

question = st.text_input("Your question:")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please type a question!")
    else:
        with st.spinner("Thinking..."):
            answer = generate_answer(question)
        st.success("Answer:")
        st.write(answer)



    

