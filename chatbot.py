from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
from llama_cpp import Llama

# Load LLM (local GGUF model)
llm = Llama(
    model_path="llama-2-7b-chat.Q3_K_M.gguf",
    n_ctx=2048,
    n_threads=6,  # adjust based on your CPU
)

# Load FAISS + embeddings
embeddings = np.load("embeddings.npy")
with open("texts.json", "r") as f:
    texts = json.load(f)
index = faiss.read_index("faiss.index")

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Retrieval function
def retrieve_transactions(query, top_k=3):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    results = [texts[i] for i in indices[0]]
    return results

# LLM-based Chat Answer
def answer_question(question):
    retrieved = retrieve_transactions(question)

    print("\n[DEBUG] Retrieved Transactions:")
    for r in retrieved:
        print(" -", r)

    # Known customers
    customers = ["Amit", "Riya", "Karan"]

    # Detect customer from question (case-insensitive)
    target_customer = None
    for c in customers:
        if c.lower() in question.lower():
            target_customer = c

    print("\n[DEBUG] Target Customer:", target_customer)

    # Filter context based on detected customer (if any)
    if target_customer:
        filtered = [t for t in retrieved if target_customer in t]
    else:
        filtered = retrieved

    print("\n[DEBUG] Filtered Context:")
    for f in filtered:
        print(" -", f)

    # If nothing remains after filtering, return a clear message
    if not filtered:
        return "No data found for your query."

    context = "\n".join(filtered)

    # Improved prompt (keeps model focused on context only)
    prompt = f"""
You are a smart retail assistant.

Rules:
1. Answer ONLY using the context provided.
2. If context is empty, say "No data found."
3. Give clear, polite, and natural answers.
4. For spending questions, calculate total step-by-step.
5. Do NOT add any extra transactions not shown in the context.
6. Be concise and accurate.

Context:
{context}

User question: {question}

Your answer:
"""

    output = llm(prompt, max_tokens=300)
    # llama_cpp returns a dict with choices; get text
    return output["choices"][0]["text"].strip()


# Run chatbot
if __name__ == "__main__":
    q = input("Ask your question: ")
    print(answer_question(q))
