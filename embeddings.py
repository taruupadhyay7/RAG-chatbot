from sentence_transformers import SentenceTransformer
import json
import numpy as np
import faiss

# Step 1: Load transaction JSON
with open("transaction.json", "r") as file:
    data = json.load(file)

# Step 2: Convert each transaction to descriptive text
texts = []
for t in data:
    sentence = f"On {t['date']}, {t['customer']} purchased a {t['product']} for ₹{t['amount']}."
    texts.append(sentence)

# Step 3: Load Embedding Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 4: Create Embeddings
embeddings = model.encode(texts)

# Step 5: Convert to float32 (FAISS requirement)
embeddings = np.array(embeddings).astype("float32")

# Step 6: Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save embeddings and index
np.save("embeddings.npy", embeddings)
with open("texts.json", "w") as f:
    json.dump(texts, f)

faiss.write_index(index, "faiss.index")

print("Embeddings + FAISS index created successfully!")
