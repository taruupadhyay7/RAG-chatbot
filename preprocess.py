import json

# Load JSON
with open("transaction.json", "r") as file:
    data = json.load(file)

# Convert to descriptive text
texts = []
for t in data:
    sentence = f"On {t['date']}, {t['customer']} purchased a {t['product']} for ₹{t['amount']}."
    texts.append(sentence)

# Print descriptive sentences
for s in texts:
    print(s)
