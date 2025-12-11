import json

with open("transaction.json","r") as file:
    data = json.load(file)
print(data)