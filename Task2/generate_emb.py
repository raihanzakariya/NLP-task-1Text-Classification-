import os
import numpy as np
from langchain import hub
from langchain_community.embeddings import HuggingFaceHubEmbeddings
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

df = pd.read_csv("../../Data/dataset.csv")
df = df[["ProductName", "Description"]]
data_emb = {}
for index, row in df.iterrows():
    name, desc = row["ProductName"], row["Description"]
    embs = model.encode(row["Description"])
    data_emb[name] = embs

os.makedirs("embeddings", exist_ok=True)
with open("embeddings/product_embs.pkl", "wb") as f:
    pickle.dump(data_emb, f)
print ("saved embeddings:", "embeddings/product_embs.pkl")
