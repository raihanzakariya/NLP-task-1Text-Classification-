
import pickle
from sklearn.metrics.pairwise import cosine_similarity


embeddings = None
def load_emb(fpath="embeddings/product_embs.pkl"):
    """functiont load saved embeddings"""
    global embeddings
    with open(fpath, "rb") as f:
        embeddings = pickle.load(f)

def cosine_function(emb1, emb2):
    """function to return cosine similarity between two embeddings"""
    score = cosine_similarity(emb1.reshape(-1, 1), emb2.reshape(-1, 1))[0][0]
    return score

def similar_products(prod_name, top_k=10):
    """function to find top-k production based on cosine similarity of embeddings"""
    global embeddings
    embedding = embeddings[prod_name]
    matches = []
    for name, embs in embeddings.items():
        if name != prod_name:
            score = cosine_function(embedding, embs)
            matches.append([name, score])
    matches = sorted(matches, key=lambda x:x[1])
    top_k = min(len(matches), top_k)
    top_k = matches[:top_k]
    return [x[0] for x in top_k]


if __name__=="__main__":
    load_emb()
    inputs = "\"Prada Striped Shell Belt Bag\""
    print ("input:", inputs)
    prods = similar_products(inputs)
    print ("similar prds:", prods)
