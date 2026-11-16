import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model only once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load data
df = pd.read_csv("data/papers_clean.csv")

# Load embeddings
embeddings = np.load("models/paper_embeddings.npy")


def recommend_papers(query, top_k=5):

    query_embedding = model.encode([query])

    similarity_scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = similarity_scores.argsort()[-top_k:][::-1]

    results = []

    for idx in top_indices:

        results.append({

            "title": df.iloc[idx]["title"],

            "authors": df.iloc[idx]["authors"],

            "year": df.iloc[idx]["year"],

            "citations": int(df.iloc[idx]["citations"]),

            "similarity": round(similarity_scores[idx] * 100, 2),

            "doi": df.iloc[idx]["doi"]

        })

    return results