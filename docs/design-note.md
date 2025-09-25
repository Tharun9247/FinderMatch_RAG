FounderMatch RAG – Design Note
1. Overview

FounderMatch RAG is a web application designed to help startup enthusiasts, investors, and entrepreneurs find the most relevant founders from a large dataset using natural language search. The key idea is to make it easy to type a plain-English query like:

"Looking for a technical co-founder with AI experience in healthcare"

…and get a list of founders who best match the request, along with explanations for why they were chosen.

We use AI embeddings to capture the semantic meaning of queries and founder profiles, combined with a hybrid scoring system using TF-IDF and cosine similarity. The goal is to make the matching smart, transparent, and explainable.

2. Dataset Creation

For this project, we generate a synthetic dataset of 700 founder profiles using the scripts/seed.py script. Each profile contains:

id – a unique identifier (UUID)

founder_name – randomly generated realistic names

email – fake but valid-looking email addresses

role – founder, co-founder, or engineer

company – startup name generated randomly

location – city + country

idea – short description of their startup idea

about – a mini-bio with skills, previous experiences, and achievements

keywords – industry-specific terms for quick matching

stage – startup stage (seed, pre-seed, Series A, etc.)

linked_in – fake LinkedIn URLs

notes – optional remarks

This CSV serves as our main database, while a corresponding embeddings.json file stores pre-computed embeddings for semantic search.

3. Matching Process
3.1 AI Embeddings

We use Hugging Face embeddings (sentence-transformers/all-MiniLM-L6-v2) to encode both queries and founder profiles into numerical vectors. These vectors capture semantic meaning, so similar concepts are close in vector space even if the words differ.

Example:

Query: “Looking for a healthcare AI co-founder”

Founder keywords: “digital health, telemedicine, machine learning”

Even though the wording is different, the embeddings detect semantic similarity, allowing a strong match.

3.2 TF-IDF & Cosine Similarity

As a fallback or hybrid approach, we also compute TF-IDF vectors from the text fields in the CSV. Cosine similarity is then used to measure relevance between the query and each profile.

The final matching score combines:

Embedding similarity (weight ~0.8)

Keyword/role/location boost (weight ~0.2)

This hybrid method ensures that even if the embeddings are imperfect, exact matches and important keywords are still considered.

4. Query Processing

When a user enters a query:

The system first generates an embedding vector for the query.

If embeddings are available, it calculates cosine similarity with all stored profile embeddings.

If embeddings are not available or Hugging Face API fails, it falls back to TF-IDF similarity.

Keyword matching is applied in parallel to boost relevant profiles based on role, location, and industry terms.

Scores are combined to produce a final ranked list of founders.

Each result includes a snippet explaining why it matched, the matched fields, and row IDs for full transparency.

5. Deployment
5.1 Why Streamlit?

We chose Streamlit because:

Rapid prototyping for Python applications

Easy to integrate with Pandas, NumPy, and AI libraries

Built-in support for interactive widgets (text input, sliders, expanders)

Streamlit Cloud allows deployment with secrets management (API keys) and HTTPS hosting

Lightweight and production-ready for small datasets

5.2 Local Setup

Install dependencies:

pip install -r requirements.txt


Set Hugging Face API token:

export HUGGINGFACE_API_KEY="your-api-key"


Generate sample data:

python scripts/seed.py


Run the app:

streamlit run streamlit_app.py


Open http://localhost:8501 in your browser.

5.3 Cloud Deployment

Push the repository to GitHub

Create a new app on Streamlit Community Cloud

Configure repository, branch, and main file

Add the HUGGINGFACE_API_KEY in secrets

Click Deploy and get a public URL

6. Limitations & Notes

Due to billing constraints, we cannot use OpenAI embeddings directly. Hugging Face embeddings are free and work offline, but OpenAI embeddings would give slightly better semantic understanding.

The application currently handles 700 founders, but the architecture allows scaling with FAISS or vector databases for larger datasets.

TF-IDF fallback ensures that even if embeddings fail, the search will still return reasonable results.

7. Conclusion

FounderMatch RAG is designed to bridge the gap between natural language queries and structured founder data. By combining embeddings, keyword boosts, and TF-IDF fallback, the system delivers transparent, explainable, and highly relevant matches for startup teams and investors.