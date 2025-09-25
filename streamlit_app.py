import streamlit as st
import json
import pandas as pd
import numpy as np
import re
import os
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import sys

# so that we can import our embeddings helper from scripts folder
SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
sys.path.insert(0, SCRIPT_DIR)
from embeddings_helper import HuggingFaceEmbeddings

# data folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# page look settings
st.set_page_config(
    page_title="FounderMatch RAG",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# some custom css for nicer cards n headers
st.markdown("""
<style>
    .main-header { text-align:center; color:#1f2937; font-size:2.5rem; font-weight:700; margin-bottom:0.5rem; }
    .sub-header { text-align:center; color:#6b7280; font-size:1.2rem; margin-bottom:2rem; }
    .result-card { background:white; border:1px solid #e5e7eb; border-radius:12px; padding:1.5rem; margin:1rem 0; box-shadow:0 1px 3px rgba(0,0,0,0.1); transition:all 0.2s ease; }
    .result-card:hover { box-shadow:0 4px 6px rgba(0,0,0,0.1); border-color:#3b82f6; }
    .founder-name { font-size:1.3rem; font-weight:600; color:#1f2937; margin-bottom:0.5rem; }
    .company-info { color:#3b82f6; font-weight:500; margin-bottom:0.5rem; }
    .location-info { color:#6b7280; margin-bottom:0.5rem; }
    .match-explanation { background:#f3f4f6; border-left:4px solid #10b981; padding:0.75rem; margin:1rem 0; border-radius:0 8px 8px 0; font-style:italic; }
    .provenance { font-size:0.875rem; color:#6b7280; background:#f9fafb; padding:0.5rem; border-radius:6px; margin-top:0.5rem; }
    .search-section { background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); padding:2rem; border-radius:16px; margin:2rem 0; color:white; }
    .search-row { display:flex; gap:0.75rem; align-items:center; width:100%; }
    .search-input { margin:0.25rem 0 0.75rem 0; }
</style>
""", unsafe_allow_html=True)

# main class for search logic
class FounderMatcher:
    def __init__(self):
        self.embeddings_data = None
        self.founders_df = None
        # some synonym mapping for keyword boost
        self.synonym_map = {
            'ai': ['artificial intelligence', 'machine learning', 'ml', 'deep learning'],
            'fintech': ['financial technology', 'finance', 'banking'],
            'healthtech': ['health technology', 'healthcare', 'medical'],
            'founder': ['ceo', 'co-founder', 'cofounder'],
            'engineer': ['developer', 'software engineer', 'technical'],
            'pm': ['product manager', 'product']
        }
        # load csv n embeddings
        self._load_initial_data()
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.tfidf_ids: List[str] = []
        self._ensure_tfidf_index()
        
        # hug face embeddings helper init
        self.hf_embeddings = HuggingFaceEmbeddings()
        self.hf_status = self.hf_embeddings.get_status()
        self.search_method = "unknown"

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0: return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    @st.cache_resource
    def _load_data_cached() -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
        # load embeddings json + csv once, cache it
        try:
            embeddings_data = {}
            emb_path = os.path.join(DATA_DIR, 'embeddings.json')
            if os.path.exists(emb_path):
                with open(emb_path, 'r', encoding='utf-8') as f:
                    embeddings_data = json.load(f)

            csv_path = os.path.join(DATA_DIR, 'founders.csv')
            if not os.path.exists(csv_path): return None, None
            founders_df = pd.read_csv(csv_path)
            return embeddings_data, founders_df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None

    def _load_initial_data(self) -> None:
        self.embeddings_data, self.founders_df = self._load_data_cached()

    def _get_searchable_text(self, row: pd.Series) -> str:
        # combine fields to a big string for tfidf fallback
        parts = []
        if pd.notna(row.get('keywords')): parts.extend([str(row['keywords'])]*3)
        if pd.notna(row.get('role')): parts.extend([str(row['role'])*2])
        if pd.notna(row.get('about')): parts.extend([str(row['about'])*2])
        if pd.notna(row.get('idea')): parts.extend([str(row['idea'])*2])
        parts.append(str(row.get('company', '')))
        parts.append(str(row.get('location', '')))
        parts.append(str(row.get('stage', '')))
        return ' '.join([p for p in parts if p])

    @st.cache_resource(show_spinner=False)
    def _build_tfidf_index_cached(_self, embeddings_data: Dict, founders_df: pd.DataFrame) -> Tuple[TfidfVectorizer, any, List[str]]:
        docs: List[str] = []
        ids: List[str] = []
        use_embed_text = bool(embeddings_data)
        if use_embed_text:
            for row_id, obj in embeddings_data.items():
                text = obj.get('searchable_text', '')
                if text:
                    ids.append(row_id)
                    docs.append(text)
        if not docs and founders_df is not None:
            for _, row in founders_df.iterrows():
                ids.append(row['id'])
                docs.append(_self._get_searchable_text(row))
        if not docs: return TfidfVectorizer(), None, []
        vectorizer = TfidfVectorizer(stop_words='english')
        matrix = vectorizer.fit_transform(docs)
        return vectorizer, matrix, ids

    def _ensure_tfidf_index(self) -> None:
        if self.founders_df is None: return
        vectorizer, matrix, ids = self._build_tfidf_index_cached(self.embeddings_data or {}, self.founders_df)
        self.tfidf_vectorizer = vectorizer
        self.tfidf_matrix = matrix
        self.tfidf_ids = ids

    def get_embedding(self, text: str) -> Optional[List[float]]:
        if not self.hf_status["hf_api_available"]: return None
        try: return self.hf_embeddings.get_embedding(text)
        except Exception: return None

    def calculate_keyword_boost(self, query: str, row: pd.Series) -> Tuple[float, List[str]]:
        boost = 0.0
        matched_fields = []
        query_lower = query.lower()
        if pd.notna(row['keywords']):
            keywords = [k.strip().lower() for k in str(row['keywords']).split(',')]
            for keyword in keywords:
                if keyword and keyword in query_lower:
                    boost += 0.3
                    matched_fields.append(f"keywords: {keyword}")
        role = str(row.get('role', '')).lower()
        if role and role in query_lower:
            boost += 0.2
            matched_fields.append(f"role: {row['role']}")
        else:
            for base, synonym_group in self.synonym_map.items():
                if base == role and any(syn in query_lower for syn in synonym_group):
                    boost += 0.15
                    matched_fields.append(f"role: {row['role']} (synonym match)")
                    break
        if pd.notna(row.get('location')):
            location_lower = str(row['location']).lower()
            if location_lower and location_lower in query_lower:
                boost += 0.2
                matched_fields.append(f"location: {row['location']}")
        query_words = [w for w in query_lower.split() if len(w) > 2]
        two_grams = [f"{query_words[i]} {query_words[i+1]}" for i in range(len(query_words)-1)]
        if pd.notna(row.get('about')):
            about_lower = str(row['about']).lower()
            for phrase in two_grams:
                if phrase in about_lower:
                    boost += 0.1
                    matched_fields.append(f"bio phrase: '{phrase}'")
                    break
        if pd.notna(row.get('idea')):
            idea_lower = str(row['idea']).lower()
            for phrase in two_grams:
                if phrase in idea_lower:
                    boost += 0.1
                    matched_fields.append(f"idea phrase: '{phrase}'")
                    break
        return boost, matched_fields

    def _tfidf_similarity(self, query: str) -> Dict[str, float]:
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None or not self.tfidf_ids: return {}
        qv = self.tfidf_vectorizer.transform([query])
        sims = (self.tfidf_matrix @ qv.T).toarray().ravel()
        return {self.tfidf_ids[i]: float(sims[i]) for i in range(len(self.tfidf_ids))}

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.embeddings_data or self.founders_df is None: return []
        if not query.strip(): return []
        if len(query) > 500: 
            query = query[:500]; st.warning("Query truncated to 500 characters")
        query_embedding = self.get_embedding(query)
        if query_embedding and self.hf_status["hf_api_available"]: self.search_method = "huggingface"
        else:
            self.search_method = "tfidf"
            if not self.hf_status["hf_api_available"]: st.warning("⚠️ Hugging Face API not available. Using TF-IDF fallback.")
        tfidf_scores = {} if query_embedding else self._tfidf_similarity(query)
        results = []
        for _, row in self.founders_df.iterrows():
            row_id = row['id']
            keyword_boost, matched_fields = self.calculate_keyword_boost(query, row)
            embedding_similarity = 0.0
            if query_embedding and row_id in self.embeddings_data:
                row_entry = self.embeddings_data[row_id]
                row_embedding = row_entry.get('embedding')
                if row_embedding:
                    embedding_similarity = self._cosine_similarity(query_embedding, row_embedding)
            tfidf_similarity = 0.0 if query_embedding else tfidf_scores.get(row_id,0.0)
            final_score = (0.8*embedding_similarity + keyword_boost) if query_embedding else (0.8*tfidf_similarity + keyword_boost)
            snippet = self.generate_snippet(query, row, matched_fields)
            result = {
                'id': row_id,
                'founder_name': row.get('founder_name','Unknown'),
                'role': row.get('role','Unknown'),
                'company': row.get('company','Unknown'),
                'location': row.get('location','Unknown'),
                'matched_fields': matched_fields,
                'snippet': snippet + f" — row id {row_id}",
                'score': final_score,
                'full_data': row.to_dict(),
                'search_method': self.search_method
            }
            results.append(result)
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:min(top_k,5)]

    def generate_snippet(self, query: str, row: pd.Series, matched_fields: List[str]) -> str:
        if not matched_fields:
            return f"General match for {row.get('founder_name','Unknown')}, {row.get('role','Unknown')} at {row.get('company','Unknown')}"
        field_summary = matched_fields[:2]
        field_text = '; '.join(field_summary)
        snippet = f"Strong match on {field_text}"
        idea_val = row.get('idea')
        if pd.notna(idea_val) and len(str(idea_val))>10:
            idea_preview = str(idea_val)
            idea_preview = idea_preview[:80]+"..." if len(idea_preview)>80 else idea_preview
            snippet += f". Building: {idea_preview}"
        return snippet

def main():
    st.markdown('<h1 class="main-header">🚀 FounderMatch RAG</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Find the perfect founder matches using natural language search</p>', unsafe_allow_html=True)

    if 'matcher' not in st.session_state:
        with st.spinner('Loading founder database...'): st.session_state.matcher = FounderMatcher()
    matcher = st.session_state.matcher

    if matcher.founders_df is None:
        st.error("❌ Failed to load data. Make sure data/founders.csv and data/embeddings.json exist.")
        st.info("Try running `python scripts/seed.py --rows 12` to generate sample data.")
        return

    st.markdown('<div class="search-section">', unsafe_allow_html=True)
    st.markdown("### 🔍 Describe who you are looking for...")

    row1_col1, row1_col2, row1_col3 = st.columns([6,1.2,1.2], vertical_alignment="center")
    with row1_col1:
        query = st.text_input(
            "Describe who you are looking for…",
            placeholder="e.g., 'Looking for a technical co-founder with AI experience in healthcare'",
            key="search_query",
            label_visibility="collapsed"
        )
    with row1_col2: search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
    with row1_col3: top_k = st.selectbox("Results", [3,4,5], index=0, label_visibility="collapsed")

    st.markdown('</div>', unsafe_allow_html=True)

    if search_clicked or query:
        if not query.strip(): st.warning("⚠️ Please enter a search query."); return
        with st.spinner('🔍 Searching...'): results = matcher.search(query, top_k)
        if not results:
            st.info("🤷‍♂️ No matches found. Try different keywords or location/role details."); return
        st.markdown(f"### 🎯 Found {len(results)} Matches"); st.markdown("---")
        for i,result in enumerate(results,1):
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            col1,col2 = st.columns([3,1])
            with col1:
                st.markdown(f'<div class="founder-name">{result["founder_name"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="company-info">{result["role"]} at {result["company"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="location-info">📍 {result["location"]}</div>', unsafe_allow_html=True)
            with col2: st.metric("Match Score", f"{result['score']:.2f}")
            st.markdown(f'<div class="match-explanation">💡 {result["snippet"]}</div>', unsafe_allow_html=True)
            method_text = "matched via Hugging Face embeddings" if result.get("search_method")=="huggingface" else "matched via TF-IDF fallback"
            st.markdown(f'<div class="provenance">🔍 Row ID: {result["id"]} | Matched fields: {len(result["matched_fields"])} | {method_text}</div>', unsafe_allow_html=True)
            with st.expander(f"📋 Show more details for {result['founder_name']}"):
                full_data = result['full_data']
                col1,col2 = st.columns(2)
                with col1:
                    st.write("**📧 Contact:**"); st.write(f"Email: {full_data.get('email','N/A')}")
                    if pd.notna(full_data.get('linked_in')): st.write(f"LinkedIn: {full_data['linked_in']}")
                    st.write("**🏢 Company Info:**"); st.write(f"Stage: {full_data.get('stage','N/A')}"); st.write(f"Keywords: {full_data.get('keywords','N/A')}")
                with col2:
                    st.write("**💡 About:**"); 
                    if pd.notna(full_data.get('about')): st.write(full_data['about'])
                    st.write("**🚀 Idea:**")
                    if pd.notna(full_data.get('idea')): st.write(full_data['idea'])
                if pd.notna(full_data.get('notes')): st.write("**📝 Notes:**"); st.write(full_data['notes'])
                st.write("**🔍 Matched Fields:**")
                if result['matched_fields']:
                    for field in result['matched_fields']: st.write(f"• {field}")
                else: st.write("• General semantic similarity match")
            st.markdown('</div>', unsafe_allow_html=True); st.markdown("")

    st.markdown("---")
    st.markdown("### ℹ️ About FounderMatch RAG")
    method_status = "✅ Using Hugging Face embeddings (sentence-transformers/all-MiniLM-L6-v2)" if matcher.hf_status["hf_api_available"] else "⚠️ Using TF-IDF fallback (Hugging Face API not available)"
    st.info(f"""
    This app searches {len(matcher.founders_df) if matcher.founders_df is not None else 700} founders using:

    • **Current Method**: {method_status}
    • **Semantic Search**: AI understands your query (HF embeddings or TF-IDF)
    • **Keyword Matching**: Matches roles, locations, industry keywords  
    • **Hybrid Scoring**: Mixes both for relevance
    • **Full Provenance**: Shows why each result matched

    Built with Streamlit • Can deploy on Streamlit Community Cloud
    """)

if __name__ == "__main__":
    main()



