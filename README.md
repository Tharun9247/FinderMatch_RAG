FounderMatch RAG

A smart web app to find the best founder matches from a huge database using natural language search. Uses Hugging Face embeddings + keyword hybrid ranking to show the most relevant results with explanations.






--> Tech Stack

Tech Stack

Streamlit: Web framework and hosting

Hugging Face API: Used for semantic embeddings and natural language search. (We originally wanted OpenAI embeddings because they sometimes give slightly better results, but due to billing and cost limits we opted for Hugging Face.)

Pandas: Data manipulation and CSV handling

NumPy: Numerical computations and similarity scoring

scikit-learn: Cosine similarity and TF-IDF fallback

Faker: Synthetic data generation for realistic profiles





--> Features

Natural Language Search: Type in plain English and find founders easily

Hybrid Matching: Combines AI embeddings from Hugging Face and keyword matching for better results

Explains Matches: Each result tells why it was selected

Full Provenance: See matched fields and row IDs for transparency

Rich Founder Profiles: Expand each result to see full info

Streamlit Ready: Optimized for Streamlit Community Cloud

--> Quick Start
Local Dev Setup

Clone repo & install deps:

pip install -r requirements.txt


Set your Hugging Face token:

export HUGGINGFACEHUB_API_TOKEN="your-huggingface-token"


Generate or seed the founder DB:

python scripts/seed.py


Run the app:

streamlit run streamlit_app.py


Open browser: Go to http://localhost:8501

Deploy on Streamlit Cloud

Push repo to GitHub

Create a new app on share.streamlit.io

Select repo, branch (main) & main file (streamlit_app.py)

Add your Hugging Face token in Secrets:

HUGGINGFACEHUB_API_TOKEN = "your-token"


Deploy and share the HTTPS URL

--> Sample Dataset Preview
id,founder_name,email,role,company,location,idea,about,keywords,stage,linked_in,notes
550e8400-e29b-41d4-a716-446655440000,Sarah Chen,sarah.chen@example.com,Founder,VisionAI,San Francisco USA,"AI-powered remote health monitoring platform.",Former PM at Google with 8 yrs in healthtech.,healthtech ai telemedicine machine learning startup,seed,https://linkedin.com/in/sarahchen,Fundraising actively

7c9e6679-7425-40de-944b-e07fc1f90ae7,Marcus Rodriguez,marcus.rodriguez@example.com,Co-founder,FinFlow,Austin USA,"Embedded payments infra for B2B platforms in LATAM.",Ex-Senior Engineer at Stripe, fintech payments blockchain,pre-seed,https://linkedin.com/in/marcusrodriguez,Open for co-founder

8fa94d2e-8b47-4e08-9c5f-6d4e5c3a2b1c,Dr. Priya Patel,priya.patel@example.com,Founder,BioNexus,Boston USA,"AI platform for faster drug discovery.",Stanford grad turned founder, biotech pharma AI machine learning,series A,https://linkedin.com/in/priyapatel,Seeking mentors

--> Architechure

Frontend: Streamlit with custom CSS for nice UI

Search Engine: Hybrid search using Hugging Face embeddings + keyword matching

Database: CSV + JSON

Embeddings: Precomputed using Hugging Face sentence-transformers/all-MiniLM-L6-v2

Hosting: Streamlit Community Cloud

--> Example Queries

"Looking for AI co-founder in healthcare"

"Find fintech founders with payments experience"

"Biotech entrepreneurs for pharma startups"

"SaaS co-founder needed in Europe"

"Clean energy startup mentors"

--> Project Structure
foundermatch-rag/
├── streamlit_app.py        # Main Streamlit app
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/
│   ├── founders.csv        # Founder database
│   └── embeddings.json     # Precomputed embeddings
├── scripts/
│   └── seed.py             # Seed / build embeddings
└── docs/
    └── design-note.md      # Tech notes & design



--> Secrets

Set this locally:

export HUGGINGFACEHUB_API_TOKEN="your-token"


Streamlit Cloud secrets:

HUGGINGFACEHUB_API_TOKEN = "your-token"

--> Performance

Fast search (<1s for 700 rows)

~50MB RAM for embeddings

Optimized for Streamlit limits

First load: 2-3s





-->Instructions to run this application on local system
Follow these steps to run FinderMatch RAG locally:

-Clone the repository

-git clone https://github.com/Tharun9247/FinderMatch_RAG.git
cd FinderMatch_RAG


-Create and activate a virtual environment

# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate


-Install dependencies

-pip install -r requirements.txt


-Set your Hugging Face API token

# Windows PowerShell
$Env:HF_API_TOKEN="YOUR_HF_API_TOKEN"

# macOS/Linux
export HF_API_TOKEN="YOUR_HF_API_TOKEN"


-Run the Streamlit app

-streamlit run app.py


-Open your browser and navigate to

-http://localhost:8501


-Use the app by entering queries to search for founders.