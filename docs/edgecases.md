--> ⚠️ Fallback to TF-IDF + Keyword Scoring (When Hugging Face API Fails)

If the Hugging Face embeddings API is unavailable (missing key, rate limits, network issues), the app automatically switches to fallback mode.

In fallback mode, TF-IDF similarity between the query and founder data is calculated.

Keyword boosts for roles, industries, locations, and synonyms are still applied.

Users are notified with a warning:

⚠️ Hugging Face API not available. Using TF-IDF fallback method.

Ensures no empty results even when embeddings fail.

Allows offline testing and reduces dependency on external API.

Developers can debug and extend embeddings later without breaking the app.

--> Empty or Very Short Queries

If user enters a blank query, the app warns them:

⚠️ Please enter a search query to find founder matches.

Prevents meaningless searches and keeps UI clean.

--> Very Long Queries (>500 chars)

Queries longer than 500 characters are truncated automatically.

User is notified about truncation to avoid silent errors or API issues.

--> No Strong Matches Found

If no results exceed a relevance threshold, app shows a friendly message:

🤷‍♂️ No strong matches found. Try adjusting your search terms or being more specific.

Prevents confusion and guides users to refine searches.

--> Partial or Missing Founder Data

Handles missing fields like about, idea, or LinkedIn gracefully.

Displays “N/A” or skips missing data in UI, avoiding crashes.

--> Synonym Matching & Industry Terms

Recognizes synonyms (e.g., AI, ML, Artificial Intelligence) for roles and keywords.

Increases relevance even if user uses different terminology than the dataset.

--> Multi-Field Matching

Combines multiple fields (keywords, role, idea, about, location) into hybrid scoring.

Ensures better overall match rather than relying on a single field.

--> Offline/Local Deployment

App still works locally with CSV + precomputed embeddings JSON.

TF-IDF fallback ensures users can demo the app without an API key.

--> Real-Time Feedback & Explanations

Each match includes why it was matched, with highlighted fields and snippet explanation.

Increases transparency and trust for users searching for co-founders.