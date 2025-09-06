Skill → Roadmap Generator

🚀 An intelligent career roadmap generator built with Python + Streamlit.
Users enter any skills (with typos, in any style), and the app generates:

📘 Skill Roadmaps (Beginner → Expert)

🌍 Possible Career Paths

🤝 Hybrid Career Paths (e.g., AI + Design → AI-Driven Designer)
Features

Skill Extraction (AI-powered)

Case-insensitive skill matching

Handles typos & synonyms (e.g., js → JavaScript, py → Python)

Career Path Suggestions

Directly from curated roadmaps.json

Hybrid paths when skills overlap across domains

Examples:

Figma + AI → AI-Driven Designer

Web Dev + AI → AI-Enhanced Full-Stack Developer

Mobile Dev + AI → AI-Powered Mobile Developer

Cloud + DevOps → Cloud DevOps Engineer

Roadmap Generation

Structured into Beginner → Intermediate → Advanced → Expert

Combines multiple skills when relevant

Hybrid roadmaps include 5–6 actionable steps per level

Streamlit UI

Simple text box for user skills

Detects intent: single roadmap vs separate roadmaps

Interactive, minimal, and beginner-friendly🛠️ Tech Stack

Python 3.9+

Streamlit
 → interactive UI

spaCy
 → intent detection & NLP

difflib → typo tolerance / fuzzy matching

roadmaps.json → knowledge base of skills, domains, careers

📂 Project Structure

.
├── app.py                # Main Streamlit app

├── roadmaps_fixed.json   # Curated skill → roadmap + careers

├── synonyms.json         # Synonym mappings for skills

├── requirements.txt      # Dependencies

└── README.md    
