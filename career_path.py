import json
import re
from pathlib import Path
from difflib import get_close_matches
from collections import OrderedDict

import streamlit as st
import spacy  # spaCy for intent detection
from sentence_transformers import SentenceTransformer, util
import torch

# --- Paths ---
ROADMAPS_PATH = Path("roadmaps_fixed.json")
SYNONYMS_PATH = Path("synonyms.json")

# --- Helpers ---
def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# --- Load Data ---
ROADMAPS = load_json(ROADMAPS_PATH)
if ROADMAPS is None:
    st.error(f"Missing {ROADMAPS_PATH}. Put your roadmaps JSON in the app directory.")
    st.stop()

SYNONYMS = load_json(SYNONYMS_PATH)

# --- Auto-generate synonyms if missing ---
def generate_synonyms_from_skills(skills):
    synonyms = {}
    for skill in skills:
        norm = skill.strip()
        low = norm.lower()
        synonyms[low] = norm
        no_nums = re.sub(r"\d+", "", low).strip()
        if no_nums and no_nums != low:
            synonyms[no_nums] = norm
        if " " in low:
            acronym = "".join(w[0] for w in low.split() if w and w[0].isalpha())
            if 1 < len(acronym) <= 6:
                synonyms[acronym] = norm
        if "javascript" in low:
            synonyms["js"] = norm
        if "typescript" in low:
            synonyms["ts"] = norm
        if "python" in low:
            synonyms["py"] = norm
        if "postgresql" in low:
            synonyms["postgres"] = norm
            synonyms["pgsql"] = norm
        if "mysql" in low:
            synonyms["maria"] = norm
        if "artificial intelligence" in low:
            synonyms["ai"] = norm
        if "machine learning" in low:
            synonyms["ml"] = norm
        if "natural language processing" in low:
            synonyms["nlp"] = norm
        if "computer vision" in low:
            synonyms["cv"] = norm
    return synonyms

if SYNONYMS is None:
    SYNONYMS = generate_synonyms_from_skills(list(ROADMAPS.keys()))
    save_json(SYNONYMS, SYNONYMS_PATH)

SKILLS = list(ROADMAPS.keys())
SKILLS_LOWER = {s.lower(): s for s in SKILLS}
SYN_LOWER = {k.lower(): v for k, v in SYNONYMS.items()}

STOPWORDS = {"in", "on", "at", "it", "an", "to", "by", "of", "for", "and", "or", "the", "a"}

# --- spaCy Intent Detection ---
nlp = spacy.load("en_core_web_sm")

def detect_intent_spacy(user_text: str) -> str:
    if not user_text or not user_text.strip():
        return "default"

    doc = nlp(user_text.lower())
    single_keywords = {"single", "one", "combined", "merge"}
    separate_keywords = {"separate", "different", "individual", "each"}

    for token in doc:
        if token.text in single_keywords:
            return "single"
        if token.text in separate_keywords:
            return "separate"

    for chunk in doc.noun_chunks:
        if "single roadmap" in chunk.text or "one roadmap" in chunk.text:
            return "single"
        if "separate" in chunk.text or "different" in chunk.text:
            return "separate"

    for token in doc:
        if token.lemma_ in {"merge", "combine", "integrate"}:
            return "single"
        if token.lemma_ in {"split", "divide"}:
            return "separate"

    return "default"

# --- Career Path Logic ---
def suggest_career_names(skills):
    careers = []
    for skill in skills:
        skill = skill.lower()
        if skill in ROADMAPS and "careers" in ROADMAPS[skill]:
            careers.extend(ROADMAPS[skill]["careers"])
    return list(dict.fromkeys(careers))  # remove duplicates

def suggest_combined_career(skills, domains):
    domains_lower = [d.lower() for d in domains]

    # --- UI/UX + Development Hybrids ---
    if "ui/ux design" in domains_lower and "frontend" in domains_lower:
        return "Frontend Developer with Design Expertise"
    if "ui/ux design" in domains_lower and "backend & web frameworks" in domains_lower:
        return "Full-Stack Developer with UX Focus"
    if "ui/ux design" in domains_lower and "programming languages" in domains_lower:
        return "Design Technologist"

    # --- UI/UX + AI Hybrids ---
    if "ui/ux design" in domains_lower and "programming languages" in domains_lower and any("ai" in d for d in domains_lower):
        return "AI-Driven Designer"

    # --- Development + AI Hybrids ---
    if "backend & web frameworks" in domains_lower and any("ai" in d for d in domains_lower):
        return "AI-Enhanced Full-Stack Developer"
    if "frontend" in domains_lower and any("ai" in d for d in domains_lower):
        return "AI-Powered Frontend Engineer"

    # --- Mobile + AI Hybrids ---
    if "mobile development" in domains_lower and any("ai" in d for d in domains_lower):
        return "AI-Powered Mobile Developer"

    # --- Cloud/DevOps Hybrids ---
    if "cloud computing" in domains_lower and "devops" in domains_lower:
        return "Cloud DevOps Engineer"

    # --- Web3 ---
    if "blockchain development" in domains_lower and "web development" in domains_lower:
        return "Web3 Developer"

    # --- AI Research ---
    if "ai" in domains_lower and "data science" in domains_lower:
        return "AI Research Scientist"

    # --- AI Infrastructure ---
    if "ai" in domains_lower and "cloud computing" in domains_lower:
        return "AI Infrastructure Engineer"

    return None

# --- Hybrid Roadmap Builder (shortened for brevity) ---
def build_hybrid_roadmap(skills, career_name):
    roadmap = OrderedDict()
    roadmap["Beginner"] = ["Learn basics of each selected skill", "Project: simple hybrid prototype"]
    roadmap["Intermediate"] = ["Combine skills into projects", "Project: hybrid application"]
    roadmap["Advanced"] = ["Master frameworks across domains", "Project: large-scale hybrid system"]
    roadmap["Expert"] = ["Lead innovation in hybrid domain", "Mentor others in hybrid specialization"]
    return roadmap

# --- Embeddings Model ---
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
skills = list(ROADMAPS.keys())
skill_embeddings = embed_model.encode(skills, convert_to_tensor=True)

# --- Extraction Logic (merged) ---
def extract_skills(user_text: str, threshold: float = 0.6):
    found = []

    if not user_text or not user_text.strip():
        return []

    text_lower = user_text.lower()

    # --- Keyword & Synonym Matching ---
    for skill in SKILLS:
        if " " in skill.lower() and skill.lower() in text_lower:
            if skill.lower() not in found:
                found.append(skill.lower())

    tokens = re.split(r'[,\n; ]+', text_lower)
    tokens = [tk.strip() for tk in tokens if tk.strip()]

    for tk in tokens:
        if not tk or tk in STOPWORDS:
            continue

        if tk in SYN_LOWER:
            mapped = SYN_LOWER[tk].lower()
            if mapped in ROADMAPS and mapped not in found:
                found.append(mapped)
            continue

        if tk in SKILLS_LOWER:
            mapped = SKILLS_LOWER[tk].lower()
            if mapped not in found:
                found.append(mapped)
            continue

        if len(tk) >= 3:
            match = get_close_matches(tk, SKILLS_LOWER.keys(), n=1, cutoff=0.75)
            if match:
                mapped = SKILLS_LOWER[match[0]].lower()
                if mapped not in found:
                    found.append(mapped)

    # --- Embeddings Matching ---
    user_embedding = embed_model.encode(user_text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_embedding, skill_embeddings)[0]

    for skill, score in zip(skills, cosine_scores):
        if float(score) >= threshold and skill.lower() not in found:
            found.append(skill.lower())

    return found

# --- Merge multiple roadmaps ---
def merge_roadmaps(skills):
    levels = ["beginner", "intermediate", "advanced", "expert"]
    merged = OrderedDict()
    for lvl in levels:
        merged[lvl.capitalize()] = []
        seen = set()
        for skill in skills:
            if skill not in ROADMAPS:
                continue
            steps = ROADMAPS[skill].get(lvl, [])
            for s in steps:
                s_norm = s.strip()
                if s_norm and s_norm not in seen:
                    merged[lvl.capitalize()].append(f"{s_norm}  — ({skill})")
                    seen.add(s_norm)
    return merged

# --- Streamlit UI ---
st.set_page_config(page_title="Skill → Roadmap", layout="wide")
st.title("Skill → Roadmap")

user_input = st.text_area(
    "Enter your skills (paragraph, comma-separated, or a sentence)",
    height=140,
    placeholder="e.g. I'm experienced in python, javascript, and figma"
)

if st.button("Generate roadmap"):
    skills = extract_skills(user_input)
    domains = [ROADMAPS[s].get("domain", "").lower() for s in skills if s in ROADMAPS]

    st.markdown(f"🔎 **Detected skills:** {', '.join(skills) if skills else 'None'}")

    intent = detect_intent_spacy(user_input)
    want_single = True if intent == "single" else False if intent == "separate" else True

    if want_single:
        combined_career = suggest_combined_career(skills, domains)

        if combined_career:
            st.subheader(f"🌍 Hybrid Career Path: {combined_career}")
            roadmap = build_hybrid_roadmap(skills, combined_career)
            for lvl, steps in roadmap.items():
                st.markdown(f"**{lvl}**")
                for step in steps:
                    st.write(f"- {step}")
        else:
            merged = merge_roadmaps(skills)
            career_names = suggest_career_names(skills)

            st.subheader("🌍 Possible Career Paths:")
            if career_names:
                for c in career_names:
                    st.markdown(f"- {c}")
            else:
                st.write("No specific career paths found for these skills.")

            st.subheader("📘 Roadmap")
            for lvl, steps in merged.items():
                st.markdown(f"**{lvl}**")
                for step in steps:
                    st.write(f"- {step}")
    else:
        st.subheader("📘 Individual Skill Roadmaps")
        if not skills:
            st.write("No skills detected.")
        for skill in skills:
            if skill in ROADMAPS:
                st.markdown(f"### {skill} — {ROADMAPS[skill].get('domain','')}")
                for lvl, steps in ROADMAPS[skill].items():
                    if lvl in {"domain", "careers"}:
                        continue
                    st.markdown(f"**{lvl.capitalize()}**")
                    for s in steps:
                        st.write(f"- {s}")
            else:
                st.warning(f"No roadmap found for {skill}")
