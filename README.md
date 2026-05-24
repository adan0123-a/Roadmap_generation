# 🗺️ Career Roadmap Generator

A FastAPI-powered web application that takes a user's skills as input and generates a **personalized, structured career roadmap** using the Groq LLM API with enforced JSON output via Pydantic validation.

---

## ✨ Features

- 🎯 **Skill-based career matching** — enter any skills and get 3–7 realistic career options
- 🛣️ **3-level roadmap** — Beginner → Intermediate → Expert progression steps
- ⚡ **Groq LLM backend** — blazing-fast inference using `llama-3.1-8b-instant`
- 🔒 **Structured output** — Pydantic + Groq `json_object` mode guarantee valid responses
- 🔁 **Singleton Groq client** — one shared client instance across the app lifecycle
- 🌐 **Dual interface** — REST JSON API + HTML form endpoint

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | [FastAPI](https://fastapi.tiangolo.com/) |
| LLM Provider | [Groq](https://groq.com/) (`llama-3.1-8b-instant`) |
| Data Validation | [Pydantic v2](https://docs.pydantic.dev/) |
| Environment Config | `python-dotenv` |
| Server | Uvicorn |

---

## 📁 Project Structure

```
career-roadmap-generator/
├── main.py                  # FastAPI app, routes, Groq logic
├── templates/
│   └── index.html           # Frontend HTML form
├── .env                     # API keys (not committed)
├── .env.example             # Template for environment variables
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/career-roadmap-generator.git
cd career-roadmap-generator
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get your free API key at [console.groq.com](https://console.groq.com)

### 5. Run the server

```bash
python main.py
```

Or with Uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Visit **http://localhost:8000** in your browser.

---

## 🔌 API Reference

### `POST /api/generate-roadmap`

Generate a career roadmap from a JSON request.

**Request body:**
```json
{
  "skills": "Python, machine learning, data visualization, SQL"
}
```

**Response:**
```json
{
  "possible_careers": [
    "Data Scientist",
    "ML Engineer",
    "Data Analyst",
    "AI Research Scientist"
  ],
  "roadmap_for": "Data Scientist",
  "roadmap": {
    "beginner": [
      "Master NumPy and Pandas for data manipulation",
      "Build end-to-end projects using scikit-learn",
      "Learn exploratory data analysis with Matplotlib/Seaborn"
    ],
    "intermediate": [
      "Study deep learning with TensorFlow or PyTorch",
      "Work with large datasets using Spark or BigQuery",
      "Contribute to open-source ML projects on GitHub"
    ],
    "expert": [
      "Design and deploy production ML pipelines",
      "Publish research or technical blog posts",
      "Lead cross-functional data science initiatives"
    ]
  }
}
```

---

### `POST /generate-roadmap`

HTML form submission endpoint (used by the frontend).

**Form field:** `skills` (string)

**Returns:** Same JSON structure as above.

---

### `GET /health`

Check application and client status.

```json
{
  "status": "healthy",
  "api_key_configured": true,
  "singleton_instance_created": true,
  "client_initialized": true
}
```

---

## 🏗️ Architecture Highlights

### Singleton Groq Client

The `GroqClientSingleton` class ensures only **one Groq client** is instantiated for the entire application lifetime, avoiding redundant connections:

```python
groq_singleton = GroqClientSingleton()
client = groq_singleton.get_client()
```

### Structured JSON Output

Groq's `response_format={"type": "json_object"}` combined with a strict system prompt enforces a predictable schema. Pydantic's `RoadmapResponse` model then validates and types the response before it reaches the client.

### System Prompt Design

The system prompt instructs the model to:
- Return **only** a valid JSON object (no markdown, no prose)
- Include 3–7 career options in `possible_careers`
- Set `roadmap_for` to exactly one of those careers
- Populate all three roadmap levels with actionable steps

---

## 📦 requirements.txt

```
fastapi
uvicorn[standard]
groq
pydantic
python-dotenv
```

---

## 🛡️ Error Handling

| Scenario | HTTP Status | Detail |
|---|---|---|
| Empty skills input | `400` | `"Skills cannot be empty"` |
| Groq JSON parse failure | `500` | `"JSON parsing error: ..."` |
| Groq API failure | `500` | `"API error: ..."` |
| Missing API key | App crash at startup | `"GROQ_API_KEY not found"` |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push and open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
