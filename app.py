import os
import json
from typing import List, Dict
from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Career Roadmap Generator API", version="1.0.0")

 
class GroqClientSingleton:
    """Singleton class to ensure only one instance of Groq client exists"""
    
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GroqClientSingleton, cls).__new__(cls)
            cls._instance._initialize_client()
        return cls._instance
    
    def _initialize_client(self):
        """Initialize the Groq client with API key"""
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self._client = Groq(api_key=groq_api_key)
    
    def get_client(self) -> Groq:
        """Return the singleton client instance"""
        if self._client is None:
            self._initialize_client()
        return self._client
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (useful for testing)"""
        cls._instance = None

# Get the singleton instance
groq_singleton = GroqClientSingleton()
client = groq_singleton.get_client()

# Verify API key  
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Pydantic models for request/response
class SkillsRequest(BaseModel):
    skills: str

class RoadmapResponse(BaseModel):
    possible_careers: List[str]
    roadmap_for: str
    roadmap: Dict[str, List[str]]

SYSTEM_PROMPT = """
You are an expert career and skills development advisor. Your task is to generate a comprehensive career roadmap based on a user's skills.
The output MUST be a valid JSON object with the following structure:
{
  "possible_careers": [
    "Career Title 1",
    "Career Title 2",
    "...",
  ],
  "roadmap_for": "The single career title from possible_careers that this roadmap is best suited for",
  "roadmap": {
    "beginner": [
      "Beginner step 1",
      "Beginner step 2",
      "..."
    ],
    "intermediate": [
      "Intermediate step 1",
      "Intermediate step 2",
      "..."
    ],
    "expert": [
      "Expert step 1",
      "Expert step 2",
      "..."
    ]
  }
}

Rules:
- "possible_careers" must contain 3–7 realistic career options that match the user's skills.
- "roadmap_for" MUST be exactly one of the strings from "possible_careers".
- All steps in "roadmap" MUST be tailored specifically to the "roadmap_for" career and the user's skills.
- All three levels MUST be present, each as a non-empty list of actionable steps.
- Do NOT include any other text, explanations, or code blocks outside the JSON.
"""

def generate_roadmap_data(skills: str) -> dict:
    """Generate roadmap data from Groq API using singleton client"""
    try:
        # Use the singleton client instance
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate a career roadmap for the following skills: {skills}."}
            ],
            response_format={"type": "json_object"}
        )
        
        json_response = completion.choices[0].message.content
        roadmap_data = json.loads(json_response)
        return roadmap_data
    
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON parsing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")

# API Endpoints

import pathlib

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        current_dir = pathlib.Path(__file__).parent
        html_path = current_dir / "templates" / "index.html"
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)

@app.post("/api/generate-roadmap", response_model=RoadmapResponse)
async def generate_roadmap_api(request: SkillsRequest):
    """Generate career roadmap from skills (JSON endpoint)"""
    if not request.skills.strip():
        raise HTTPException(status_code=400, detail="Skills cannot be empty")
    
    roadmap_data = generate_roadmap_data(request.skills)
    return roadmap_data

@app.post("/generate-roadmap")
async def generate_roadmap_form(skills: str = Form(...)):
    """Form endpoint for HTML form submission"""
    if not skills.strip():
        raise HTTPException(status_code=400, detail="Skills cannot be empty")
    
    roadmap_data = generate_roadmap_data(skills)
    return JSONResponse(content=roadmap_data)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check if singleton client is working
    client_instance = groq_singleton.get_client()
    return {
        "status": "healthy", 
        "api_key_configured": bool(groq_api_key),
        "singleton_instance_created": groq_singleton._instance is not None,
        "client_initialized": client_instance is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)