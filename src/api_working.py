import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
from dotenv import load_dotenv
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv('REACT_URL', 'http://localhost:3000'),"http://archimedes-7.ics.uci.edu"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ZotGPT API configuration
ZOTGPT_API_KEY = os.getenv('ZOTGPT_API_KEY')
ZOTGPT_DEPLOYMENT_ID = os.getenv('ZOTGPT_DEPLOYMENT_ID', 'gpt-4o')
ZOTGPT_API_VERSION = os.getenv('ZOTGPT_API_VERSION', '2025-02-05')
ZOTGPT_API_BASE = os.getenv('ZOTGPT_API_BASE', 'https://azureapi.zotgpt.uci.edu/openai')
ZOTGPT_API_URL = f"{ZOTGPT_API_BASE}/deployments/{ZOTGPT_DEPLOYMENT_ID}/chat/completions?api-version={ZOTGPT_API_VERSION}"

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message]

class ChatResponse(BaseModel):
    response: str
    source_urls: Optional[List[str]] = None

class HealthCheck(BaseModel):
    status: str
    version: str

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        version="1.0.0"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint using ZotGPT API without RAG.
    """
    try:
        logger.info(f"Received chat request with message: {request.message}")
        
        # Format conversation history for ZotGPT
        messages = [{"role": msg.role, "content": msg.content} for msg in request.history]
        messages.append({"role": "user", "content": request.message})
        
        # Prepare the request payload
        payload = {
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ZOTGPT_API_KEY}"
        }
        
        # Call ZotGPT API
        logger.info(f"Calling ZotGPT API at: {ZOTGPT_API_URL}")
        response = requests.post(ZOTGPT_API_URL, headers=headers, json=payload)
        
        # Check if the request was successful
        if response.status_code != 200:
            logger.error(f"ZotGPT API error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=500,
                detail=f"Error from ZotGPT API: {response.text}"
            )
        
        # Parse the response
        response_data = response.json()
        logger.info(f"ZotGPT API response received")
        
        if "choices" in response_data and len(response_data["choices"]) > 0:
            answer = response_data["choices"][0]["message"]["content"]
            
            return ChatResponse(
                response=answer,
                source_urls=[]  # No sources yet since we're not using RAG
            )
        else:
            logger.error(f"Unexpected response format: {response_data}")
            raise HTTPException(
                status_code=500,
                detail="Received an unexpected response format from the ZotGPT API"
            )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8080))
    
    logger.info(f"Starting server on 0.0.0.0:{port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

