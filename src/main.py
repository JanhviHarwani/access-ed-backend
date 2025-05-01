import os
import re
from functools import lru_cache
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from pinecone_manager import PineconeManager
from rag_handler import RAGHandler
from auth import Token, User, authenticate_user, create_access_token, get_current_active_user
from dotenv import load_dotenv
import logging
from contextlib import asynccontextmanager
from datetime import timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
load_dotenv()

# Authentication settings
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Global variables for components
pinecone_manager = None
rag_handler = None

# Lifespan context manager for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up the application...")
    global pinecone_manager, rag_handler
    
    try:
        # Initialize components
        pinecone_manager = PineconeManager()
        rag_handler = RAGHandler()
        logger.info("Components initialized")
    
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise e
    
    yield
    
    # Shutdown
    logger.info("Shutting down the application...")

app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message]

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = None
    source_urls: Optional[List[str]] = None
    source_titles: Optional[List[str]] = None
    is_general_chat: bool = False

class HealthCheck(BaseModel):
    status: str
    version: str

# Dependency to verify components are initialized
async def verify_components():
    if not all([pinecone_manager, rag_handler]):
        raise HTTPException(
            status_code=503,
            detail="Application components not properly initialized"
        )
    return True

# Authentication endpoint
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        version="1.0.0"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    _: bool = Depends(verify_components),
    current_user: User = Depends(get_current_active_user)
):
    """
    Chat endpoint that handles user queries using RAG.
    """
    try:
        logger.info(f"Received chat request from user {current_user.username} with message: {request.message}")
        
        # Check if this is a general chat message
        if rag_handler._is_general_chat(request.message):
            logger.info("Processing general chat message")
            response = rag_handler._handle_general_chat(request.message)
            logger.info(f"General chat response: {response}")
            
            # For general chat, return a response with no sources
            return ChatResponse(
                response=response,
                sources=None,
                source_urls=None,
                source_titles=None,
                is_general_chat=True
            )
        
        # For non-general chat, proceed with normal processing
        results = pinecone_manager.search(request.message)
        
        if not results:
            logger.warning("No results found in Pinecone")
            results = type('obj', (object,), {'matches': []})
        
        # Convert history to list of dicts
        history = [
            {"role": msg.role, "content": msg.content} 
            for msg in request.history
        ]
        
        # Generate response
        response = rag_handler.generate_response(
            query=request.message,
            retrieved_docs=results.matches,
            conversation_history=history
        )
        
        # Extract sources from retrieved documents
        sources = []
        source_urls = []
        source_titles = []
        
        # Get document file paths and metadata
        for doc in results.matches:
            if hasattr(doc, 'metadata'):
                source_path = doc.metadata.get('source')
                source_url = doc.metadata.get('source_url')
                title = doc.metadata.get('title')
                
                if source_path and source_path not in sources:
                    sources.append(source_path)
                if source_url and source_url not in source_urls:
                    source_urls.append(source_url)
                if title and title not in source_titles:
                    source_titles.append(title)
        
        logger.info("Successfully generated response")
        
        # Return response with sources for non-general chat
        return ChatResponse(
            response=response,
            sources=sources if sources else None,
            source_urls=source_urls if source_urls else None,
            source_titles=source_titles if source_titles else None,
            is_general_chat=False
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom exception handler for HTTP exceptions"""
    logger.error(f"HTTP error occurred: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unexpected errors"""
    logger.error(f"Unexpected error occurred: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8000))
    
    logger.info(f"Starting server on 0.0.0.0:{port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )