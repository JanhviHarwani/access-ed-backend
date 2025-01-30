import os
from functools import lru_cache
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from document_processor import DocumentProcessor
from pinecone_manager import PineconeManager
from rag_handler import RAGHandler
from dotenv import load_dotenv
import logging
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
load_dotenv()

# Global variables for components
doc_processor = None
pinecone_manager = None
rag_handler = None

@lru_cache()
def get_model():
    """Lazy loading of the model"""
    logger.info("Loading SentenceTransformer model...")
    return SentenceTransformer("BAAI/bge-base-en")

# Lifespan context manager for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up the application...")
    global doc_processor, pinecone_manager, rag_handler
    
    try:
        # Initialize basic components first
        doc_processor = DocumentProcessor()
        pinecone_manager = PineconeManager()
        rag_handler = RAGHandler()
        
        # Process documents if RELOAD_DOCUMENTS is true
        if os.getenv('RELOAD_DOCUMENTS', 'false').lower() == 'true':
            logger.info("Processing documents and updating Pinecone index...")
            documents = doc_processor.process_documents()
            pinecone_manager.add_documents(documents)
            logger.info(f"Processed and uploaded {len(documents)} document chunks")
        else:
            logger.info("Using existing Pinecone index - skipping document processing")
            
        logger.info("Basic components initialized")
    
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
    allow_origins=[os.getenv('REACT_URL', 'http://localhost:3000')],
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

class HealthCheck(BaseModel):
    status: str
    version: str

# Dependency to verify components are initialized
async def verify_components():
    if not all([doc_processor, pinecone_manager, rag_handler]):
        raise HTTPException(
            status_code=503,
            detail="Application components not properly initialized"
        )
    return True

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
    _: bool = Depends(verify_components)
):
    """
    Chat endpoint that handles user queries using RAG.
    
    Parameters:
    - request: ChatRequest containing the message and conversation history
    
    Returns:
    - ChatResponse containing the assistant's response and sources
    """
    try:
        logger.info(f"Received chat request with message: {request.message}")
        
        # Ensure model is loaded
        _ = get_model()
        
        # Search for relevant documents
        results = pinecone_manager.search(request.message)
        
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
        for doc in results.matches:
            if hasattr(doc, 'metadata') and doc.metadata.get('source'):
                source = doc.metadata['source']
                if source not in sources:
                    sources.append(source)
        
        logger.info("Successfully generated response")
        
        return ChatResponse(
            response=response,
            sources=sources
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
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unexpected errors"""
    logger.error(f"Unexpected error occurred: {str(exc)}")
    return {
        "error": "An unexpected error occurred",
        "status_code": 500
    }

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