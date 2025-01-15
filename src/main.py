# backend/main.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from document_processor import DocumentProcessor
from pinecone_manager import PineconeManager
from rag_handler import RAGHandler
from dotenv import load_dotenv

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv('REACT_URL')],  # Add your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
load_dotenv()
doc_processor = DocumentProcessor()
pinecone_manager = PineconeManager()
rag_handler = RAGHandler()

# Process documents on startup
@app.on_event("startup")
async def startup_event():
    RELOAD_DOCUMENTS = os.getenv('RELOAD_DOCUMENTS', 'false').lower() == 'true'
    
    if RELOAD_DOCUMENTS:
        print("Processing documents and updating Pinecone index...")
        documents = doc_processor.process_documents()
        pinecone_manager.add_documents(documents)
        print(f"Processed and uploaded {len(documents)} document chunks")
    else:
        print("Using existing Pinecone index - skipping document processing")

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message]

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Format conversation history
        conversation_history = "\n".join([
            f"{msg.role}: {msg.content}" 
            for msg in request.history
        ])

        # Get response using RAG
        results = pinecone_manager.search(request.message)
        response = rag_handler.generate_response(
            query=request.message,
            retrieved_docs=results.matches,
            conversation_history=conversation_history
        )
        
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)