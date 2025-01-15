# src/chat_interface.py

import gradio as gr
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
import os
from dotenv import load_dotenv

class ChatInterface:
    def __init__(self, vector_store):
        load_dotenv()
        
        # Initialize language model
        self.llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )
        
        # Initialize conversation chain
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(),
            memory=self.memory
        )

    def respond(self, message, history):
        """Generate response for user message."""
        try:
            # Get response from chain
            response = self.chain({"question": message})
            
            # Format response with sources
            answer = response['answer']
            return answer
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"

    def create_interface(self):
        """Create and return Gradio interface."""
        interface = gr.ChatInterface(
            self.respond,
            title="Accessibility Education Assistant",
            description="Ask questions about making your teaching more accessible!",
            examples=[
                "How can I make my course materials accessible for blind students?",
                "What accommodations should I consider for students with ADHD?",
                "How do I create accessible PDFs?"
            ]
        )
        return interface