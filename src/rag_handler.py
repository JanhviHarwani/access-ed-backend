from openai import OpenAI
import os
from typing import List, Dict
import re
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGHandler:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def generate_response(self, query: str, retrieved_docs: List[Dict], conversation_history: List[Dict] = None) -> str:
        try:
            # Handle general chat (greetings, thanks, etc.)
            if self._is_general_chat(query):
                return self._handle_general_chat(query)
            
            # If no relevant documents were retrieved
            if not retrieved_docs:
                return "I don't have enough relevant information to answer your question accurately. Could you please rephrase or ask something else?"
                
            # Check relevance score of retrieved documents
            if not self._check_relevance(retrieved_docs):
                return "I don't have enough relevant information to answer your question accurately. Could you please rephrase or ask something else?"
            
            # Prepare context and sources
            context_data = self._prepare_context(retrieved_docs)
            formatted_history = self._format_conversation_history(conversation_history)
            
            # Create system prompt that encourages including source references
            system_prompt = """You are an expert assistant helping educators make education accessible 
            for students with disabilities. Provide accurate, practical information while maintaining 
            a professional and direct tone. After providing your response, ALWAYS end with a line that says 
            'For more information, visit: [source_url]' using the source URLs from the context."""
            
            # Construct the user prompt with explicit source URL instruction
            user_prompt = f"""Context: {context_data['content']}

Based on the above context, please answer the following question:
{query}

Remember to end your response by directing users to the source using this format:
For more information, visit: [actual_url_here]

You can find the source URL in this context:
{context_data['source_info']}"""

            # Generate response using chat completion
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *formatted_history,
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error processing your request. Please try again."

    def _check_relevance(self, retrieved_docs: List[Dict], threshold: float = 0.6) -> bool:
        """Check if retrieved documents are relevant enough"""
        if not retrieved_docs:
            return False
        return any(doc.score > threshold for doc in retrieved_docs)

    def _is_general_chat(self, query: str) -> bool:
        """Detect general chat queries like greetings"""
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        gratitude = ['thanks', 'thank you', 'appreciate']
        farewell = ['bye', 'goodbye', 'see you', 'farewell']
        
        query_lower = query.lower()
        return any(word in query_lower for word in greetings + gratitude + farewell)

    def _handle_general_chat(self, query: str) -> str:
        """Handle general chat queries with appropriate responses"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['hello', 'hi', 'hey']):
            return "How can I assist you with making education more accessible?"
            
        if any(word in query_lower for word in ['thanks', 'thank you']):
            return "You're welcome! Let me know if you have any other questions."
            
        if any(word in query_lower for word in ['bye', 'goodbye']):
            return "Goodbye! Feel free to return if you need more assistance."
        
        return "How can I help you with accessibility-related questions?"

    def _format_conversation_history(self, conversation_history: List[Dict] = None) -> List[Dict]:
        """Format conversation history for the prompt"""
        if not conversation_history:
            return []
            
        formatted_history = []
        for message in conversation_history[-5:]:  # Only use last 5 messages for context
            role = message.get('role', 'user')
            content = message.get('content', '')
            formatted_history.append({
                "role": role,
                "content": content
            })
            
        return formatted_history

    def _prepare_context(self, retrieved_docs: List[Dict]) -> dict:
        """Prepare context and sources from retrieved documents"""
        contexts = []
        source_info = []
        
        for doc in retrieved_docs:
            if hasattr(doc, 'metadata'):
                content = doc.metadata.get('content', '')
                
                if content:
                    # Extract source URL and title
                    url_match = re.search(r'Source URL: (.*?)\n', content)
                    title_match = re.search(r'Title: (.*?)\n', content)
                    
                    url = url_match.group(1).strip() if url_match else None
                    title = title_match.group(1).strip() if title_match else None
                    
                    # Clean the content by removing the metadata header
                    content_lines = content.split('\n')
                    cleaned_content = []
                    skip_until_content = True
                    
                    for line in content_lines:
                        if skip_until_content and line.strip() == 'Content:':
                            skip_until_content = False
                            continue
                        if not skip_until_content and line.strip():
                            cleaned_content.append(line)
                    
                    # Add to contexts and source info
                    if cleaned_content:
                        contexts.append('\n'.join(cleaned_content))
                    if url:
                        source_info.append(f"Source: {title if title else 'Document'} - {url}")

        return {
            'content': '\n\n'.join(contexts),
            'source_info': '\n'.join(source_info)
        }

    def _extract_clean_content(self, content: str) -> str:
        """Extract clean content from the document removing metadata headers"""
        lines = content.split('\n')
        clean_lines = []
        found_content = False
        
        for line in lines:
            if 'Content:' in line:
                found_content = True
                continue
            if found_content and line.strip():
                clean_lines.append(line)
                
        return '\n'.join(clean_lines)