

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
            a professional and direct tone. After providing your response, include unique sources with proper 
            link formatting, like this: 'For more information, visit: [source_title](source_url)' 
            using the source URLs from the context. Do not repeat the same URL multiple times."""
            
            # Construct the user prompt with explicit source URL instruction
            user_prompt = f"""Context: {context_data['content']}

Based on the above context, please answer the following question:
{query}

Remember to include 2-3 most relevant source links at the end of your response using markdown link format:
For more information, visit: [Title of Source](actual_url_here)

IMPORTANT: Only include each unique source URL once, even if it appears multiple times in the context.
Each URL should be properly formatted as a markdown link with descriptive text.

You can find the source URLs and titles in this context:
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
            
            response_content = response.choices[0].message.content
            
            # Clean up any potential duplicated or improperly formatted URLs
            response_content = self._clean_source_references(response_content)
            
            return response_content
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error processing your request. Please try again."

    def _clean_source_references(self, response: str) -> str:
        """Clean up source references to ensure unique, properly formatted URLs"""
        
        # Find all URLs in the response
        url_pattern = r'https?://[^\s\)"\']+(?:\([^\s]*\)|[^\s\)\",\']*)'
        all_urls = re.findall(url_pattern, response)
        
        # Get unique URLs
        seen_urls = set()
        duplicates = []
        
        for url in all_urls:
            # Clean up URL - remove trailing punctuation
            clean_url = re.sub(r'[\.\,\;\:\)\]\}]+$', '', url)
            
            # Check if it's a duplicate
            if clean_url in seen_urls:
                duplicates.append(url)
            else:
                seen_urls.add(clean_url)
        
        # Handle cases where URLs appear in both formatted and unformatted ways
        # Look for patterns like: For more information, visit: URL (URL) or For more information, visit: URL(URL)
        for dup in duplicates:
            # Find and remove redundant parenthetical URLs
            pattern = r'\([^\)]*' + re.escape(dup) + r'[^\)]*\)'
            response = re.sub(pattern, '', response)
            
            # Remove duplicate plain URLs that follow a markdown link
            pattern = r'\]\(' + re.escape(dup) + r'\)[\s\n]*' + re.escape(dup)
            response = re.sub(pattern, '](' + dup + ')', response)
        
        # Find sections like "For more information, visit:" followed by bare URLs
        visit_pattern = r'For more information, visit:\s*(https?://[^\s]+)'
        
        def replace_with_markdown_link(match):
            url = match.group(1)
            return f'For more information, visit: [{url}]({url})'
        
        # Replace bare URLs with markdown links
        response = re.sub(visit_pattern, replace_with_markdown_link, response)
        
        return response

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
        seen_urls = set()  # Track unique source URLs
        
        for doc in retrieved_docs:
            if hasattr(doc, 'metadata'):
                content = doc.metadata.get('content', '')
                
                if content:
                    # Extract source URL and title
                    url_match = re.search(r'Source URL: (.*?)\n', content)
                    title_match = re.search(r'Title: (.*?)\n', content)
                    
                    url = url_match.group(1).strip() if url_match else None
                    title = title_match.group(1).strip() if title_match else None
                    
                    # Only add unique URLs
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        
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


