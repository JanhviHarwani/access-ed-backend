import requests
import os
from typing import List, Dict
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGHandler:
    def __init__(self):
        self.api_key = os.getenv('ZOTGPT_API_KEY')  # Ensure this environment variable is set
        self.deployment_id = os.getenv('ZOTGPT_DEPLOYMENT_ID')  # Set this in your environment
        self.api_url = f"https://azureapi.zotgpt.uci.edu/openai/deployments/{self.deployment_id}/chat/completions?api-version=2024-02-01"

    def generate_response(self, query: str, retrieved_docs: List[Dict], conversation_history: List[Dict] = None) -> str:
        """Generate a response to the user's query"""
        try:
            # Handle general chat with direct responses, completely bypassing API
            if self._is_general_chat(query):
                logger.info("Handling general chat query with direct response")
                direct_response = self._handle_general_chat(query)
                logger.info(f"Returning direct response: {direct_response}")
                return direct_response
            
            # For non-general chat, proceed with normal processing
            if not retrieved_docs:
                logger.info(f"Query '{query}' did not retrieve any documents")
                return ("I apologize, but I couldn't find any relevant information in my knowledge base. "
                       "I'm specifically trained to help with educational accessibility topics. "
                       "Could you please ask a question related to making education more accessible for students with disabilities?")
            
            # Process normal educational accessibility queries
            context_data = self._prepare_context(retrieved_docs)
            formatted_history = self._format_conversation_history(conversation_history)
            
            system_prompt = """You are an expert assistant helping educators make education accessible 
            for students with disabilities. Provide accurate, practical information while maintaining 
            a professional and direct tone. If the question is not directly related to educational accessibility,
            try to connect it to accessibility principles or explain why it's outside your expertise."""
            
            source_url = context_data['source_urls'][0] if context_data['source_urls'] else None
            source_citation = f"\n\nFor more information, visit: [{source_url}]({source_url})" if source_url else ""
            
            user_prompt = f"""Context: {context_data['content']}

Based on the above context, please answer the following question:
{query}

Your response must end exactly with this: {source_citation}"""
            
            # Make API call
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    *formatted_history,
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key,
                "Cache-Control": "no-cache"
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            if response.status_code != 200:
                logger.error(f"ZotGPT API error: {response.status_code} - {response.text}")
                return "I apologize, but I encountered an error connecting to the language model. Please try again later."
            
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                response_content = response_data["choices"][0]["message"]["content"]
                # For general chat, return as is without modification
                if self._is_general_chat(query):
                    return response_content.split("\n\nFor more information")[0]  # Remove any source citation
                # For normal queries, ensure source citation is present
                return response_content
            else:
                logger.error(f"Unexpected response format: {response_data}")
                return "I apologize, but I encountered an error processing your request. Please try again."
            
        except requests.RequestException as e:
            logger.error(f"Error calling ZotGPT API: {str(e)}")
            return "I apologize, but I encountered an error connecting to the API. Please try again later."
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error processing your request. Please try again."

    def _is_general_chat(self, query: str) -> bool:
        """Detect general chat queries like greetings"""
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        gratitude = ['thanks', 'thank you', 'appreciate']
        farewell = ['bye', 'goodbye', 'see you', 'farewell']
        
        query_lower = query.lower()
        return any(word in query_lower for word in greetings + gratitude + farewell)

    def _handle_general_chat(self, query: str) -> str:
        """Handle general chat queries with direct responses - NO API calls or citations"""
        query_lower = query.lower()
        
        # Direct responses without any formatting or citations
        if any(word in query_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
            return "Hello! How can I assist you with making education more accessible?"
            
        if any(word in query_lower for word in ['thanks', 'thank you', 'appreciate']):
            return "You're welcome! Let me know if you have any other questions."
            
        if any(word in query_lower for word in ['bye', 'goodbye', 'see you', 'farewell']):
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
        source_urls = []
        source_titles = []
        seen_urls = set()
        
        for doc in retrieved_docs:
            if hasattr(doc, 'metadata'):
                content = doc.metadata.get('content', '')
                source_url = doc.metadata.get('source_url', '')
                title = doc.metadata.get('title', '')
                
                if content:
                    # Clean the content
                    content_lines = content.split('\n')
                    cleaned_content = []
                    for line in content_lines:
                        if line.strip() and not line.startswith(('Source URL:', 'Title:')):
                            cleaned_content.append(line)
                    
                    if cleaned_content:
                        contexts.append('\n'.join(cleaned_content))
                    
                    # Add source URL and title if available
                    if source_url and source_url not in seen_urls:
                        seen_urls.add(source_url)
                        source_urls.append(source_url)
                        if title:
                            source_titles.append(title)
        
        return {
            'content': '\n\n'.join(contexts),
            'source_urls': source_urls,
            'source_titles': source_titles
        }

    def _check_relevance(self, retrieved_docs: List[Dict], threshold: float = 0.6) -> bool:
        """
        Check document relevance using multiple criteria:
        1. Semantic similarity score from vector search
        2. Content validation through key topic presence
        """
        if not retrieved_docs:
            return False
        
        # Core educational accessibility topics
        core_topics = {
            'education': ['learning', 'student', 'teach', 'school', 'classroom', 'curriculum', 'instruction', 'class', 'programming', 'slides', 'lecture', 'course'],
            'accessibility': ['disability', 'accessible', 'accommodation', 'modification', 'support', 'assistive', 'access', 'inclusive'],
            'technology': ['software', 'tool', 'device', 'technology', 'digital', 'online', 'platform', 'computer', 'python', 'code']
        }
        
        # Log all scores for debugging
        scores = [doc.score for doc in retrieved_docs]
        logger.info(f"All document scores: {scores}")
        
        for doc in retrieved_docs:
            # Log individual score
            logger.info(f"Checking document with score: {doc.score}")
            
            if doc.score <= threshold:
                logger.info(f"Document score {doc.score} below threshold {threshold}")
                continue
                
            # Check if document content contains relevant topic combinations
            content = doc.metadata.get('content', '').lower()
            query_lower = doc.metadata.get('query', '').lower()
            
            # Count matches for each topic area
            topic_matches = {
                topic: sum(1 for term in terms if term in content or term in query_lower)
                for topic, terms in core_topics.items()
            }
            
            logger.info(f"Topic matches found: {topic_matches}")
            
            # Document is relevant if:
            # 1. It matches terms from at least two topic areas
            # 2. Has a good relevance score
            matching_topics = sum(1 for matches in topic_matches.values() if matches > 0)
            if matching_topics >= 2 and doc.score > threshold:
                logger.info(f"Document is relevant: matches in {matching_topics} topics with score {doc.score}")
                return True
                
        logger.info("No documents met relevance criteria")
        return False

    def _is_education_accessibility_related(self, query: str) -> bool:
        """Check if the query is related to educational accessibility"""
        education_keywords = [
            'education', 'learning', 'student', 'teach', 'school', 'classroom', 'accommodation',
            'disability', 'accessible', 'accessibility', 'iep', 'modification', 'curriculum',
            'instruction', 'assessment', 'support', 'academic', 'universal design'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in education_keywords)


