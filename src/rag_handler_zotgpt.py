import requests
import os
import re
import json
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGHandler:
    def __init__(self):
        self.api_key = os.getenv('ZOTGPT_API_KEY')  # Ensure this environment variable is set
        self.deployment_id = os.getenv('ZOTGPT_DEPLOYMENT_ID')  # Set this in your environment
        self.api_url = f"https://azureapi.zotgpt.uci.edu/openai/deployments/{self.deployment_id}/chat/completions?api-version=2024-02-01"

    def generate_response(self, query: str, retrieved_docs: List[Dict], conversation_history: List[Dict] = None) -> str:
        try:
            if self._is_general_chat(query):
                return self._handle_general_chat(query)
            
            if not retrieved_docs or not self._check_relevance(retrieved_docs):
                return "I don't have enough relevant information to answer your question accurately. Could you please rephrase or ask something else?"
            
            response = self._call_zotgpt_api(query, retrieved_docs, conversation_history)
            response_data = response.json()
            # logger.info(f"ZotGPT API response: {response_data}")
            
            context_data = self._prepare_context(retrieved_docs, response_data)
            
            if "choices" in response_data:
                return response_data["choices"][0]["message"]["content"]
            else:
                logger.error(f"Unexpected response: {response_data}")
                return "I encountered an error processing your request. Please try again."
        
        except requests.RequestException as e:
            logger.error(f"Error calling ZotGPT API: {str(e)}")
            return "I apologize, but I encountered an error connecting to the API. Please try again later."

    def _call_zotgpt_api(self, query: str, retrieved_docs: List[Dict], conversation_history: List[Dict]) -> requests.Response:
        formatted_history = self._format_conversation_history(conversation_history)
        system_prompt = ("You are an expert assistant helping educators make education accessible "
                         "for students with disabilities. Provide accurate, practical information while maintaining "
                         "a professional and direct tone. After providing your response, ALWAYS end with a line that says "
                         "'For more information, visit: [source_url]' using the source URLs from the context.")
        
        context_data = self._prepare_context(retrieved_docs, {})
        user_prompt = (f"Context: {context_data['content']}\n\n"
                      "Based on the above context, please answer the following question:\n"
                      f"{query}\n\n"
                      "Remember to end your response by directing users to the source using this format:\n"
                      "For more information, visit: [actual_url_here]\n\n"
                      "You can find the source URL in this context:\n"
                      f"{context_data['source_info']}")
        
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
        
        return requests.post(self.api_url, headers=headers, json=payload)
    
    def _check_relevance(self, retrieved_docs: List[Dict], threshold: float = 0.6) -> bool:
        return any(doc['score'] > threshold for doc in retrieved_docs)

    def _is_general_chat(self, query: str) -> bool:
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        gratitude = ['thanks', 'thank you', 'appreciate']
        farewell = ['bye', 'goodbye', 'see you', 'farewell']
        return any(word in query.lower() for word in greetings + gratitude + farewell)

    def _handle_general_chat(self, query: str) -> str:
        query_lower = query.lower()
        if any(word in query_lower for word in ['hello', 'hi', 'hey']):
            return "How can I assist you with making education more accessible?"
        if any(word in query_lower for word in ['thanks', 'thank you']):
            return "You're welcome! Let me know if you have any other questions."
        if any(word in query_lower for word in ['bye', 'goodbye']):
            return "Goodbye! Feel free to return if you need more assistance."
        return "How can I help you with accessibility-related questions?"

    def _format_conversation_history(self, conversation_history: List[Dict] = None) -> List[Dict]:
        if not conversation_history:
            return []
        return [{"role": msg.get('role', 'user'), "content": msg.get('content', '')} for msg in conversation_history[-5:]]

    def _prepare_context(self, retrieved_docs: List[Dict], response_data: Dict) -> dict:
        contexts, source_info = [], []
        
        for doc in retrieved_docs:
            content = doc.get('metadata', {}).get('content', '')
            if content:
                url_match = re.search(r'Source URL: (.*?)\n', content)
                title_match = re.search(r'Title: (.*?)\n', content)
                url = url_match.group(1).strip() if url_match else None
                title = title_match.group(1).strip() if title_match else None
                
                print(f"Extracted URL: {url}")  # Debugging URL extraction
                
                cleaned_content = self._extract_clean_content(content)
                
                if cleaned_content:
                    contexts.append(cleaned_content)
                if url:
                    source_info.append(f"Source: {title if title else 'Document'} - {url}")
        
        print(f"Final Source Info: {source_info}")  # Debugging Source Info List

        if source_info:
            source_url = source_info[0].split('- ')[-1] if '- ' in source_info[0] else source_info[0]  # Ensure correct extraction
            print(f"Replacing [source_url] with: {source_url}")  # Debugging replacement
            
            if "choices" in response_data and response_data["choices"]:
                response_text = response_data["choices"][0]["message"]["content"]
                response_text = response_text.replace("[source_url]", source_url)
                response_data["choices"][0]["message"]["content"] = response_text
                
        return {'content': '\n\n'.join(contexts), 'source_info': '\n'.join(source_info)}

    
    def _extract_clean_content(self, content: str) -> str:
        lines, clean_lines, found_content = content.split('\n'), [], False
        for line in lines:
            if 'Content:' in line:
                found_content = True
                continue
            if found_content and line.strip():
                clean_lines.append(line)
        return '\n'.join(clean_lines)

