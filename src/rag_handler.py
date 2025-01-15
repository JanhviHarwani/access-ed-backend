from openai import OpenAI
import os
from typing import List, Dict
import re

class RAGHandler:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        # Define topics that are within scope
        self.valid_topics = {
            
            # Core Accessibility
            'accessible for cs',
            'accessibility',
            'universal design',
            
            # Disabilities & Impairments
            'blindness',
            'visual impairment',
            'deafness',
            'hearing impairment',
            
            # Technical Solutions
            'assistive technology',
            'screen reader',
            'braille',
            'alt text and image descriptions',
            'making documents accessible',
            'making pdfs accessible',
            
            # Education Specific
            'making cs education inclusive',
            'education',
            'teaching strategies',
            'classroom adaptation',
            'curriculum',
            'learning',
            
            # Support & Resources
            'accommodations',
            # sticking to one work keyword
            # 'lived experience of coding with disability', 
            'organizations',
            'research on accessing cs'
            'cs accessible',
            'course accessible',
            'cs',
            'accessible by alt text',
            'alt text',
            'deafness',
            'cs inclusive',
            'making cs inclusive',
            'making documents accessible',
            'making pdfs accessible',
            'organizations',
            'organisations',
            'pdfs'
            'experience',
            'computer science',
            'computer science accessible'
            'accessibility',
            'disability',
            'visual impairment',
            'blindness',
            'deafness',
            'hearing impairment',
            'assistive technology',
            'accommodation',
            'universal design',
            'screen reader',
            'braille',
            'education',
            'learning',
            'teaching',
            'classroom',
            'student',
            'instructor',
            'materials',
            'curriculum',
            'ide',
            'vs',
            'project',
            'accessing computer science',
'accessing programming',
'accessing coding',
'accessible programming',
'accessible IDE',
'coding with a screen reader',
'making pdfs accessible',
'making slides accessible',
'making presentations accessible',
'making lectures accessible',
'accessible for cs',
'accessibility',
'universal design',



'blindness',
'blind',
'low vision',
'vision impairment',
'visually impaired'
'deafness',
'hearing impairment,'
'lived experience',


'assistive technology',
'screen reader',
'braille',
'alt text',
'alternative text',
'image description',
'making documents accessible',
'making pdfs accessible',
'making slides accessible',
'making presentations accessible',
'making lectures accessible',
'accessible coding',
'IDE',

'making cs education inclusive',
'education',
'teaching strategies',
'classroom adaptation',
'curriculum',
'learning',
'syllabus',


'accommodations',
'lived experience of coding with disability',
'organizations',
'research on accessing cs',


'accessing computer science',
'accessing programming',
'accessing coding',
'accessible programming',
'accessible IDE',
'coding with a screen reader',
        }

    def generate_response(self, query: str, retrieved_docs: List[Dict], conversation_history: str = "") -> str:
        # Handle general chat (greetings, thanks, etc.)
        if self._is_general_chat(query):
            return self._handle_general_chat(query)
        
        # Check if query is within scope
        if not self._is_query_in_scope(query):
            return self._handle_out_of_scope_query()
            
        # If no relevant documents were retrieved, it's likely out of scope
        if not retrieved_docs:
            return self._handle_out_of_scope_query()
            
        # Check relevance score of retrieved documents
        if not self._check_relevance(retrieved_docs):
            return self._handle_out_of_scope_query()
        
        # Prepare context and create prompt
        context = self._prepare_context(retrieved_docs)
        prompt = self._create_conversational_prompt(query, context, conversation_history)
        
        # Generate response using chat completion
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": """You are an expert assistant helping educators make education accessible 
                                for visually impaired students. Maintain a conversational and helpful tone 
                                while providing accurate, practical information. Use the context provided 
                                to give specific, actionable advice. Acknowledge and build upon previous 
                                parts of the conversation when relevant."""
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500  # Increased for more detailed responses
        )
        
        return response.choices[0].message.content

    def _is_query_in_scope(self, query: str) -> bool:
        # Convert query to lowercase for matching
        query_lower = query.lower()
        
        # Check if query contains any valid topics
        contains_valid_topic = any(topic in query_lower for topic in self.valid_topics)
        
        # List of common out-of-scope patterns
        out_of_scope_patterns = [
            r'\b(weather|forecast)\b',
            r'\b(stock|market|price)\b',
            r'\b(news|current events)\b',
            r'\b(sports|game|score)\b',
            r'\b(recipe|cook|food)\b',
            r'\b(movie|show|watch)\b',
            r'\b(time|date)\b',
            r'what(\'s| is) the .* today',
            r'how (many|much) .* cost',
            r'where (can|do) I (find|buy|get)',
        ]
        
        # Check if query matches any out-of-scope patterns
        matches_out_of_scope = any(re.search(pattern, query_lower) 
                                 for pattern in out_of_scope_patterns)
        
        return contains_valid_topic and not matches_out_of_scope

    def _check_relevance(self, retrieved_docs: List[Dict], threshold: float = 0.7) -> bool:
        """Check if retrieved documents are relevant enough"""
        if not retrieved_docs:
            return False
            
        # Check if any document has a score above threshold
        return any(doc.score > threshold for doc in retrieved_docs)

    def _handle_out_of_scope_query(self) -> str:
        """Handle queries that are determined to be out of scope"""
        return ("I apologize, but I can only answer questions related to educational accessibility, "
                "assistive technologies, and supporting students with visual or hearing impairments. "
                "Please feel free to ask me anything about these topics, and I'll be happy to help!")

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
            return "Hello! I'm here to help you make education more accessible for visually impaired students. What would you like to know?"
            
        if any(word in query_lower for word in ['thanks', 'thank you']):
            return "You're welcome! Feel free to ask if you have any more questions about accessibility."
            
        if any(word in query_lower for word in ['bye', 'goodbye']):
            return "Goodbye! Don't hesitate to return if you need more assistance with accessibility matters."
            
        return "I'm here to help with your accessibility-related questions!"

    def _prepare_context(self, retrieved_docs: List[Dict]) -> str:
        """Prepare context from retrieved documents"""
        context = []
        for doc in retrieved_docs:
            content = doc.metadata.get('content', '')
            source = doc.metadata.get('source', '')
            if content:
                context.append(f"Content: {content}\nSource: {source}\n")
        return "\n".join(context)

    def _create_conversational_prompt(self, query: str, context: str, conversation_history: str) -> str:
        """Create a prompt that includes conversation history and context"""
        prompt = f"""Previous conversation:
{conversation_history}

Context from knowledge base:
{context}

Current question: {query}

Please provide a conversational response that:
1. Acknowledges any relevant previous context from our conversation
2. Incorporates information from the knowledge base
3. Maintains a helpful and friendly tone
4. Provides specific, practical advice
5. Encourages further questions if needed

Remember to:
- Focus on educational accessibility and support for visually impaired students
- Provide concrete examples when possible
- Break down complex concepts into understandable parts
- Suggest actionable steps when appropriate
- Keep the tone warm and encouraging while maintaining professionalism
"""
        return prompt

    def _format_sources(self, retrieved_docs: List[Dict]) -> str:
        """Format source citations from retrieved documents"""
        sources = set()
        for doc in retrieved_docs:
            source = doc.metadata.get('source', '')
            if source:
                sources.add(source)
        
        if not sources:
            return ""
            
        formatted_sources = "\n\nSources:\n" + "\n".join(f"- {source}" for source in sources)
        return formatted_sources