# src/response_formatter.py

from typing import List, Dict
import re

class ResponseFormatter:
    def format_answer(self, results: List[Dict], query: str) -> str:
        if not results:
            return "I couldn't find any relevant information about that."

        # Collect and deduplicate content
        contents = []
        seen_content = set()
        for result in results:
            content = result.metadata['content']
            if content not in seen_content:
                contents.append(content)
                seen_content.add(content)

        # Format introduction
        response = f"Here's what I found about {query}:\n\n"

        # Organize content by sections
        sections = self._organize_sections(contents)
        
        # Format each section
        for section_title, points in sections.items():
            if points:
                response += f"{section_title}:\n"
                for point in points:
                    response += f"• {point}\n"
                response += "\n"

        # Add sources
        response += "\nSources:\n"
        seen_sources = set()
        for result in results:
            source = result.metadata.get('source', '')
            if source and source not in seen_sources:
                response += f"- {source}\n"
                seen_sources.add(source)

        return response

    def _organize_sections(self, contents: List[str]) -> Dict[str, List[str]]:
        sections = {
            "Definition & Overview": [],
            "Types & Examples": [],
            "Benefits & Applications": [],
            "Important Considerations": []
        }

        for content in contents:
            sentences = self._split_into_sentences(content)
            for sentence in sentences:
                # Categorize sentence based on content
                if any(word in sentence.lower() for word in ["is", "are", "means", "refers"]):
                    sections["Definition & Overview"].append(sentence)
                elif any(word in sentence.lower() for word in ["example", "such as", "like", "including"]):
                    sections["Types & Examples"].append(sentence)
                elif any(word in sentence.lower() for word in ["help", "benefit", "use", "enable", "allow"]):
                    sections["Benefits & Applications"].append(sentence)
                elif any(word in sentence.lower() for word in ["important", "should", "must", "need", "consider"]):
                    sections["Important Considerations"].append(sentence)

        return {k: v for k, v in sections.items() if v}  # Remove empty sections

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]