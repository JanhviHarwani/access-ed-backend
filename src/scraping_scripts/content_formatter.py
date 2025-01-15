# src/scraping_scripts/content_formatter.py

from typing import Dict
import re

class ContentFormatter:
    @staticmethod
    def format_content(content: Dict) -> Dict:
        """Format content while preserving original information."""
        
        def clean_empty_lines(text: str) -> str:
            # Remove multiple empty lines but preserve paragraph structure
            lines = text.split('\n')
            # Remove lines that are just whitespace
            lines = [line for line in lines if line.strip()]
            # Join with double newline to preserve paragraphs
            return '\n\n'.join(lines)
        
        def format_sections(text: str) -> str:
            # Split content into sections based on main headings
            sections = []
            current_section = []
            
            for line in text.split('\n'):
                # Check if line is a main heading (like Overview, Prevention, Treatment)
                if line.strip() and all(c.isupper() or c.isspace() for c in line):
                    # If we have content in current section, add it
                    if current_section:
                        sections.append('\n'.join(current_section))
                        current_section = []
                    # Add the heading
                    current_section.append(f"\n{line.strip()}\n{'='*len(line.strip())}")
                else:
                    # Add content lines
                    if line.strip():
                        current_section.append(line.strip())
            
            # Add the last section
            if current_section:
                sections.append('\n'.join(current_section))
            
            return '\n\n'.join(sections)
        
        formatted_content = {
            'title': content['title'].strip(),
            'url': content['url'],
            'timestamp': content['timestamp'],
            'content': format_sections(clean_empty_lines(content['content']))
        }
        
        return formatted_content

    @staticmethod
    def save_formatted_content(formatted_content: Dict, filepath: str):
        """Save formatted content to file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Title: {formatted_content['title']}\n")
            f.write(f"Source URL: {formatted_content['url']}\n")
            f.write(f"Retrieved: {formatted_content['timestamp']}\n")
            f.write("\nContent:\n")
            f.write(formatted_content['content'])

# Example usage
if __name__ == "__main__":
    # Sample content
    content = {
        'title': "Eye care, vision impairment and blindness",
        'url': "https://www.who.int/health-topics/blindness-and-vision-loss",
        'timestamp': "2024-11-22 01:36:15",
        'content': """Overview
Eye conditions are remarkably common. Those who live long enough will experience at least one eye condition during their lifetime.

Prevention
Eye conditions that can be targeted effectively with preventive strategies include..."""
    }
    
    formatter = ContentFormatter()
    formatted = formatter.format_content(content)
    formatter.save_formatted_content(formatted, "formatted_content.txt")