from typing import Dict
import re
from datetime import datetime

class ContentFormatter:
    @staticmethod
    def format_content(content: Dict) -> Dict:
        """Format PDF content while preserving document structure."""
        
        def clean_empty_lines(text: str) -> str:
            lines = text.split('\n')
            lines = [line for line in lines if line.strip()]
            return '\n\n'.join(lines)
        
        def remove_unwanted_content(text: str) -> str:
            """Remove headers, footers, and other unwanted sections."""
            lines = text.split('\n')
            cleaned_lines = []
            skip_section = False
            
            # Patterns to remove
            unwanted_patterns = [
                r'^SIGACCESS\s*$',
                r'^Newsletter\s*$',
                r'^Issue \d+\s*$',
                r'^Page \d+.*$',
                r'.*June 2017.*$',
                r'^Source File:.*$',
                r'^Processed:.*$',
                r'.*========+.*$',
                r'.*-----+.*$',
                r'^\s*\[.*\]\s*$',  # References in square brackets
                r'https?://\S+',  # URLs
                r'.*knowledge-base.*$',
                r'.*\d{4}\.\d+\.\d+.*$',  # DOI-like numbers
            ]
            
            # Sections to skip
            skip_sections = [
                'Abstract',
                'Acknowledgments',
                'References',
                'About the Author',
                'Figure',
                'Table'
            ]
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Skip unwanted patterns
                if any(re.search(pattern, line) for pattern in unwanted_patterns):
                    continue
                
                # Check for section skipping
                if any(section in line for section in skip_sections):
                    skip_section = True
                    continue
                
                # Check for end of skip section - new numbered section
                if re.match(r'^\d+(\.\d+)?[\s]', line):
                    skip_section = False
                
                if skip_section:
                    continue
                
                # Skip lines with specific characteristics
                if any([
                    '@' in line,  # email addresses
                    line.startswith('University of'),
                    line.startswith('Page'),
                    line.startswith('Newsletter'),
                    line.startswith('Issue'),
                    len(line) <= 3,  # Very short lines
                    line.endswith('.pdf'),
                    line.startswith('Title:'),
                    line.startswith('Source File:'),
                    line.startswith('Processed:'),
                    'ACM' in line,
                    'proceedings' in line.lower(),
                    'conference' in line.lower()
                ]):
                    continue
                
                # Remove citation numbers in square brackets
                line = re.sub(r'\s*\[\d+\]', '', line)
                
                # Clean up extra spaces
                line = ' '.join(line.split())
                
                if line:
                    cleaned_lines.append(line)
            
            return '\n'.join(cleaned_lines)

        def format_sections(text: str) -> str:
            sections = []
            current_section = []
            
            text = remove_unwanted_content(text)
            
            for line in text.split('\n'):
                # Check if line is a section heading (numbered sections only)
                is_heading = re.match(r'^\d+(\.\d+)?[\s]', line.strip())
                
                if is_heading:
                    if current_section:
                        sections.append('\n'.join(current_section))
                        current_section = []
                    heading = line.strip()
                    current_section.append(f"\n{heading}\n")
                else:
                    if line.strip():
                        current_section.append(line.strip())
            
            if current_section:
                sections.append('\n'.join(current_section))
            
            return '\n\n'.join(sections)
        
        formatted_content = {
            'title': content.get('title', 'AccessCSforAll: Making Computer Science Accessible'),
            'source': content.get('file_path', 'PDF Document'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'content': format_sections(clean_empty_lines(content.get('content', '')))
        }
        
        return formatted_content

    @staticmethod
    def save_formatted_content(formatted_content: Dict, filepath: str):
        """Save formatted PDF content to file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Title: {formatted_content['title']}\n")
            f.write(f"Source URL: {formatted_content['source']}\n")
            f.write(f"Retrieved: {formatted_content['timestamp']}\n\n")
            f.write("Content:\n")
            f.write(formatted_content['content'])