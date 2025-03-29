from typing import Dict, Any, List, Optional
import os
import io
import tempfile
import logging
import re
from crewai.tools import BaseTool

logger = logging.getLogger(__name__)

try:
    import PyPDF2
except ImportError:
    logger.warning("PyPDF2 not installed. PDF tools will not function properly.")
    

class PDFSearchTool(BaseTool):
    """Tool for extracting and searching content from PDF documents"""
    
    name: str = "PDFSearchTool"
    description: str = "Extract and search content from PDF documents"
    
    def _run(self, pdf_path: str, query: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract content from a PDF file and optionally search for specific information
        
        Args:
            pdf_path: Path to the PDF file
            query: Optional search query
            
        Returns:
            A dictionary containing the extracted content and search results
        """
        try:
            # Check if PyPDF2 is available
            if 'PyPDF2' not in globals():
                return {
                    "success": False,
                    "error": "PyPDF2 is not installed. Run 'pip install PyPDF2' to enable PDF processing."
                }
            
            # Check if file exists
            if not os.path.exists(pdf_path):
                return {
                    "success": False,
                    "error": f"PDF file not found: {pdf_path}"
                }
            
            # Open and read PDF
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                # Extract text from all pages
                text = ""
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                
                # Search for query if provided
                search_results = []
                if query and text:
                    # Simple search implementation; could be enhanced with NLP
                    query_terms = query.lower().split()
                    lines = text.split('\n')
                    
                    for i, line in enumerate(lines):
                        line_lower = line.lower()
                        if any(term in line_lower for term in query_terms):
                            # Find the page number for this line
                            char_count = sum(len(lines[j]) + 1 for j in range(i))
                            
                            # Approximate page number based on character count
                            # This is a simplification; actual page mapping would be more complex
                            total_chars = len(text)
                            approx_page = int((char_count / total_chars) * num_pages) + 1
                            
                            search_results.append({
                                'text': line.strip(),
                                'approximate_page': approx_page
                            })
                
                return {
                    "success": True,
                    "path": pdf_path,
                    "num_pages": num_pages,
                    "text": text,
                    "search_results": search_results if query else []
                }
        
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "path": pdf_path
            }


class PDFToTextTool(BaseTool):
    """Tool for converting PDFs to text for knowledge base ingestion"""
    
    name: str = "PDFToTextTool"
    description: str = "Convert PDF documents to text for knowledge base ingestion"
    
    def _run(self, pdf_path: str) -> Dict[str, Any]:
        """
        Convert a PDF file to text
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            A dictionary containing the extracted text
        """
        # Implementation similar to PDFSearchTool but focused on text extraction
        try:
            result = PDFSearchTool()._run(pdf_path)
            
            if not result.get("success", False):
                return result
            
            # Clean up the text (remove excessive whitespace, fix common OCR issues)
            text = result.get("text", "")
            text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
            text = re.sub(r'-\s+', '', text)  # Fix hyphenated words
            
            return {
                "success": True,
                "path": pdf_path,
                "text": text,
                "num_pages": result.get("num_pages", 0),
                "title": self._extract_title(text)
            }
        
        except Exception as e:
            logger.error(f"Error converting PDF to text {pdf_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "path": pdf_path
            }
    
    def _extract_title(self, text: str) -> str:
        """Extract the title from the PDF text (simple heuristic)"""
        # Simple heuristic: first non-empty line is likely the title
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return lines[0] if lines else "Unknown Title"
