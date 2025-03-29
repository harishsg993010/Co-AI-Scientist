from typing import Dict, Any
import requests
from bs4 import BeautifulSoup
import logging
from crewai.tools import BaseTool

from web_scraper import get_website_text_content, extract_metadata

logger = logging.getLogger(__name__)


class ScrapeWebsiteTool(BaseTool):
    """Tool for scraping content from websites"""
    
    name: str = "ScrapeWebsiteTool"
    description: str = "Scrape and extract content from a website URL"
    
    def _run(self, url: str) -> Dict[str, Any]:
        """
        Scrape content from a website
        
        Args:
            url: The URL of the website to scrape
            
        Returns:
            A dictionary containing the extracted content and metadata
        """
        try:
            # Extract main text content
            content = get_website_text_content(url)
            if not content:
                return {
                    "success": False,
                    "error": "Failed to extract content from the website",
                    "url": url
                }
            
            # Extract metadata
            metadata = extract_metadata(url)
            
            return {
                "success": True,
                "url": url,
                "content": content,
                "title": metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "author": metadata.get("author", ""),
                "published_date": metadata.get("published_date", ""),
                "length": len(content)
            }
        
        except Exception as e:
            logger.error(f"Error scraping website {url}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }


class SpecializedScraperTool(BaseTool):
    """Tool for scraping content from specialized websites (e.g., academic journals)"""
    
    name: str = "SpecializedScraperTool"
    description: str = "Scrape content from specialized websites with custom handling"
    
    def _run(self, url: str, site_type: str = "general") -> Dict[str, Any]:
        """
        Scrape content from specialized websites
        
        Args:
            url: The URL of the website to scrape
            site_type: Type of website ("academic", "news", etc.)
            
        Returns:
            A dictionary containing the extracted content
        """
        if site_type == "academic":
            return self._scrape_academic_site(url)
        elif site_type == "news":
            return self._scrape_news_site(url)
        else:
            # Fall back to general scraping
            return ScrapeWebsiteTool()._run(url)
    
    def _scrape_academic_site(self, url: str) -> Dict[str, Any]:
        """Scrape content from academic websites"""
        try:
            # General scraping first
            base_result = ScrapeWebsiteTool()._run(url)
            if not base_result.get("success", False):
                return base_result
            
            # Extract additional academic-specific information
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try to find DOI, authors, publication date, etc.
            doi = ""
            doi_elem = soup.find('meta', attrs={'name': 'citation_doi'})
            if doi_elem:
                doi = doi_elem.get('content', '')
            
            # Look for authors in academic format
            authors = []
            author_elems = soup.find_all('meta', attrs={'name': 'citation_author'})
            for author_elem in author_elems:
                authors.append(author_elem.get('content', ''))
            
            # Add academic-specific fields to result
            base_result["doi"] = doi
            base_result["authors"] = authors if authors else [base_result.get("author", "")]
            
            return base_result
        
        except Exception as e:
            logger.error(f"Error scraping academic website {url}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    def _scrape_news_site(self, url: str) -> Dict[str, Any]:
        """Scrape content from news websites"""
        # Similar implementation as academic but focused on news attributes
        # This is a simplified implementation; enhance as needed
        return ScrapeWebsiteTool()._run(url)
