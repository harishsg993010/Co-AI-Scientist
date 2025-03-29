import trafilatura
import requests
from typing import Optional, Dict, List, Any
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_website_text_content(url: str) -> Optional[str]:
    """
    Extract the main text content from a website
    
    Args:
        url: The URL of the website to scrape
        
    Returns:
        The extracted text content or None if extraction failed
    """
    try:
        # Send a request to the website
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            logger.warning(f"Failed to download content from URL: {url}")
            return None
            
        # Extract the main content
        text = trafilatura.extract(downloaded)
        if not text:
            logger.warning(f"Failed to extract text content from URL: {url}")
            return None
            
        return text
    except Exception as e:
        logger.error(f"Error scraping website {url}: {str(e)}")
        return None


def extract_metadata(url: str) -> Dict[str, Any]:
    """
    Extract metadata from a webpage (title, description, etc.)
    
    Args:
        url: The URL of the website
        
    Returns:
        A dictionary containing metadata
    """
    metadata = {
        "title": "",
        "description": "",
        "keywords": [],
        "author": "",
        "published_date": ""
    }
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata["title"] = title_tag.text.strip()
        
        # Extract meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            property = meta.get('property', '').lower()
            content = meta.get('content', '')
            
            if name == 'description' or property == 'og:description':
                metadata["description"] = content
            elif name == 'keywords':
                metadata["keywords"] = [k.strip() for k in content.split(',')]
            elif name == 'author':
                metadata["author"] = content
            elif property == 'article:published_time':
                metadata["published_date"] = content
                
        return metadata
    except Exception as e:
        logger.error(f"Error extracting metadata from {url}: {str(e)}")
        return metadata
