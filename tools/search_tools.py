from typing import List, Dict, Any, Optional
import os
import json
import requests
import logging
from crewai.tools import BaseTool

logger = logging.getLogger(__name__)

# Get API key from environment
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")


class SerperDevTool(BaseTool):
    """Tool for searching the web using Serper.dev API"""
    
    name: str = "SerperDevSearchTool"
    description: str = "Search the web for information using Google Search API"
    
    def _run(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web using the Serper.dev API
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            A list of search results
        """
        if not SERPER_API_KEY:
            logger.error("SERPER_API_KEY is not set")
            return [{"error": "SERPER_API_KEY is not set in environment variables"}]
        
        try:
            url = "https://google.serper.dev/search"
            payload = json.dumps({
                "q": query,
                "gl": "us",
                "hl": "en",
                "num": max_results
            })
            
            headers = {
                'X-API-KEY': SERPER_API_KEY,
                'Content-Type': 'application/json'
            }
            
            response = requests.post(url, headers=headers, data=payload)
            response.raise_for_status()
            search_results = response.json()
            
            # Extract and format the search results
            formatted_results = []
            
            # Process organic results
            if 'organic' in search_results:
                for result in search_results['organic'][:max_results]:
                    formatted_results.append({
                        'title': result.get('title', ''),
                        'link': result.get('link', ''),
                        'snippet': result.get('snippet', ''),
                        'position': result.get('position', 0),
                        'type': 'organic'
                    })
            
            # Process knowledge graph if available
            if 'knowledgeGraph' in search_results:
                kg = search_results['knowledgeGraph']
                formatted_results.append({
                    'title': kg.get('title', ''),
                    'type': 'knowledge_graph',
                    'description': kg.get('description', ''),
                    'attributes': kg.get('attributes', {}),
                    'link': kg.get('link', '')
                })
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error searching with Serper.dev API: {str(e)}")
            return [{"error": f"Search failed: {str(e)}"}]


class ResearchAPITool(BaseTool):
    """Tool for querying domain-specific research APIs"""
    
    name: str = "ResearchAPITool"
    description: str = "Query domain-specific research APIs for scientific information"
    
    def _run(self, query: str, api_type: str = "arxiv") -> List[Dict[str, Any]]:
        """
        Query research APIs for scientific information
        
        Args:
            query: The search query
            api_type: Type of API to query ("arxiv", "pubmed", etc.)
            
        Returns:
            A list of research results
        """
        if api_type == "arxiv":
            return self._query_arxiv(query)
        elif api_type == "pubmed":
            return self._query_pubmed(query)
        else:
            return [{"error": f"Unsupported API type: {api_type}"}]
    
    def _query_arxiv(self, query: str) -> List[Dict[str, Any]]:
        """Query the arXiv API"""
        try:
            # Format the query for arXiv API
            formatted_query = query.replace(' ', '+')
            url = f"http://export.arxiv.org/api/query?search_query=all:{formatted_query}&start=0&max_results=5"
            
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse the XML response
            # This is a simplified parser; in a real app, use a proper XML parser
            results = []
            entries = response.text.split('<entry>')[1:]  # Skip the first split which is header
            
            for entry in entries:
                try:
                    title = entry.split('<title>')[1].split('</title>')[0].strip()
                    summary = entry.split('<summary>')[1].split('</summary>')[0].strip()
                    link = entry.split('<id>')[1].split('</id>')[0].strip()
                    
                    results.append({
                        'title': title,
                        'link': link,
                        'summary': summary,
                        'source': 'arxiv'
                    })
                except IndexError:
                    continue
            
            return results
        
        except Exception as e:
            logger.error(f"Error querying arXiv API: {str(e)}")
            return [{"error": f"arXiv query failed: {str(e)}"}]
    
    def _query_pubmed(self, query: str) -> List[Dict[str, Any]]:
        """Query the PubMed API"""
        # Simplified implementation; in a real app, use a proper PubMed API client
        return [{"info": "PubMed API functionality not yet implemented"}]
