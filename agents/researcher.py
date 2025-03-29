from typing import List, Dict, Any, Optional, Union
import logging
from crewai.tools import BaseTool

from .base import BaseResearchAgent
from tools.search_tools import SerperDevTool, ResearchAPITool
from tools.scraping_tools import ScrapeWebsiteTool
from tools.pdf_tools import PDFSearchTool
from config import DEFAULT_MODEL, RESEARCHER_PROMPT

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseResearchAgent):
    """Agent specialized in finding and gathering information"""
    
    def __init__(
        self,
        name: str = "Researcher",
        llm: Union[str, Any] = DEFAULT_MODEL,
        tools: Optional[List[BaseTool]] = None,
        verbose: bool = False,
        allow_delegation: bool = True,
        memory: bool = True,
        max_iter: int = 20,
        cache: bool = True,
        respect_context_window: bool = True
    ):
        """
        Initialize a researcher agent
        
        Args:
            name: The agent's name
            llm: LLM model to use
            tools: List of tools available to the agent
            verbose: Whether to log verbose output
            allow_delegation: Whether to allow task delegation
            memory: Whether the agent should maintain memory of interactions
            max_iter: Maximum iterations before agent must provide best answer
            cache: Enable caching for tool usage
            respect_context_window: Keep messages under context window size by summarizing
        """
        # Define the researcher's role, goal, and backstory
        role = "Research Specialist"
        goal = "Find accurate, relevant, and comprehensive information from various sources"
        backstory = (
            "You are an expert researcher with skills in finding information from various sources. "
            "You excel at searching the web, academic databases, and documents to gather "
            "relevant information on any topic. You are thorough, detail-oriented, and have a "
            "knack for discovering hard-to-find information. You always cite your sources and "
            "provide comprehensive reports on your findings."
        )
        
        # Set up default tools if none provided
        default_tools = [
            SerperDevTool(),
            ResearchAPITool(),
            ScrapeWebsiteTool(),
            PDFSearchTool()
        ]
        
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            name=name,
            tools=tools or default_tools,
            llm=llm,
            verbose=verbose,
            allow_delegation=allow_delegation,
            memory=memory,
            max_iter=max_iter,
            cache=cache,
            respect_context_window=respect_context_window
        )
    
    def research_topic(self, topic: str, focus_area: Optional[str] = None, hypothesis: Optional[str] = None) -> Dict[str, Any]:
        """
        Research a topic and gather information
        
        Args:
            topic: The research topic
            focus_area: Specific area to focus on (optional)
            hypothesis: Hypothesis to investigate (optional)
            
        Returns:
            Dictionary containing research results
        """
        # Log the research action
        self.log_action("research_topic", {
            "topic": topic,
            "focus_area": focus_area,
            "hypothesis": hypothesis
        })
        
        # Update the agent's prompt with specific research parameters
        research_prompt = RESEARCHER_PROMPT.format(
            topic=topic,
            focus_area=focus_area or "general information",
            hypothesis=hypothesis or "no specific hypothesis"
        )
        
        # In a real implementation, this would call the LLM through CrewAI
        # Here we'll just return a placeholder for the results
        return {
            "topic": topic,
            "hypothesis": hypothesis,
            "focus_area": focus_area,
            "status": "completed",
            "sources_found": [],  # Would contain actual sources
            "research_notes": ""  # Would contain actual research notes
        }
    
    def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web for information
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results
        """
        search_tool = next((tool for tool in self.tools if isinstance(tool, SerperDevTool)), None)
        if not search_tool:
            logger.warning("SerperDevTool not available for web search")
            return []
        
        self.log_action("search_web", {"query": query, "max_results": max_results})
        return search_tool._run(query, max_results)
    
    def scrape_website(self, url: str) -> Dict[str, Any]:
        """
        Scrape content from a website
        
        Args:
            url: Website URL
            
        Returns:
            Dictionary containing scraped content
        """
        scrape_tool = next((tool for tool in self.tools if isinstance(tool, ScrapeWebsiteTool)), None)
        if not scrape_tool:
            logger.warning("ScrapeWebsiteTool not available for website scraping")
            return {"success": False, "error": "Scraping tool not available"}
        
        self.log_action("scrape_website", {"url": url})
        return scrape_tool._run(url)
