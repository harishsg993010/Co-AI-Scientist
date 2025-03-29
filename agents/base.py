from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime
from crewai import Agent
from crewai.tools import BaseTool

logger = logging.getLogger(__name__)


class BaseResearchAgent(Agent):
    """Base class for all research agents in the system"""
    
    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        name: str = None,
        tools: Optional[List[BaseTool]] = None,
        llm: Union[str, Any] = None,
        verbose: bool = False,
        allow_delegation: bool = True,
        memory: bool = True,
        max_iter: int = 20,
        max_rpm: Optional[int] = None,
        max_execution_time: Optional[int] = None,
        max_retry_limit: int = 2,
        cache: bool = True,
        respect_context_window: bool = True,
        function_calling_llm: Optional[Any] = None,
        system_template: Optional[str] = None,
        prompt_template: Optional[str] = None,
        response_template: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize a base research agent
        
        Args:
            role: The agent's role
            goal: The agent's goal
            backstory: The agent's backstory
            name: The agent's name
            tools: List of tools available to the agent
            llm: LLM model to use
            verbose: Whether to log verbose output
            allow_delegation: Whether to allow task delegation
            memory: Whether the agent should maintain memory of interactions
            max_iter: Maximum iterations before agent must provide best answer
            max_rpm: Maximum requests per minute to avoid rate limits
            max_execution_time: Maximum execution time in seconds
            max_retry_limit: Maximum number of retries when an error occurs
            cache: Enable caching for tool usage
            respect_context_window: Keep messages under context window size by summarizing
            function_calling_llm: Language model for tool calling, overrides crew's LLM if specified
            system_template: Custom system prompt template
            prompt_template: Custom prompt template
            response_template: Custom response template
            **kwargs: Additional keyword arguments
        """
        # Initialize the parent Agent class
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            verbose=verbose,
            allow_delegation=allow_delegation,
            name=name,
            tools=tools or [],
            llm=llm,  # Use llm instead of model
            memory=memory,
            max_iter=max_iter,
            max_rpm=max_rpm,
            max_execution_time=max_execution_time,
            max_retry_limit=max_retry_limit,
            cache=cache,
            respect_context_window=respect_context_window,
            function_calling_llm=function_calling_llm,
            system_template=system_template,
            prompt_template=prompt_template,
            response_template=response_template,
            **kwargs
        )
        
        # Initialize our own action history for additional tracking
        self._action_history = []
    
    def log_action(self, action: str, details: Dict[str, Any]):
        """Log an action taken by the agent"""
        log_entry = {
            "agent": self.name,
            "role": self.role,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        self._action_history.append(log_entry)
        
        if self.verbose:
            logger.info(f"Agent {self.name} ({self.role}): {action}")
            
        return log_entry
    
    def get_memory(self) -> List[Dict[str, Any]]:
        """Get the agent's action history log (extends built-in agent memory)"""
        return self._action_history
