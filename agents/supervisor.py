from typing import List, Dict, Any, Optional, Union
import logging
from crewai.tools import BaseTool

from .base import BaseResearchAgent
from models import Hypothesis, ResearchCycle
from config import DEFAULT_MODEL, SUPERVISOR_PROMPT

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseResearchAgent):
    """Agent specialized in coordinating the research process"""
    
    def __init__(
        self,
        name: str = "Supervisor",
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
        Initialize a supervisor agent
        
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
        # Define the supervisor's role, goal, and backstory
        role = "Research Supervisor"
        goal = "Coordinate the research process and make strategic decisions"
        backstory = (
            "You are an expert research coordinator with a talent for managing complex research "
            "projects. Your background in research methodology and project management allows you "
            "to identify promising research directions, allocate resources effectively, and make "
            "strategic decisions about which hypotheses to pursue. You excel at seeing the big picture "
            "while keeping track of important details."
        )
        
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            name=name,
            tools=tools or [],  # Supervisor typically doesn't need external tools
            llm=llm,
            verbose=verbose,
            allow_delegation=allow_delegation,
            memory=memory,
            max_iter=max_iter,
            cache=cache,
            respect_context_window=respect_context_window
        )
    
    def prioritize_hypotheses(
        self,
        hypotheses: List[Hypothesis],
        evaluations: List[Dict[str, Any]],
        max_to_pursue: int = 3
    ) -> Dict[str, Any]:
        """
        Prioritize hypotheses based on evaluations
        
        Args:
            hypotheses: List of hypotheses
            evaluations: List of hypothesis evaluations
            max_to_pursue: Maximum number of hypotheses to prioritize
            
        Returns:
            Dictionary containing prioritized hypotheses
        """
        # Log the prioritization action
        self.log_action("prioritize_hypotheses", {
            "num_hypotheses": len(hypotheses),
            "max_to_pursue": max_to_pursue
        })
        
        # In a real implementation, this would call the LLM through CrewAI
        # Here we'll just return a placeholder for the prioritization
        return {
            "prioritized_hypothesis_ids": [],  # Would contain actual prioritized IDs
            "rationale": "",  # Would contain actual rationale
            "deprioritized_hypothesis_ids": [],  # Would contain actual deprioritized IDs
            "deprioritization_reasons": {}  # Would contain actual reasons
        }
    
    def manage_research_cycle(
        self,
        research_cycle: ResearchCycle,
        hypotheses: List[Hypothesis],
        remaining_cycles: int
    ) -> Dict[str, Any]:
        """
        Make decisions about the current research cycle
        
        Args:
            research_cycle: Current research cycle
            hypotheses: List of hypotheses
            remaining_cycles: Number of remaining research cycles
            
        Returns:
            Dictionary containing cycle management decisions
        """
        # Log the cycle management action
        self.log_action("manage_research_cycle", {
            "cycle_id": research_cycle.id,
            "cycle_number": research_cycle.cycle_number,
            "num_hypotheses": len(hypotheses),
            "remaining_cycles": remaining_cycles
        })
        
        # Update the agent's prompt with specific management parameters
        supervisor_prompt = SUPERVISOR_PROMPT
        
        # In a real implementation, this would call the LLM through CrewAI
        # Here we'll just return a placeholder for the decisions
        return {
            "continue_cycle": True,  # Whether to continue the current cycle
            "hypotheses_to_refine": [],  # Hypothesis IDs to refine
            "hypotheses_to_discard": [],  # Hypothesis IDs to discard
            "new_hypotheses_needed": False,  # Whether to generate new hypotheses
            "cycle_summary": "",  # Summary of the cycle
            "next_actions": []  # List of next actions
        }
