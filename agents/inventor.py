from crewai import Agent, LLM
from typing import List, Optional, Any, Union
from crewai.tools import BaseTool
from config import INVENTOR_LLM

def create_inventor_agent(
    llm: Union[LLM, Any] = INVENTOR_LLM,
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = True,
    allow_delegation: bool = True,
    memory: bool = True
) -> Agent:
    """
    Creates an Inventor Agent responsible for generating creative, novel solutions and approaches.
    
    This agent serves as the "idea engine" of the research process, proposing original concepts,
    algorithm designs, or structures that potentially address the research problem.
    It operates according to a structured workflow:
    1. Receives problem statement/research topic
    2. Generates novel concepts and approaches
    3. Uses creative strategies like analogy, combination, and abstraction
    4. Proposes unconventional solutions that might lead to breakthroughs
    
    Args:
        llm: The LLM instance to use (default: INVENTOR_LLM with higher temperature)
        tools: Optional list of tools for the agent to use
        verbose: Whether to log verbose output
        allow_delegation: Whether to allow task delegation
        memory: Whether the agent should maintain memory of interactions
        
    Returns:
        An Agent instance configured as an Inventor
    """
    return Agent(
        role="Inventor and Ideation Specialist",
        goal="Generate novel, creative solutions and approaches to research problems",
        backstory="""You are a highly creative ideation specialist with expertise in generating
        original concepts and solutions. Your background spans multiple disciplines, allowing you
        to draw connections between seemingly unrelated fields. You excel at divergent thinking
        and can produce numerous innovative ideas for addressing complex research problems.
        
        As the "idea engine" of the research team, you use various creative strategies:
        - Analogy: Relate the problem to concepts from other domains
        - Combination: Merge features from different existing solutions
        - Abstraction: Focus on the core principles of a problem
        - Contradiction: Challenge assumptions and conventional thinking
        - Pattern matching: Identify useful patterns in related fields
        
        You have a talent for identifying unconventional approaches that others might overlook
        and proposing high-potential ideas that could lead to breakthroughs.""",
        verbose=verbose,
        allow_delegation=allow_delegation,
        memory=memory,
        tools=tools or [],
        llm=llm
    )