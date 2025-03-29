import os
import logging
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import CrewAI LLM
try:
    from crewai import LLM
    CREWAI_AVAILABLE = True
except ImportError:
    logger.warning("CrewAI not available, will use fallback mechanisms")
    CREWAI_AVAILABLE = False

# API keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")

# Base LLM parameters
LLM_PARAMS = {
    "model":
    "openai/gpt-4o",  # Using GPT-4o as requested by user
    "temperature": 0.7,
    "max_tokens": 1000,  # Increased to 1000 tokens for GPT-4o for better reasoning
    "top_p": 0.9,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.1,
    "stop": ["END"],
    "seed": 42
}


# Function to create LLM with error handling
def create_llm_instance(params: Dict[str, Any]) -> Optional[Any]:
    """
    Create an LLM instance with the provided parameters,
    with proper error handling.
    
    Args:
        params: Dictionary of LLM parameters
        
    Returns:
        LLM instance or None if creation fails
    """
    if not CREWAI_AVAILABLE:
        return None

    if not OPENAI_API_KEY:
        logger.error("No OpenAI API key found in environment")
        return None

    try:
        return LLM(**params)
    except Exception as e:
        logger.error(f"Error creating LLM instance: {str(e)}")
        return None


# LLM Configuration with defaults for various agent types
DEFAULT_PARAMS = LLM_PARAMS.copy()
MANAGER_PARAMS = {
    **LLM_PARAMS, "temperature": 0.7
}  # Lower temperature for more focused management
INVENTOR_PARAMS = {
    **LLM_PARAMS, "temperature": 0.9
}  # Higher temperature for more creativity

# Create LLM instances with error handling
DEFAULT_LLM = create_llm_instance(DEFAULT_PARAMS)
MANAGER_LLM = create_llm_instance(MANAGER_PARAMS)
INVENTOR_LLM = create_llm_instance(INVENTOR_PARAMS)

# Log LLM configuration status
if DEFAULT_LLM:
    logger.info("Default LLM configuration initialized successfully")
else:
    logger.warning("Default LLM configuration failed to initialize")

# Legacy compatibility variables
DEFAULT_MODEL = "openai/gpt-4o"  # Using gpt-4o as requested by user
MANAGER_MODEL = "openai/gpt-4o"  # Using gpt-4o as requested by user
DEFAULT_TEMPERATURE = 0.7  # For backward compatibility
EMBEDDING_MODEL = "text-embedding-ada-002"

# Knowledge Base Settings
VECTOR_DB_PATH = "data/vector_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Research Settings
MAX_HYPOTHESES = 5
MAX_RESEARCH_CYCLES = 3
MAX_SOURCES_PER_QUERY = 5
ELO_K_FACTOR = 32  # For hypothesis ranking

# Web interface settings
PORT = 5000
HOST = "0.0.0.0"
DEBUG = True

# Agent role descriptions
AGENT_ROLES = {
    "researcher": "Searches for and gathers information from various sources",
    "analyst": "Analyzes information and drafts research reports",
    "evaluator": "Critiques hypotheses and evaluates research quality",
    "refiner": "Refines hypotheses based on new information and feedback",
    "supervisor": "Coordinates the research process and manages other agents"
}

# Prompt templates for agents
RESEARCHER_PROMPT = """You are a Research Agent responsible for finding information about {topic}. 
Your goal is to gather relevant, accurate, and up-to-date information from multiple sources.
When searching, focus on {focus_area} and look for information that can help with the hypothesis: {hypothesis}.
Provide all relevant information and cite your sources clearly.
"""

ANALYST_PROMPT = """You are an Analyst Agent responsible for analyzing research data and drafting reports.
Review the information provided by the Researcher Agent about {topic} and the hypothesis: {hypothesis}.
Synthesize the information to identify patterns, connections, and insights.
Draft a well-structured analysis that evaluates the hypothesis based on the evidence.
"""

EVALUATOR_PROMPT = """You are an Evaluator Agent responsible for critically assessing hypotheses and research quality.
Review the hypothesis: {hypothesis} and the analysis provided.
Evaluate the strength of the evidence, identify potential biases or gaps, and assess the logical coherence.
Rate the hypothesis on a scale from 1-10 for plausibility, novelty, and evidence support.
Provide specific feedback on how the hypothesis and research can be improved.
"""

REFINER_PROMPT = """You are a Refinement Agent responsible for improving hypotheses based on feedback and evidence.
Consider the original hypothesis: {hypothesis}, the research findings, and the evaluator's feedback.
Refine the hypothesis to address weaknesses, incorporate new evidence, and improve specificity.
If the evidence contradicts the hypothesis, modify it or propose an alternative.
Provide a clear explanation of your refinements and why they improve the hypothesis.
"""

SUPERVISOR_PROMPT = """You are a Supervisor Agent coordinating the autonomous research process.
Manage the research workflow, delegate tasks to appropriate agents, and ensure progress toward answering research questions.
Your goal is to maximize research quality while minimizing unnecessary work.
Monitor the state of the research, identify when to proceed to the next stage, and determine when the research cycle should end.
Make decisions about which hypotheses to pursue further based on evaluation scores and potential impact.
"""
