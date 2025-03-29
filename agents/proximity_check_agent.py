"""
Proximity Check Agent for the AI Co-Scientist System.
Evaluates the relevance and alignment of research with the original goals.
"""
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime

from agents.base import BaseResearchAgent
from crewai.tools import BaseTool

logger = logging.getLogger(__name__)


class ProximityCheckAgent(BaseResearchAgent):
    """
    Proximity Check Agent responsible for ensuring research
    stays aligned with the scientist's original goals and constraints.
    """
    
    def __init__(
        self,
        name: str = "Proximity Check Agent",
        tools: Optional[List[BaseTool]] = None,
        llm: Union[str, Any] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the Proximity Check Agent
        
        Args:
            name: Agent name
            tools: List of tools available to the agent
            llm: LLM to use
            verbose: Whether to log verbose output
            **kwargs: Additional keyword arguments
        """
        # Define the agent's role, goal, and backstory
        role = "Research Alignment and Relevance Assessor"
        goal = ("Ensure research hypotheses and proposals remain aligned with the scientist's " 
                "original goals, preferences, and constraints throughout the research process")
        backstory = (
            "You are a specialized AI agent designed to monitor and evaluate the alignment of "
            "evolving research with the scientist's original intent. You've been trained to "
            "understand research goals, experimental constraints, and scientific preferences, "
            "then use that understanding to assess how closely developing hypotheses match these "
            "parameters. Your assessments help prevent research drift and ensure that the scientific "
            "process remains focused on the questions that matter most to the scientist. Your work "
            "is critical for maintaining the relevance and utility of the research outputs."
        )
        
        # Initialize the base agent
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            name=name,
            tools=tools,
            llm=llm,
            verbose=verbose,
            **kwargs
        )
    
    def define_research_parameters(self, 
                                 research_goal: str,
                                 preferences: List[str] = None,
                                 constraints: List[str] = None,
                                 attributes: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Define the parameters against which research will be checked
        
        Args:
            research_goal: Primary research goal
            preferences: Optional scientist preferences
            constraints: Optional experimental or research constraints
            attributes: Optional additional attributes to consider
            
        Returns:
            Dictionary containing defined research parameters
        """
        action = "Defining research parameters for proximity checking"
        self.log_action(action, {
            "research_goal": research_goal,
            "has_preferences": bool(preferences),
            "has_constraints": bool(constraints),
            "has_attributes": bool(attributes)
        })
        
        # Compile the parameters
        parameters = {
            "research_goal": research_goal,
            "preferences": preferences or [],
            "constraints": constraints or [],
            "attributes": attributes or {},
            "derived_keywords": [],
            "primary_focuses": [],
            "out_of_scope_areas": []
        }
        
        # This would involve actual parameter analysis using the LLM
        # to extract keywords, focuses and out-of-scope areas
        # For now, we'll return the basic structure
        
        return parameters
    
    def check_hypothesis_proximity(self,
                                 hypothesis: Dict[str, Any],
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check how closely a hypothesis aligns with research parameters
        
        Args:
            hypothesis: Hypothesis to evaluate
            parameters: Research parameters to check against
            
        Returns:
            Dictionary containing proximity check results
        """
        action = f"Checking proximity of hypothesis: {hypothesis.get('id', 'unknown')}"
        self.log_action(action, {
            "hypothesis_id": hypothesis.get('id', 'unknown'),
            "research_goal": parameters.get('research_goal', '')
        })
        
        # This would involve actual proximity checking using the LLM
        # For now, we'll return a placeholder structure
        return {
            "hypothesis": hypothesis,
            "research_goal": parameters.get('research_goal', ''),
            "goal_alignment_score": 0,  # 0-10 scale
            "preference_alignment_scores": {},  # preference -> score
            "constraint_compliance_scores": {},  # constraint -> score
            "attribute_relevance_scores": {},  # attribute -> score
            "overall_proximity_score": 0,  # 0-10 scale
            "alignment_analysis": "",
            "proximity_classification": ""  # high, medium, low
        }
    
    def calculate_hypothesis_similarity(self,
                                   h1: Dict[str, Any],
                                   h2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the similarity between two hypotheses
        
        Args:
            h1: First hypothesis
            h2: Second hypothesis
            
        Returns:
            Dictionary containing similarity results
        """
        action = f"Calculating similarity between hypotheses: {h1.get('id', 'unknown')} and {h2.get('id', 'unknown')}"
        self.log_action(action, {
            "h1_id": h1.get('id', 'unknown'),
            "h2_id": h2.get('id', 'unknown')
        })
        
        # In a real implementation:
        # 1. Compare core concepts and assumptions between hypotheses
        # 2. Identify shared elements and unique contributions
        # 3. Calculate semantic similarity score
        # 4. Generate explanation of relationship between hypotheses
        
        return {
            "hypothesis_1": h1,
            "hypothesis_2": h2,
            "similarity_score": 0,  # 0-10 scale
            "shared_concepts": [],
            "unique_concepts_h1": [],
            "unique_concepts_h2": [],
            "complementary_aspects": [],
            "relationship_type": "",  # competing, complementary, derivative, etc.
            "similarity_explanation": ""
        }
    
    def build_proximity_graph(self,
                           hypotheses: List[Dict[str, Any]],
                           research_goal: str) -> Dict[str, Any]:
        """
        Build a proximity graph showing similarities between all hypotheses
        
        Args:
            hypotheses: List of hypotheses to include in the graph
            research_goal: The research goal for context
            
        Returns:
            Dictionary containing proximity graph
        """
        action = f"Building proximity graph for {len(hypotheses)} hypotheses"
        self.log_action(action, {
            "hypothesis_count": len(hypotheses),
            "research_goal": research_goal
        })
        
        # In a real implementation:
        # 1. Calculate pairwise similarities between all hypotheses
        # 2. Create a graph structure with hypotheses as nodes
        # 3. Add edges between hypotheses with similarity scores as weights
        # 4. Identify clusters of related hypotheses
        
        # This would involve multiple similarity calculations
        # For now, we'll return a placeholder structure
        similarities = []
        
        # Calculate similarities between all pairs of hypotheses
        for i, h1 in enumerate(hypotheses):
            for j, h2 in enumerate(hypotheses):
                if i < j:  # Only calculate for unique pairs
                    similarity = self.calculate_hypothesis_similarity(h1, h2)
                    similarities.append(similarity)
        
        return {
            "nodes": hypotheses,
            "edges": similarities,
            "clusters": [],  # Groups of related hypotheses
            "central_concepts": [],  # Concepts that connect multiple hypotheses
            "diversity_score": 0,  # Measure of overall diversity in the graph
            "core_areas": [],  # Areas of research with high concentration of hypotheses
            "unexplored_areas": [],  # Areas with minimal hypothesis coverage
            "graph_visualization_data": {}  # Data for visualizing the graph
        }
    
    def batch_proximity_check(self,
                            hypotheses: List[Dict[str, Any]],
                            parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check multiple hypotheses against research parameters and build a proximity graph
        
        Args:
            hypotheses: List of hypotheses to evaluate
            parameters: Research parameters to check against
            
        Returns:
            Dictionary containing batch proximity check results and graph
        """
        action = f"Batch checking proximity for {len(hypotheses)} hypotheses"
        self.log_action(action, {
            "hypothesis_count": len(hypotheses),
            "research_goal": parameters.get('research_goal', '')
        })
        
        # This would involve multiple proximity checks
        # For now, we'll return a placeholder structure
        results = []
        for hypothesis in hypotheses:
            # For each hypothesis, perform a proximity check
            result = self.check_hypothesis_proximity(hypothesis, parameters)
            results.append(result)
        
        # Sort by overall proximity score (descending)
        sorted_results = sorted(
            results, 
            key=lambda x: x.get('overall_proximity_score', 0), 
            reverse=True
        )
        
        # Calculate average score
        total_score = sum(r.get('overall_proximity_score', 0) for r in results)
        avg_score = total_score / len(results) if results else 0
        
        # Count hypotheses in each proximity category
        high_count = sum(1 for r in results if r.get('proximity_classification') == 'high')
        medium_count = sum(1 for r in results if r.get('proximity_classification') == 'medium')
        low_count = sum(1 for r in results if r.get('proximity_classification') == 'low')
        
        # Build proximity graph
        proximity_graph = self.build_proximity_graph(hypotheses, parameters.get('research_goal', ''))
        
        return {
            "sorted_results": sorted_results,
            "highest_proximity": sorted_results[0] if sorted_results else None,
            "lowest_proximity": sorted_results[-1] if sorted_results else None,
            "average_proximity_score": avg_score,
            "proximity_distribution": {
                "high": high_count,
                "medium": medium_count,
                "low": low_count
            },
            "proximity_graph": proximity_graph,
            "alignment_summary": "",
            "diverse_coverage_assessment": ""  # Assessment of how well the hypotheses cover diverse aspects
        }
    
    def provide_realignment_suggestions(self,
                                      hypothesis: Dict[str, Any],
                                      proximity_check: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide suggestions to bring a hypothesis into better alignment
        
        Args:
            hypothesis: Hypothesis to realign
            proximity_check: Results from proximity check
            
        Returns:
            Dictionary containing realignment suggestions
        """
        action = f"Providing realignment suggestions for hypothesis: {hypothesis.get('id', 'unknown')}"
        self.log_action(action, {
            "hypothesis_id": hypothesis.get('id', 'unknown'),
            "proximity_score": proximity_check.get('overall_proximity_score', 0)
        })
        
        # This would involve actual suggestion generation using the LLM
        # For now, we'll return a placeholder structure
        return {
            "hypothesis": hypothesis,
            "proximity_score": proximity_check.get('overall_proximity_score', 0),
            "misalignment_areas": [],
            "suggested_modifications": [],
            "realigned_hypothesis_draft": "",
            "expected_proximity_improvement": 0
        }