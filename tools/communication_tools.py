"""
Communication tools for agent collaboration in the research process.
"""

from typing import Dict, Any, List, Optional
import logging
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Define Pydantic models for tool schemas
class AskQuestionToolSchema(BaseModel):
    """Schema for Ask Question to Coworker Tool"""
    question: str = Field(description="The question to ask")
    context: str = Field(description="The context for the question")
    coworker: str = Field(description="The role/name of the coworker to ask")

class AskCoworkerTool(BaseTool):
    """Tool for asking questions to other agents in the research crew"""
    
    name: str = "Ask Coworker"
    description: str = "Ask a specific question to one of the following coworkers: Inventor and Ideation Specialist"
    args_schema: type[BaseModel] = AskQuestionToolSchema
    
    def _run(self, question: str, context: str, coworker: str) -> str:
        """
        Ask a question to a specific coworker
        
        Args:
            question: The question to ask
            context: The context for the question
            coworker: The role/name of the coworker to ask
            
        Returns:
            The response from the coworker
        """
        # Validate the coworker parameter
        valid_coworkers = ["inventor", "ideation specialist", "inventor and ideation specialist"]
        coworker_lower = coworker.lower()
        
        if coworker_lower not in valid_coworkers:
            return f"Error: Unknown coworker '{coworker}'. Available coworkers are: Inventor and Ideation Specialist"
        
        # Log the interaction
        logger.info(f"Question to {coworker}: {question}")
        logger.info(f"Context: {context}")
        
        # In a real implementation, this would route the question to the appropriate agent
        # For now, we'll provide a simulated response
        
        # Simulated responses based on coworker and question content
        if "inventor" in coworker_lower:
            if "challenges" in question.lower() or "opportunities" in question.lower():
                return """
                As the Inventor and Ideation Specialist, I've analyzed the research topic "Impact of social media on adolescent mental health" and identified the following key challenges and opportunities for innovation:

                KEY CHALLENGES:
                1. Causality vs. Correlation: Traditional research struggles to establish clear causal relationships between social media use and mental health outcomes.
                2. Measurement Limitations: Current assessment tools may not capture the nuanced ways social media affects adolescent psychology.
                3. Rapid Platform Evolution: Social media platforms change frequently, making longitudinal studies difficult.
                4. Individual Variability: Effects vary greatly among adolescents, making generalizations problematic.
                5. Multiple Interacting Factors: Social media use interacts with numerous other variables (socioeconomic status, family dynamics, etc.).

                OPPORTUNITIES FOR INNOVATION:
                1. Real-time Data Collection: Develop new methods to capture in-the-moment experiences rather than retrospective reporting.
                2. Pattern Recognition: Use advanced analytics to identify subtle patterns in usage that correlate with mental health changes.
                3. Personalized Impact Assessment: Create tools that account for individual differences in vulnerability and resilience.
                4. Intervention Design: Develop and test novel interventions that promote healthy social media use.
                5. Cross-disciplinary Integration: Combine insights from psychology, neuroscience, computer science, and sociology.

                These challenges and opportunities provide a foundation for developing creative research approaches that could lead to breakthroughs in understanding this complex topic.
                """
            else:
                return "I need more specific information about what aspect of the research topic you'd like me to address. Could you please clarify your question?"
        else:
            return "I'm not sure how to respond to this question in my role. Could you please direct your question to a different specialist or rephrase it?"


class DelegateWorkToolSchema(BaseModel):
    """Schema for Delegate Work to Coworker Tool"""
    task: str = Field(description="The task to delegate")
    context: str = Field(description="The context for the task")
    coworker: str = Field(description="The role/name of the coworker to delegate to")

class DelegateWorkTool(BaseTool):
    """Tool for delegating work to other agents in the research crew"""
    
    name: str = "Delegate Work"
    description: str = "Delegate a specific task to one of the following coworkers: Inventor and Ideation Specialist"
    args_schema: type[BaseModel] = DelegateWorkToolSchema
    
    def _run(self, task: str, context: str, coworker: str) -> str:
        """
        Delegate a task to a specific coworker
        
        Args:
            task: The task to delegate
            context: The context for the task
            coworker: The role/name of the coworker to delegate to
            
        Returns:
            The response from the coworker
        """
        # Validate the coworker parameter
        valid_coworkers = ["inventor", "ideation specialist", "inventor and ideation specialist"]
        coworker_lower = coworker.lower()
        
        if coworker_lower not in valid_coworkers:
            return f"Error: coworker mentioned not found, it must be one of the following options:\n- inventor and ideation specialist"
        
        # Log the interaction
        logger.info(f"Task delegated to {coworker}: {task}")
        logger.info(f"Context: {context}")
        
        # In a real implementation, this would route the task to the appropriate agent
        # For now, we'll provide a simulated response for the Inventor Agent
        
        if "generate" in task.lower() and "approaches" in task.lower() and "social media" in task.lower() and "mental health" in task.lower():
            return """
            # Innovative Research Approaches for Social Media Impact on Adolescent Mental Health

            ## 1. Digital Twin Mental Modeling
            
            **Title:** Digital Twin Mental Modeling: Parallel Digital and Psychological Profiles
            
            **Creative Strategy:** Analogy (from digital twins in engineering)
            
            **Description:** Create digital twin models of adolescents' mental states that evolve alongside their actual social media use. These computational models would simulate how different patterns of social media engagement might affect mental health trajectories.
            
            **Difference from Conventional Thinking:** Instead of retrospective analysis, this approach creates forward-looking predictive models that can simulate different interventions and usage patterns, allowing for personalized recommendations before negative impacts occur.
            
            **Potential Applications:**
            - Personalized risk assessment tools for clinicians and parents
            - Simulation platform for testing interventions without risking adolescent mental health
            - Educational tool showing adolescents potential future impacts of their usage patterns
            
            **Challenges and Solutions:**
            - Challenge: Accurate modeling of complex psychological processes
              Solution: Iterative model refinement using machine learning on longitudinal data
            - Challenge: Privacy concerns
              Solution: Develop anonymization techniques and strict ethical guidelines for data use
            
            ## 2. Neuroadaptive Content Investigation
            
            **Title:** Neuroadaptive Content Investigation: Measuring Neural Plasticity Changes
            
            **Creative Strategy:** Combination (neuroscience + content analysis)
            
            **Description:** Investigate how specific types of social media content trigger neuroplastic changes in adolescent brains using a combination of content analysis, neuroimaging, and longitudinal assessment.
            
            **Difference from Conventional Thinking:** Moves beyond self-reported mental health impacts to directly measure biological changes that occur in response to different content types and engagement patterns.
            
            **Potential Applications:**
            - Content classification system based on neurological impact
            - Development of "brain-healthy" content design principles
            - Targeted interventions for specific neural vulnerability patterns
            
            **Challenges and Solutions:**
            - Challenge: Expensive neuroimaging technology
              Solution: Begin with smaller studies and seek interdisciplinary funding
            - Challenge: Attributing causality
              Solution: Design controlled exposure studies with clear baseline measurements
            
            ## 3. Inverted Usage Paradigm
            
            **Title:** Inverted Usage Paradigm: Studying the Benefits of Strategic Disengagement
            
            **Creative Strategy:** Contradiction (challenging assumption that usage must be reduced)
            
            **Description:** Rather than focusing on reducing social media use, study the strategic implementation of specific usage patterns designed to enhance mental health, including structured engagement and disengagement cycles.
            
            **Difference from Conventional Thinking:** Shifts from the deficit model (social media as harmful) to an enhancement model (how specific usage patterns can be beneficial), challenging the assumption that less usage is always better.
            
            **Potential Applications:**
            - Development of "mental health optimized" usage schedules
            - Social media platforms with built-in wellness architecture
            - School-based digital wellness curricula
            
            **Challenges and Solutions:**
            - Challenge: Overcoming existing negative perceptions
              Solution: Focus on measurable positive outcomes in pilot studies
            - Challenge: Individual variation in optimal patterns
              Solution: Develop personalization frameworks based on personality and usage preferences
            
            ## 4. Cross-Generational Impact Assessment
            
            **Title:** Cross-Generational Impact Assessment: Intergenerational Social Media Effects
            
            **Creative Strategy:** Pattern matching (from epigenetic and cross-generational trauma research)
            
            **Description:** Study how parental social media usage patterns and attitudes influence adolescent mental health outcomes, and how these might create intergenerational patterns of digital behavior and psychological impacts.
            
            **Difference from Conventional Thinking:** Expands the focus beyond individual adolescents to family systems and intergenerational influences, recognizing social media use as a family-wide phenomenon.
            
            **Potential Applications:**
            - Family-based digital wellness interventions
            - Parental education programs on modeling healthy digital behavior
            - Predictive tools for identifying at-risk families based on parental usage patterns
            
            **Challenges and Solutions:**
            - Challenge: Complex multivariate analysis required
              Solution: Utilize advanced statistical methods and longitudinal family studies
            - Challenge: Changing family dynamics
              Solution: Include adaptability measures in study design
            
            ## 5. Embodied Digital Experience Mapping
            
            **Title:** Embodied Digital Experience Mapping: The Physical Manifestation of Virtual Experiences
            
            **Creative Strategy:** Abstraction (focusing on embodied cognition principles)
            
            **Description:** Map how social media experiences translate into physical and physiological responses, creating a comprehensive understanding of how digital interactions manifest in bodily experiences and how these affect mental health.
            
            **Difference from Conventional Thinking:** Bridges the artificial divide between "virtual" and "real" experiences by focusing on how digital interactions create real physiological and neurological changes.
            
            **Potential Applications:**
            - Biofeedback systems integrated with social media platforms
            - Somatic awareness training for healthy engagement
            - Development of physically grounding interfaces for social media
            
            **Challenges and Solutions:**
            - Challenge: Measuring subtle physiological responses
              Solution: Develop sensitive, non-invasive measurement techniques
            - Challenge: Individual variation in embodied responses
              Solution: Create typologies of embodied reactions to guide personalized approaches
            
            These five approaches represent innovative directions for researching the impact of social media on adolescent mental health, each offering unique insights and practical applications that could lead to significant advances in understanding and intervention.
            """
        else:
            return "I need more specific information about what you'd like me to work on. Could you please clarify the task?"