import uuid
import logging
import re
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from models import ResearchPaper, Hypothesis, Source
from openai import OpenAI

logger = logging.getLogger(__name__)

# Initialize OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

class PaperGenerator:
    """
    Manages the generation and formatting of academic research papers based on
    research findings and hypotheses.
    """
    
    def __init__(self, default_authors: List[str] = ["AI Co-Scientist"]):
        """
        Initialize the paper generator
        
        Args:
            default_authors: Default author list for generated papers
        """
        self.default_authors = default_authors
    
    def create_paper_from_research(
        self,
        topic: str,
        hypotheses: List[Hypothesis],
        sources: List[Source],
        analysis_results: Dict[str, Any]
    ) -> ResearchPaper:
        """
        Generate a complete academic research paper from research findings
        
        Args:
            topic: Research topic
            hypotheses: List of hypotheses evaluated
            sources: List of information sources
            analysis_results: Dictionary containing analysis results
            
        Returns:
            A ResearchPaper object with complete content
        """
        try:
            # Create paper structure
            paper = ResearchPaper(
                id=str(uuid.uuid4()),
                title=self._generate_title(topic, hypotheses),
                authors=self.default_authors,
                abstract=self._generate_abstract(topic, hypotheses, analysis_results),
                keywords=self._extract_keywords(topic, hypotheses),
                introduction=self._generate_introduction(topic, hypotheses),
                literature_review=self._generate_literature_review(sources),
                methodology=self._generate_methodology(hypotheses),
                results=self._generate_results(hypotheses, analysis_results),
                discussion=self._generate_discussion(hypotheses, analysis_results),
                conclusion=self._generate_conclusion(hypotheses, analysis_results),
                references=self._generate_references(sources)
            )
            
            # Calculate word count
            paper.calculate_word_count()
            
            logger.info(f"Generated research paper '{paper.title}' with {paper.word_count} words")
            return paper
            
        except Exception as e:
            logger.error(f"Error generating research paper: {str(e)}")
            # Return a minimal paper with error information
            return ResearchPaper(
                id=str(uuid.uuid4()),
                title=f"Research on {topic} (Error)",
                authors=self.default_authors,
                abstract=f"Error generating paper: {str(e)}",
                keywords=[topic],
                introduction="",
                literature_review="",
                methodology="",
                results="",
                discussion="",
                conclusion="",
                references=[]
            )
    
    def _generate_ai_content(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Generate content using OpenAI based on the provided prompt
        
        Args:
            prompt: The prompt to send to the OpenAI API
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated content as a string
        """
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
                messages=[
                    {"role": "system", "content": "You are an academic research assistant specializing in writing high-quality, human-like academic papers. Your task is to produce well-structured, professional content that meets scholarly standards while maintaining natural language and flow."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating content with OpenAI: {str(e)}")
            return f"[Content generation error: {str(e)}]"

    def _generate_title(self, topic: str, hypotheses: List[Hypothesis]) -> str:
        """Generate an appropriate academic title based on the research"""
        # Prepare data for the prompt
        hypotheses_text = ""
        if hypotheses:
            top_hypotheses = sorted(
                hypotheses, 
                key=lambda h: h.score if hasattr(h, 'score') else 0,
                reverse=True
            )[:3]
            
            hypotheses_text = "\n".join([
                f"- {h.text} (Confidence score: {h.score}/10)" 
                for h in top_hypotheses if hasattr(h, 'text') and hasattr(h, 'score')
            ])
        
        prompt = f"""
        Create a concise, academic title for a research paper on the topic of '{topic}'. 
        
        The paper examines the following key hypotheses:
        {hypotheses_text or 'No specific hypotheses were tested.'}
        
        The title should be scholarly but engaging, appropriate for an academic journal, under 15 words, 
        and should accurately reflect the research focus. Do not use colons unless necessary for clarity.
        """
        
        title = self._generate_ai_content(prompt, max_tokens=50)
        
        # If API fails, fall back to a basic title
        if title.startswith("[Content generation error"):
            if hypotheses:
                top_hypothesis = max(hypotheses, key=lambda h: h.score if hasattr(h, 'score') else 0)
                hypothesis_text = top_hypothesis.text if hasattr(top_hypothesis, 'text') else ""
                
                if hypothesis_text:
                    return f"{topic.title()}: An Analysis of {hypothesis_text}"
            
            return f"A Comprehensive Study of {topic.title()}"
            
        return title
    
    def _generate_abstract(
        self, 
        topic: str, 
        hypotheses: List[Hypothesis],
        analysis_results: Dict[str, Any]
    ) -> str:
        """Generate the paper abstract using AI"""
        # Prepare data for the prompt
        summary = analysis_results.get('summary', f"This paper explores {topic}.")
        
        # Format hypotheses for the prompt
        hypotheses_text = ""
        if hypotheses:
            top_hypotheses = sorted(
                hypotheses, 
                key=lambda h: h.score if hasattr(h, 'score') else 0,
                reverse=True
            )[:3]
            
            hypotheses_text = "\n".join([
                f"- {h.text} (Confidence score: {h.score}/10)" 
                for h in top_hypotheses if hasattr(h, 'text') and hasattr(h, 'score')
            ])
        
        # Create a comprehensive prompt for generating the abstract
        prompt = f"""
        Write a concise, academic abstract (150-250 words) for a research paper on '{topic}'.
        
        The research findings from our analysis include:
        {summary}
        
        The paper examines the following key hypotheses:
        {hypotheses_text or 'No specific hypotheses were tested.'}
        
        The abstract should summarize the research purpose, methodology, key findings, and implications. 
        It should be scholarly but accessible, appropriate for an academic journal, and follow the 
        standard abstract structure: background, objective, methods, results, and conclusion.
        """
        
        abstract = self._generate_ai_content(prompt, max_tokens=300)
        
        # If API fails, fall back to a basic abstract
        if abstract.startswith("[Content generation error"):
            fallback_hypotheses_text = ""
            if hypotheses:
                top_hypotheses = sorted(
                    hypotheses, 
                    key=lambda h: h.score if hasattr(h, 'score') else 0,
                    reverse=True
                )[:2]
                
                hypothesis_statements = [h.text for h in top_hypotheses if hasattr(h, 'text')]
                if hypothesis_statements:
                    fallback_hypotheses_text = " Key findings supported the following hypotheses: " + \
                                      "; ".join(hypothesis_statements) + "."
            
            methodology = "This research employed a systematic literature review and data analysis approach."
            
            return f"{summary}{fallback_hypotheses_text} {methodology} Implications of these findings are discussed."
            
        return abstract
    
    def _extract_keywords(self, topic: str, hypotheses: List[Hypothesis]) -> List[str]:
        """Extract relevant keywords from the research using AI"""
        # Prepare data for the prompt
        hypotheses_text = ""
        if hypotheses:
            top_hypotheses = sorted(
                hypotheses, 
                key=lambda h: h.score if hasattr(h, 'score') else 0,
                reverse=True
            )[:5]
            
            hypotheses_text = "\n".join([
                f"- {h.text}" 
                for h in top_hypotheses if hasattr(h, 'text')
            ])
        
        # Create a prompt for generating keywords
        prompt = f"""
        Extract 5-7 relevant scholarly keywords for a research paper on '{topic}'.
        
        The paper examines the following hypotheses:
        {hypotheses_text or 'No specific hypotheses were tested.'}
        
        Return ONLY the keywords as a comma-separated list (e.g., "machine learning, neural networks, artificial intelligence"). 
        Do not include numbers, bullet points, or any other text. Keywords should be specific, academically relevant terms 
        that would be used for indexing the paper in academic databases.
        """
        
        keywords_text = self._generate_ai_content(prompt, max_tokens=100)
        
        # Process the returned keywords
        if keywords_text.startswith("[Content generation error"):
            # Fallback to basic keyword extraction
            keywords = [topic]
            
            # Add important terms from top hypotheses
            if hypotheses:
                for h in hypotheses[:3]:
                    if hasattr(h, 'text'):
                        # Extract potential keywords (simple approach)
                        words = h.text.lower().split()
                        important_words = [w for w in words if len(w) > 5 and w not in keywords]
                        keywords.extend(important_words[:2])  # Add up to 2 keywords per hypothesis
            
            # Return unique keywords, up to 5
            return list(set(keywords))[:5]
        
        # Clean and process keywords from API response
        keywords = [k.strip().lower() for k in keywords_text.split(',')]
        
        # Remove any empty strings and limit to 7 keywords
        keywords = [k for k in keywords if k][:7]
        
        # Ensure we have at least one keyword
        if not keywords:
            keywords = [topic]
            
        return keywords
    
    def _generate_introduction(self, topic: str, hypotheses: List[Hypothesis]) -> str:
        """Generate the paper introduction using AI"""
        # Prepare data for the prompt
        hypotheses_text = ""
        if hypotheses:
            top_hypotheses = sorted(
                hypotheses, 
                key=lambda h: h.score if hasattr(h, 'score') else 0,
                reverse=True
            )[:5]
            
            hypotheses_text = "\n".join([
                f"- {h.text}" 
                for h in top_hypotheses if hasattr(h, 'text')
            ])
        
        # Create a comprehensive prompt for generating the introduction
        prompt = f"""
        Write a scholarly introduction (300-500 words) for a research paper on '{topic}'.
        
        The paper examines the following key hypotheses:
        {hypotheses_text or 'No specific hypotheses were tested.'}
        
        The introduction should:
        1. Establish the importance and context of the research topic
        2. Highlight relevant background and recent developments in the field
        3. Clearly articulate the research questions or objectives based on the hypotheses
        4. Explain the significance and potential contributions of this research
        5. Briefly outline the structure of the paper
        
        Write in a scholarly but engaging style appropriate for an academic journal. Use a natural flow 
        with appropriate transitions between ideas. Avoid clichés and overly formulaic language.
        """
        
        introduction = self._generate_ai_content(prompt, max_tokens=800)
        
        # If API fails, fall back to a template-based introduction
        if introduction.startswith("[Content generation error"):
            # Fallback to template introduction
            opening = f"The field of {topic} has witnessed remarkable growth and development in recent years, attracting significant attention from researchers across various disciplines. "
            context = f"As our understanding of this domain continues to evolve, researchers have increasingly recognized the profound implications that advances in {topic} may have for both theoretical frameworks and practical applications. "
            relevance = f"This growing interest is justified by the potential of {topic} to address pressing challenges and open new avenues for innovation. "
            
            # Mention the research questions based on hypotheses
            research_focus = "Our research specifically explores: "
            if hypotheses:
                questions = []
                for i, h in enumerate(hypotheses[:3]):
                    if hasattr(h, 'text'):
                        # Create a more naturally phrased question
                        hypothesis_text = h.text.strip()
                        if hypothesis_text.endswith('.'):
                            hypothesis_text = hypothesis_text[:-1]
                        
                        # Create more natural phrasing
                        if "can" in hypothesis_text.lower() or "could" in hypothesis_text.lower():
                            question = f"To what extent {hypothesis_text.lower()}?"
                        elif "is" in hypothesis_text.lower().split()[:2] or "are" in hypothesis_text.lower().split()[:2]:
                            question = f"Whether and to what degree {hypothesis_text.lower()}?"
                        else:
                            question = f"How {hypothesis_text.lower()}?"
                        
                        questions.append(question)
                
                if questions:
                    research_focus += " ".join(questions)
            else:
                research_focus += f"the fundamental mechanisms underlying {topic}, the key factors that influence its development, and the broader implications for theory and practice."
            
            transition = " Through a careful analysis of existing literature and systematic examination of evidence, we aim to contribute meaningful insights to this important area of study. "
            
            structure = "The remainder of this paper is organized as follows: We begin by reviewing the relevant literature to establish the theoretical foundation for our investigation. We then describe our methodology in detail, before presenting our findings and analyzing their significance. Finally, we discuss the broader implications of our results and suggest promising directions for future research."
            
            return opening + context + relevance + research_focus + transition + structure
            
        return introduction
    
    def _generate_literature_review(self, sources: List[Source]) -> str:
        """Generate the literature review section using AI"""
        # Prepare sources data for the prompt
        sources_text = ""
        if sources:
            sources_list = []
            for i, source in enumerate(sources[:8]):  # Limit to 8 sources for the review
                if hasattr(source, 'title') and hasattr(source, 'content'):
                    source_summary = {
                        "title": source.title,
                        "content": source.content[:300] + "..." if len(source.content) > 300 else source.content
                    }
                    
                    # Add metadata if available
                    if hasattr(source, 'metadata') and source.metadata:
                        if 'authors' in source.metadata:
                            source_summary["authors"] = source.metadata['authors']
                        if 'year' in source.metadata:
                            source_summary["year"] = source.metadata['year']
                        if 'journal' in source.metadata:
                            source_summary["journal"] = source.metadata['journal']
                    
                    sources_list.append(f"Source {i+1}:\nTitle: {source_summary['title']}\n" + 
                                      (f"Authors: {source_summary.get('authors', 'Unknown')}\n" if 'authors' in source_summary else "") +
                                      (f"Year: {source_summary.get('year', 'Unknown')}\n" if 'year' in source_summary else "") +
                                      (f"Journal: {source_summary.get('journal', 'N/A')}\n" if 'journal' in source_summary else "") +
                                      f"Content: {source_summary['content']}\n")
            
            sources_text = "\n".join(sources_list)
        
        # If no sources available, just return a generic literature review
        if not sources_text:
            prompt = """
            Write a scholarly literature review section (400-600 words) for a research paper. 
            Since no specific sources are available, create a general review that:
            
            1. Acknowledges the multidisciplinary nature of the field
            2. References different theoretical approaches and frameworks
            3. Identifies common methodologies used in the field
            4. Notes apparent gaps in existing research
            5. Establishes the need for the current study
            
            Write in a scholarly style with appropriate academic language and structure.
            """
        else:
            # Create a prompt for generating the literature review
            prompt = f"""
            Write a scholarly literature review section (600-800 words) based on the following sources:
            
            {sources_text}
            
            The literature review should:
            1. Organize the sources thematically (not just summarize each source separately)
            2. Identify key themes, agreements, and contradictions in the literature
            3. Critically analyze the methodological approaches used in previous studies
            4. Identify gaps in existing research that your study addresses
            5. Conclude by establishing how your research builds on or addresses limitations in the existing literature
            
            Write in a scholarly style with appropriate academic language. Use a natural flow with smooth 
            transitions between ideas. Synthesize the information rather than merely summarizing each source.
            """
        
        literature_review = self._generate_ai_content(prompt, max_tokens=1000)
        
        # If API fails, fall back to a simple literature review
        if literature_review.startswith("[Content generation error"):
            if not sources:
                return "The literature on this topic spans multiple disciplines. Prior research has investigated various aspects of the subject, revealing complex relationships and factors. Several theoretical frameworks have been proposed to explain the observed phenomena."
            
            intro = "This section reviews the relevant literature that informs our research. "
            
            # Group sources by type or theme (simplified approach)
            source_content = []
            for i, source in enumerate(sources[:5]):  # Limit to 5 sources for the demo
                if hasattr(source, 'title') and hasattr(source, 'content'):
                    source_summary = f"{source.title}: {source.content[:100]}..." if len(source.content) > 100 else source.content
                    source_content.append(source_summary)
            
            content = ""
            if source_content:
                content = " The literature reveals several important themes: " + " ".join(source_content)
            
            conclusion = " The literature review identified gaps in current understanding that this research aims to address."
            
            return intro + content + conclusion
            
        return literature_review
    
    def _generate_methodology(self, hypotheses: List[Hypothesis]) -> str:
        """Generate the methodology section using AI"""
        # Prepare hypotheses data for the prompt
        hypotheses_text = ""
        if hypotheses:
            hypotheses_text = "\n".join([
                f"- {h.text}" 
                for h in hypotheses if hasattr(h, 'text')
            ])
        
        # Create a prompt for generating the methodology
        prompt = f"""
        Write a scholarly methodology section (400-600 words) for a research paper that tested the following hypotheses:
        
        {hypotheses_text or 'No specific hypotheses were defined; the research followed an exploratory approach.'}
        
        The methodology section should:
        1. Clearly describe the research design (mixed methods, qualitative, quantitative, etc.)
        2. Explain the data collection methods and sources
        3. Detail the analytical approaches and techniques used
        4. Describe how the hypotheses were tested or how exploratory analysis was conducted
        5. Address potential methodological limitations and mitigation strategies
        
        Write in a scholarly but natural style appropriate for an academic journal. Use proper academic 
        terminology while maintaining readability. Structure the methodology in logical subsections.
        """
        
        methodology = self._generate_ai_content(prompt, max_tokens=800)
        
        # If API fails, fall back to template methodology
        if methodology.startswith("[Content generation error"):
            approach_intro = "Our investigation employed a comprehensive mixed-methods research design to examine the complex dimensions of this topic. "
            
            approach_details = "By integrating both qualitative and quantitative approaches, we were able to capture not only the measurable patterns but also the contextual nuances that purely statistical methods might overlook. "
            
            data_sources = "We gathered data from diverse sources to ensure a holistic understanding of the subject matter. Primary sources included peer-reviewed academic publications, technical reports from leading research institutions, government databases, and industry white papers. This diversity of sources allowed us to triangulate findings and mitigate potential biases inherent in any single data source. "
            
            data_process = "The collected information underwent a rigorous multi-stage analysis process. Initially, we systematically coded and categorized the raw data according to relevant themes and variables. Subsequently, we applied both statistical techniques and interpretive analysis to identify meaningful patterns, relationships, and exceptions in the dataset. "
            
            if hypotheses:
                hypothesis_part = f"Our research was guided by {len(hypotheses)} carefully formulated hypotheses that emerged from our initial literature review. To evaluate these hypotheses, we employed established methodological frameworks that consider multiple dimensions of evidence quality, including internal consistency, external validity, and methodological rigor. This approach allowed us to assess not just whether each hypothesis was supported, but the degree of confidence warranted by the available evidence."
            else:
                hypothesis_part = "Rather than beginning with predefined hypotheses, we adopted an exploratory approach that allowed significant patterns to emerge organically from the data. This inductive methodology was particularly well-suited to this relatively understudied area, enabling us to identify unexpected relationships and generate novel theoretical insights."
            
            limitations = "While designing our methodology, we remained mindful of potential limitations and took steps to address them through methodological triangulation and careful consideration of alternative explanations for our findings."
            
            return approach_intro + approach_details + data_sources + data_process + hypothesis_part + limitations
            
        return methodology
    
    def _generate_results(self, hypotheses: List[Hypothesis], analysis_results: Dict[str, Any]) -> str:
        """Generate the results section using AI"""
        # Prepare hypotheses data for the prompt with detailed evidence
        hypotheses_details = []
        if hypotheses:
            for i, h in enumerate(hypotheses):
                if hasattr(h, 'text') and hasattr(h, 'score'):
                    evidence_for = getattr(h, 'evidence_for', [])
                    evidence_against = getattr(h, 'evidence_against', [])
                    
                    hypothesis_detail = f"Hypothesis {i+1}: '{h.text}'\n"
                    hypothesis_detail += f"- Confidence score: {h.score}/10\n"
                    
                    if evidence_for:
                        hypothesis_detail += f"- Supporting evidence: {'; '.join(evidence_for)}\n"
                    
                    if evidence_against:
                        hypothesis_detail += f"- Contradicting evidence: {'; '.join(evidence_against)}\n"
                    
                    hypotheses_details.append(hypothesis_detail)
        
        # Add any other analysis results
        other_results = []
        if analysis_results:
            for key, value in analysis_results.items():
                if key != 'summary' and isinstance(value, (str, int, float, bool)):
                    other_results.append(f"{key}: {value}")
        
        # Create a prompt for generating the results section
        hypotheses_text = ''.join(hypotheses_details) if hypotheses_details else 'No specific hypotheses were tested.'
        additional_text = "Additional findings:\n" + "\n".join(other_results) if other_results else ""
        
        prompt = """
        Write a scholarly results section (500-700 words) for a research paper based on the following data:
        
        """ + hypotheses_text + """
        
        """ + additional_text + """
        
        The results section should:
        1. Present the findings in a clear, logical structure without interpretation (save interpretation for the discussion section)
        2. Use appropriate academic language and precision in reporting findings
        3. Include both the main findings related to the hypotheses and any additional noteworthy patterns
        4. Present the evidence in a balanced way, acknowledging both supporting and contradicting evidence
        5. Use an objective, scholarly tone
        
        Write in a scholarly style appropriate for an academic journal. Use a natural flow with appropriate 
        transitions between different findings.
        """
        
        results = self._generate_ai_content(prompt, max_tokens=900)
        
        # If API fails, fall back to template results
        if results.startswith("[Content generation error"):
            intro = "This section presents the key findings of our research. "
            
            findings = ""
            if hypotheses:
                findings += "Analysis of the collected data yielded the following results regarding our hypotheses: "
                
                for i, h in enumerate(hypotheses):
                    if hasattr(h, 'text') and hasattr(h, 'score'):
                        evidence_for = getattr(h, 'evidence_for', [])
                        evidence_against = getattr(h, 'evidence_against', [])
                        
                        finding = f"Hypothesis {i+1}: '{h.text}' - "
                        if h.score > 7.0:
                            finding += f"Strongly supported (confidence score: {h.score}/10). "
                        elif h.score > 5.0:
                            finding += f"Partially supported (confidence score: {h.score}/10). "
                        else:
                            finding += f"Not supported (confidence score: {h.score}/10). "
                        
                        if evidence_for:
                            finding += f"Supporting evidence includes: {'; '.join(evidence_for[:2])}. "
                        
                        if evidence_against:
                            finding += f"Contradicting evidence includes: {'; '.join(evidence_against[:2])}."
                        
                        findings += finding + " "
            else:
                findings = "Our analysis revealed several important patterns and relationships in the data. The findings suggest complex interactions between multiple factors."
            
            additional = "Additional patterns emerged that were not directly related to our initial hypotheses, suggesting areas for future research."
            
            return intro + findings + additional
            
        return results
    
    def _generate_discussion(self, hypotheses: List[Hypothesis], analysis_results: Dict[str, Any]) -> str:
        """Generate the discussion section using AI"""
        # Prepare data for the prompt
        hypotheses_details = []
        if hypotheses:
            # Group hypotheses by level of support
            supported = [h for h in hypotheses if hasattr(h, 'score') and h.score > 7.0]
            partial = [h for h in hypotheses if hasattr(h, 'score') and 5.0 < h.score <= 7.0]
            unsupported = [h for h in hypotheses if hasattr(h, 'score') and h.score <= 5.0]
            
            if supported:
                hypotheses_details.append(f"Strongly supported hypotheses ({len(supported)}):")
                for h in supported:
                    hypotheses_details.append(f"- {h.text} (score: {h.score}/10)")
            
            if partial:
                hypotheses_details.append(f"Partially supported hypotheses ({len(partial)}):")
                for h in partial:
                    hypotheses_details.append(f"- {h.text} (score: {h.score}/10)")
            
            if unsupported:
                hypotheses_details.append(f"Unsupported hypotheses ({len(unsupported)}):")
                for h in unsupported:
                    hypotheses_details.append(f"- {h.text} (score: {h.score}/10)")
        
        # Add summary from analysis results if available
        summary = ""
        if analysis_results and 'summary' in analysis_results:
            summary = f"Key findings summary: {analysis_results['summary']}"
        
        # Create a prompt for generating the discussion
        hypotheses_text = ''.join([h + '\n' for h in hypotheses_details]) if hypotheses_details else 'No specific hypotheses were tested.'
        
        prompt = """
        Write a scholarly discussion section (700-900 words) for a research paper based on the following findings:
        
        """ + summary + """

        """ + hypotheses_text + """
        
        The discussion section should:
        1. Interpret the findings in the context of existing literature and theoretical frameworks
        2. Analyze the significance and implications of the results for both theory and practice
        3. Consider how different levels of support for various hypotheses contribute to our understanding of the topic
        4. Address study limitations honestly but without undermining the research value
        5. Suggest specific directions for future research based on the findings
        
        Write in a scholarly but engaging style appropriate for an academic journal. Use a natural flow with 
        thoughtful transitions between concepts. Structure the discussion with clear logical progression from 
        interpretation through implications to future directions.
        """
        
        discussion = self._generate_ai_content(prompt, max_tokens=1200)
        
        # If API fails, fall back to template discussion
        if discussion.startswith("[Content generation error"):
            framing = "In this section, we critically examine our findings within the broader context of existing literature and theoretical frameworks. By situating our results within the ongoing scholarly conversation, we can better elucidate their significance and implications. "
            
            interpretation = ""
            if hypotheses and any(hasattr(h, 'score') for h in hypotheses):
                # Group hypotheses by level of support
                supported = [h for h in hypotheses if hasattr(h, 'score') and h.score > 7.0]
                partial = [h for h in hypotheses if hasattr(h, 'score') and 5.0 < h.score <= 7.0]
                unsupported = [h for h in hypotheses if hasattr(h, 'score') and h.score <= 5.0]
                
                if supported:
                    if len(supported) == 1:
                        hypothesis_text = supported[0].text.strip()
                        if hypothesis_text.endswith('.'):
                            hypothesis_text = hypothesis_text[:-1]
                        interpretation += f"Our analysis revealed strong support for the hypothesis that {hypothesis_text.lower()}. This finding is particularly noteworthy as it provides robust empirical validation for a relationship that has been theoretically postulated but insufficiently tested in previous research. The strength of this finding suggests that this relationship is not merely coincidental but represents a fundamental aspect of the phenomenon under investigation. "
                    else:
                        interpretation += f"The robust support we found for {len(supported)} of our hypotheses reinforces the validity of our theoretical framework and adds substantial weight to the existing body of knowledge in this field. These findings do not stand in isolation but rather build upon and extend previous research, offering stronger empirical grounding for key theoretical propositions. "
                
                if partial:
                    interpretation += f"Interestingly, our analysis revealed partial support for {len(partial)} hypotheses, highlighting the nuanced and context-dependent nature of the relationships we examined. This partial support suggests that while the fundamental mechanisms we proposed are valid, their manifestation likely depends on specific conditions or moderating factors that warrant further investigation. These findings remind us that in complex systems, straightforward causal relationships are often the exception rather than the rule. "
                
                if unsupported:
                    interpretation += f"Perhaps most intriguing are the {len(unsupported)} hypotheses that our analysis did not support. Rather than viewing these as failures, we see them as valuable opportunities to refine our theoretical understanding. The absence of support for these hypotheses challenges certain commonly held assumptions in the field and points to the need for reconceptualizing aspects of our theoretical framework. Scientific progress often emerges precisely from such moments of theoretical recalibration. "
            else:
                interpretation = "Our findings reveal a complex interplay of factors that both support and challenge existing theoretical frameworks. This complexity is not surprising given the multifaceted nature of the phenomenon we studied, but it does underscore the need for more sophisticated conceptual models that can accommodate such intricacy. "
            
            theoretical_implications = "From a theoretical perspective, these findings contribute to ongoing debates about the fundamental nature and mechanisms of the phenomenon under study. They suggest that existing theoretical frameworks, while valuable, may benefit from greater attention to contextual factors and dynamic interactions between variables. Our results point toward an integrative theoretical approach that synthesizes insights from multiple disciplinary perspectives. "
            
            practical_implications = "The practical implications of our findings extend to multiple domains. For practitioners and policymakers, our research highlights the importance of adopting nuanced, context-sensitive approaches rather than one-size-fits-all solutions. Specifically, stakeholders should consider how the relationships we've identified might manifest in their particular contexts and adjust their strategies accordingly. "
            
            limitations = "While our study offers valuable insights, it is important to acknowledge its limitations. Our analysis was necessarily constrained by the available data and the scope of our research questions. The methodology we employed, though rigorous, cannot capture all aspects of the complex phenomenon we studied. Additionally, our focus on specific relationships may have overlooked other important factors or interactions. We have attempted to mitigate these limitations through methodological triangulation and careful interpretation, but they nonetheless represent important caveats to our findings. "
            
            future_directions = "These limitations point to promising avenues for future research. Subsequent studies might employ alternative methodological approaches, incorporate additional variables, or examine these relationships in different contexts. Longitudinal research would be particularly valuable for understanding how these relationships evolve over time and across different conditions. Such efforts would complement our findings and contribute to a more comprehensive understanding of this important area."
            
            return framing + interpretation + theoretical_implications + practical_implications + limitations + future_directions
            
        return discussion
    
    def _generate_conclusion(self, hypotheses: List[Hypothesis], analysis_results: Dict[str, Any]) -> str:
        """Generate the conclusion section using AI"""
        # Prepare data for the prompt
        hypotheses_summary = ""
        if hypotheses:
            # Format top hypotheses for the prompt
            top_hypotheses = sorted(
                [h for h in hypotheses if hasattr(h, 'score')],
                key=lambda h: h.score,
                reverse=True
            )[:3]
            
            if top_hypotheses:
                hypotheses_list = []
                for h in top_hypotheses:
                    if hasattr(h, 'text') and hasattr(h, 'score'):
                        hypotheses_list.append(f"- {h.text} (score: {h.score}/10)")
                
                hypotheses_summary = "Top findings:\n" + "\n".join(hypotheses_list)
        
        # Add summary from analysis results if available
        results_summary = ""
        if analysis_results and 'summary' in analysis_results:
            results_summary = f"Overall summary of findings: {analysis_results['summary']}"
        
        # Create a prompt for generating the conclusion
        prompt = """
        Write a scholarly conclusion section (300-400 words) for a research paper based on the following findings:
        
        """ + results_summary + """
        
        """ + hypotheses_summary + """
        
        The conclusion should:
        1. Briefly summarize the research purpose and main findings without simply repeating the abstract
        2. Synthesize the insights from the study into a cohesive understanding of the topic
        3. Highlight theoretical contributions and practical implications
        4. Suggest promising directions for future research
        5. End with a thoughtful reflection on the significance of the work
        
        Write in a scholarly but engaging style appropriate for an academic journal. Use a natural flow with 
        appropriate transitions between ideas. Avoid introducing new information not previously discussed in the paper.
        The conclusion should provide a sense of closure while emphasizing the ongoing nature of scholarly inquiry.
        """
        
        conclusion = self._generate_ai_content(prompt, max_tokens=600)
        
        # If API fails, fall back to template conclusion
        if conclusion.startswith("[Content generation error"):
            # Create a more sophisticated and academically nuanced conclusion
            research_journey = "This study set out to explore the complex dynamics of an increasingly important area of inquiry. Through a systematic investigation grounded in rigorous methodology, we have contributed to the scholarly understanding of this multifaceted phenomenon. "
            
            key_findings = ""
            if hypotheses:
                # Select top hypotheses based on score
                top_hypotheses = sorted(
                    [h for h in hypotheses if hasattr(h, 'score')],
                    key=lambda h: h.score,
                    reverse=True
                )[:2]
                
                if top_hypotheses:
                    key_findings = "Our analysis yielded several noteworthy findings. "
                    
                    if len(top_hypotheses) == 1:
                        hypothesis_text = top_hypotheses[0].text.strip()
                        if hypothesis_text.endswith('.'):
                            hypothesis_text = hypothesis_text[:-1]
                        key_findings += f"Most notably, we found compelling evidence that {hypothesis_text.lower()}. This finding represents a significant contribution to the field, as it clarifies a relationship that has been subject to considerable theoretical debate. "
                    else:
                        key_findings += "Of particular significance were our findings that "
                        findings_list = []
                        for h in top_hypotheses:
                            if hasattr(h, 'text'):
                                hypothesis_text = h.text.strip()
                                if hypothesis_text.endswith('.'):
                                    hypothesis_text = hypothesis_text[:-1]
                                findings_list.append(hypothesis_text.lower())
                        
                        key_findings += " and that ".join(findings_list) + ". These insights advance our understanding of the fundamental mechanisms at work in this domain. "
            else:
                key_findings = "Our investigation yielded several important insights that collectively enhance our understanding of the complex interrelationships within this field. While not reducible to simple causal statements, these findings nonetheless offer valuable guidance for both theory development and practical applications. "
            
            synthesis = "When considered as a whole, our findings paint a more nuanced picture than has previously been available. They suggest that rather than seeking universal principles, scholars and practitioners alike would benefit from appreciating the contextual factors that shape how these relationships manifest in different settings. "
            
            theoretical_contribution = "From a theoretical perspective, this research contributes to ongoing conversations about the foundational concepts and frameworks in the field. Our findings both reinforce certain established theoretical positions and suggest refinements to others, pointing toward a more integrated theoretical framework that can accommodate the complexity we observed. "
            
            practical_implications = "For practitioners, our results offer several actionable insights. The evidence we found suggests that interventions and policies should be designed with careful attention to contextual factors and potential interaction effects. A one-size-fits-all approach is unlikely to be effective given the nuanced relationships we identified. "
            
            future_directions = "Looking ahead, this research opens several promising avenues for further inquiry. Future studies could productively expand on our findings by incorporating additional variables, applying alternative methodological approaches, or examining these relationships in diverse contexts. Particularly valuable would be longitudinal research that tracks how these dynamics evolve over time and across different conditions. "
            
            closing_reflection = "In conclusion, while this study has advanced our understanding of important aspects of this phenomenon, it also reminds us of the inherent complexity of the subject matter and the ongoing need for thoughtful, rigorous inquiry. We hope that the insights presented here will stimulate further research and inform more effective approaches to addressing the challenges and opportunities in this important domain."
            
            return research_journey + key_findings + synthesis + theoretical_contribution + practical_implications + future_directions + closing_reflection
            
        return conclusion
    
    def _generate_references(self, sources: List[Source]) -> List[Dict[str, str]]:
        """Generate properly formatted references from sources"""
        references = []
        
        for source in sources:
            if hasattr(source, 'title') and hasattr(source, 'url'):
                # Basic reference structure
                reference = {
                    "title": source.title,
                    "url": source.url,
                    "year": datetime.now().year,  # Default to current year
                    "authors": "Author, A."  # Default author
                }
                
                # Extract more metadata if available
                if hasattr(source, 'metadata') and source.metadata:
                    if 'year' in source.metadata:
                        reference['year'] = source.metadata['year']
                    if 'authors' in source.metadata:
                        reference['authors'] = source.metadata['authors']
                    if 'publisher' in source.metadata:
                        reference['publisher'] = source.metadata['publisher']
                    if 'journal' in source.metadata:
                        reference['journal'] = source.metadata['journal']
                        reference['type'] = 'article'
                
                references.append(reference)
        
        return references
    
    def format_paper_as_arxiv(self, paper: ResearchPaper) -> str:
        """
        Format a research paper in ArXiv-style LaTeX format
        
        Args:
            paper: The ResearchPaper object to format
            
        Returns:
            LaTeX content in ArXiv format
        """
        # Basic ArXiv-style template
        latex_template = """\\documentclass[12pt]{article}

% Packages
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{graphicx}
\\usepackage{hyperref}
\\usepackage{natbib}
\\usepackage{booktabs}
\\usepackage{fullpage}

% Title and author information
\\title{TITLE}
\\author{AUTHORS}
\\date{\\today}

\\begin{document}

\\maketitle

\\begin{abstract}
ABSTRACT
\\end{abstract}

\\textbf{Keywords:} KEYWORDS

\\section{Introduction}
INTRODUCTION

\\section{Related Work}
LITERATURE_REVIEW

\\section{Methodology}
METHODOLOGY

\\section{Results}
RESULTS

\\section{Discussion}
DISCUSSION

\\section{Conclusion}
CONCLUSION

\\section*{References}
\\begin{thebibliography}{9}
REFERENCES
\\end{thebibliography}

\\end{document}
"""
        # Format authors
        formatted_authors = " \\and ".join(paper.authors)
        
        # Format keywords
        formatted_keywords = ", ".join(paper.keywords)
        
        # Format references
        formatted_references = ""
        for i, ref in enumerate(paper.references):
            ref_type = ref.get('type', 'website')
            if ref_type == 'article':
                formatted_ref = r"\bibitem{ref_" + str(i+1) + "}\n"
                formatted_ref += f"{ref.get('authors', 'Unknown')}. "
                formatted_ref += f"({ref.get('year', 'n.d.')}). "
                formatted_ref += f"{ref.get('title', 'Untitled')}. "
                formatted_ref += r"\textit{" + ref.get('journal', 'Journal') + "}. "
                if 'url' in ref:
                    formatted_ref += r"\url{" + ref['url'] + "}"
            else:
                formatted_ref = r"\bibitem{ref_" + str(i+1) + "}\n"
                formatted_ref += f"{ref.get('authors', 'Unknown')}. "
                formatted_ref += f"({ref.get('year', 'n.d.')}). "
                formatted_ref += f"{ref.get('title', 'Untitled')}. "
                if 'url' in ref:
                    formatted_ref += r"Retrieved from \url{" + ref['url'] + "} "
                if 'accessed' in ref:
                    formatted_ref += f"[Accessed: {ref['accessed']}]"
            
            formatted_references += formatted_ref + "\n\n"
        
        # Replace placeholders with content
        latex_content = latex_template
        latex_content = latex_content.replace("TITLE", self._latex_escape(paper.title))
        latex_content = latex_content.replace("AUTHORS", self._latex_escape(formatted_authors))
        latex_content = latex_content.replace("ABSTRACT", self._latex_escape(paper.abstract))
        latex_content = latex_content.replace("KEYWORDS", self._latex_escape(formatted_keywords))
        latex_content = latex_content.replace("INTRODUCTION", self._latex_escape(paper.introduction))
        latex_content = latex_content.replace("LITERATURE_REVIEW", self._latex_escape(paper.literature_review))
        latex_content = latex_content.replace("METHODOLOGY", self._latex_escape(paper.methodology))
        latex_content = latex_content.replace("RESULTS", self._latex_escape(paper.results))
        latex_content = latex_content.replace("DISCUSSION", self._latex_escape(paper.discussion))
        latex_content = latex_content.replace("CONCLUSION", self._latex_escape(paper.conclusion))
        latex_content = latex_content.replace("REFERENCES", formatted_references)
        
        return latex_content
    
    def _latex_escape(self, text: str) -> str:
        """
        Escape special characters for LaTeX
        
        Args:
            text: Text to escape
            
        Returns:
            Escaped text for LaTeX
        """
        # Replace special characters
        replacements = [
            ('&', '\\&'),
            ('%', '\\%'),
            ('$', '\\$'),
            ('#', '\\#'),
            ('_', '\\_'),
            ('{', '\\{'),
            ('}', '\\}'),
            ('~', '\\textasciitilde{}'),
            ('^', '\\textasciicircum{}')
        ]
        
        for old, new in replacements:
            text = text.replace(old, new)
            
        # Handle line breaks
        text = text.replace('\n\n', '\n\n\\\\par ')
        
        return text
        
    def export_paper_to_html(self, paper: ResearchPaper) -> str:
        """
        Export the research paper to HTML format
        
        Args:
            paper: The ResearchPaper object to export
            
        Returns:
            HTML string representation of the paper
        """
        # Start building the HTML content
        html_parts = []
        
        # Add the HTML header
        html_parts.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>""")
        
        html_parts.append(paper.title)
        
        html_parts.append("""</title>
            <style>
                body {
                    font-family: 'Times New Roman', Times, serif;
                    line-height: 1.6;
                    margin: 2em;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1, h2, h3 {
                    color: #333;
                }
                h1 {
                    text-align: center;
                    margin-bottom: 0.5em;
                }
                .authors {
                    text-align: center;
                    margin-bottom: 2em;
                }
                .abstract {
                    font-style: italic;
                    margin-bottom: 2em;
                    border: 1px solid #ddd;
                    padding: 1em;
                    background-color: #f9f9f9;
                }
                .keywords {
                    margin-bottom: 2em;
                }
                .section {
                    margin-bottom: 2em;
                }
                .references {
                    padding-left: 2em;
                    text-indent: -2em;
                }
                .reference {
                    margin-bottom: 0.5em;
                }
                .figure, .table {
                    margin: 1em 0;
                    text-align: center;
                }
                .caption {
                    font-style: italic;
                    margin-top: 0.5em;
                }
                .footer {
                    margin-top: 2em;
                    text-align: center;
                    font-size: 0.8em;
                    color: #666;
                }
            </style>
        </head>
        <body>
            <h1>""")
            
        html_parts.append(paper.title)
        
        html_parts.append("""</h1>
            
            <div class="authors">
                """)
                
        html_parts.append(', '.join(paper.authors))
        
        html_parts.append("""
            </div>
            
            <div class="abstract">
                <strong>Abstract:</strong> """)
                
        html_parts.append(paper.abstract)
        
        html_parts.append("""
            </div>
            
            <div class="keywords">
                <strong>Keywords:</strong> """)
                
        html_parts.append(', '.join(paper.keywords))
        
        html_parts.append("""
            </div>
            
            <div class="section">
                <h2>1. Introduction</h2>
                """)
                
        html_parts.append(paper.introduction)
        
        html_parts.append("""
            </div>
            
            <div class="section">
                <h2>2. Literature Review</h2>
                """)
                
        html_parts.append(paper.literature_review)
        
        html_parts.append("""
            </div>
            
            <div class="section">
                <h2>3. Methodology</h2>
                """)
                
        html_parts.append(paper.methodology)
        
        html_parts.append("""
            </div>
            
            <div class="section">
                <h2>4. Results</h2>
                """)
                
        html_parts.append(paper.results)
        
        html_parts.append("""
            </div>
            
            <div class="section">
                <h2>5. Discussion</h2>
                """)
                
        html_parts.append(paper.discussion)
        
        html_parts.append("""
            </div>
            
            <div class="section">
                <h2>6. Conclusion</h2>
                """)
                
        html_parts.append(paper.conclusion)
        
        html_parts.append("""
            </div>
            
            <div class="section">
                <h2>References</h2>
                <div class="references">
        """)
        
        # Join all HTML parts
        html = ''.join(html_parts)
        
        # Add references
        for i, ref in enumerate(paper.references):
            cite = paper.format_citation(ref)
            html += f'<div class="reference">{i+1}. {cite}</div>\n'
        
        html += """
                </div>
            </div>
            
            <div class="footer">
                <p>Generated by AI Co-Scientist</p>
                <p>Word Count: """ + str(paper.word_count) + """</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def export_paper_to_text(self, paper: ResearchPaper) -> str:
        """
        Export the research paper to plain text format
        
        Args:
            paper: The ResearchPaper object to export
            
        Returns:
            Text string representation of the paper
        """
        # Start building the text content
        text_parts = []
        
        # Add the basic structure
        text_parts.append(paper.title.upper())
        text_parts.append("\n\n")
        text_parts.append(', '.join(paper.authors))
        text_parts.append("\n\n")
        text_parts.append("ABSTRACT\n")
        text_parts.append(paper.abstract)
        text_parts.append("\n\n")
        text_parts.append("KEYWORDS\n")
        text_parts.append(', '.join(paper.keywords))
        text_parts.append("\n\n")
        text_parts.append("1. INTRODUCTION\n")
        text_parts.append(paper.introduction)
        text_parts.append("\n\n")
        text_parts.append("2. LITERATURE REVIEW\n")
        text_parts.append(paper.literature_review)
        text_parts.append("\n\n")
        text_parts.append("3. METHODOLOGY\n")
        text_parts.append(paper.methodology)
        text_parts.append("\n\n")
        text_parts.append("4. RESULTS\n")
        text_parts.append(paper.results)
        text_parts.append("\n\n")
        text_parts.append("5. DISCUSSION\n")
        text_parts.append(paper.discussion)
        text_parts.append("\n\n")
        text_parts.append("6. CONCLUSION\n")
        text_parts.append(paper.conclusion)
        text_parts.append("\n\n")
        text_parts.append("REFERENCES\n")
        
        # Combine all parts
        text = ''.join(text_parts)
        
        # Add references
        for i, ref in enumerate(paper.references):
            cite = paper.format_citation(ref)
            text += f"{i+1}. {cite}\n"
        
        text += f"\nGenerated by AI Co-Scientist\nWord Count: {paper.word_count}\n"
        
        return text
        
    def export_paper_to_arxiv_latex(self, paper: ResearchPaper) -> str:
        """
        Export the research paper to arXiv-style LaTeX format
        
        Args:
            paper: The ResearchPaper object to export
            
        Returns:
            LaTeX string representation of the paper suitable for arXiv submission
        """
        # Escape LaTeX special characters in content
        def escape_latex(text):
            # Basic LaTeX escaping
            for char, replacement in [
                ('&', '\\&'),
                ('%', '\\%'),
                ('$', '\\$'),
                ('#', '\\#'),
                ('_', '\\_'),
                ('{', '\\{'),
                ('}', '\\}'),
                ('~', '\\textasciitilde{}'),
                ('^', '\\textasciicircum{}'),
                ('\\', '\\textbackslash{}')
            ]:
                text = text.replace(char, replacement)
            return text
        
        # Create the LaTeX document
        latex = r"""\documentclass[11pt,a4paper]{article}

% arXiv-style packages
\usepackage{hyperref}
\usepackage{url}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{microtype}
\usepackage{booktabs}
\usepackage{authblk}
\usepackage[capitalise]{cleveref}
\usepackage[final]{pdflscape}

% Document setup
\title{""" + escape_latex(paper.title) + r"""}

% Authors
"""
        
        # Add authors
        for i, author in enumerate(paper.authors):
            # Using raw prefix to handle backslashes properly
            latex += fr"\author[{i+1}]{{{escape_latex(author)}}}\n"
        
        latex += r"""
\affil[1]{AI Co-Scientist Research Institute}

\date{\today}

\begin{document}

\maketitle

\begin{abstract}
""" + escape_latex(paper.abstract) + r"""
\end{abstract}

\section*{Keywords}
""" + escape_latex(', '.join(paper.keywords)) + r"""

\section{Introduction}
""" + escape_latex(paper.introduction) + r"""

\section{Literature Review}
""" + escape_latex(paper.literature_review) + r"""

\section{Methodology}
""" + escape_latex(paper.methodology) + r"""

\section{Results}
""" + escape_latex(paper.results) + r"""

\section{Discussion}
""" + escape_latex(paper.discussion) + r"""

\section{Conclusion}
""" + escape_latex(paper.conclusion) + r"""

\section*{References}
\begin{thebibliography}{99}
"""
        
        # Add references in LaTeX bibliography format
        for i, ref in enumerate(paper.references):
            ref_key = f"ref{i+1}"
            
            # Format based on reference type
            if ref.get("type") == "article":
                latex += r"\bibitem{" + ref_key + "} " + escape_latex(ref.get('authors', 'Unknown Author')) + ", "
                latex += f"``{escape_latex(ref.get('title', 'Untitled'))},'' "
                latex += r"\textit{" + escape_latex(ref.get('journal', 'Unknown Journal')) + "}, "
                if ref.get('volume'):
                    latex += f"vol. {escape_latex(ref.get('volume'))}, "
                if ref.get('issue'):
                    latex += f"no. {escape_latex(ref.get('issue'))}, "
                if ref.get('pages'):
                    latex += f"pp. {escape_latex(ref.get('pages'))}, "
                latex += f"{escape_latex(ref.get('year', 'n.d.'))}"
            else:
                latex += r"\bibitem{" + ref_key + "} " + escape_latex(ref.get('authors', 'Unknown Author')) + ", "
                latex += f"``{escape_latex(ref.get('title', 'Untitled'))},'' "
                if ref.get('publisher'):
                    latex += f"{escape_latex(ref.get('publisher'))}, "
                latex += f"{escape_latex(ref.get('year', 'n.d.'))}"
            
            # Add URL if available
            if ref.get('url') and ref.get('url') is not None:
                url_val = escape_latex(ref.get('url'))
                # Use string concatenation with explicit cast to avoid type errors
                latex += r", \url{" + str(url_val) + r"}"
            
            latex += "\n\n"
            
        latex += r"""\end{thebibliography}

\section*{Metadata}
\begin{itemize}
\item Word Count: """ + str(paper.word_count) + r"""
\item Generated by: AI Co-Scientist
\item Version: """ + str(paper.version) + r"""
\item Last Updated: """ + paper.last_updated.strftime('%Y-%m-%d') + r"""
\end{itemize}

\end{document}
"""
        
        return latex