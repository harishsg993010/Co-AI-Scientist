from typing import List, Dict, Any, Optional, Tuple
import uuid
import logging
import math
import random
from datetime import datetime

from models import Hypothesis, HypothesisStatus
from config import MAX_HYPOTHESES, ELO_K_FACTOR

logger = logging.getLogger(__name__)


class HypothesisManager:
    """Manager for handling hypothesis generation, evaluation, and refinement"""
    
    def __init__(self, max_hypotheses: int = MAX_HYPOTHESES):
        """
        Initialize the hypothesis manager
        
        Args:
            max_hypotheses: Maximum number of hypotheses to maintain
        """
        self.hypotheses = {}  # Dictionary of hypothesis_id -> Hypothesis
        self.max_hypotheses = max_hypotheses
        self.ranked_ids = []  # Ranked list of hypothesis IDs
    
    def add_hypothesis(self, text: str) -> Hypothesis:
        """
        Add a new hypothesis
        
        Args:
            text: Hypothesis text
            
        Returns:
            The created Hypothesis object
        """
        hypothesis_id = str(uuid.uuid4())
        hypothesis = Hypothesis(
            id=hypothesis_id,
            text=text,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status=HypothesisStatus.GENERATED
        )
        
        self.hypotheses[hypothesis_id] = hypothesis
        self.ranked_ids.append(hypothesis_id)
        
        # Trim if exceeding maximum
        if len(self.hypotheses) > self.max_hypotheses:
            self._trim_hypotheses()
        
        logger.info(f"Added new hypothesis: {text[:50]}...")
        return hypothesis
    
    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """
        Get a hypothesis by ID
        
        Args:
            hypothesis_id: Hypothesis ID
            
        Returns:
            The Hypothesis object or None if not found
        """
        return self.hypotheses.get(hypothesis_id)
    
    def get_all_hypotheses(self) -> List[Hypothesis]:
        """
        Get all hypotheses
        
        Returns:
            List of all Hypothesis objects
        """
        return list(self.hypotheses.values())
    
    def get_ranked_hypotheses(self) -> List[Hypothesis]:
        """
        Get hypotheses in ranked order
        
        Returns:
            List of Hypothesis objects in ranked order
        """
        return [self.hypotheses[h_id] for h_id in self.ranked_ids if h_id in self.hypotheses]
    
    def update_hypothesis_status(self, hypothesis_id: str, status: HypothesisStatus) -> bool:
        """
        Update a hypothesis status
        
        Args:
            hypothesis_id: Hypothesis ID
            status: New status
            
        Returns:
            True if successful, False otherwise
        """
        hypothesis = self.get_hypothesis(hypothesis_id)
        if not hypothesis:
            return False
        
        hypothesis.status = status
        hypothesis.updated_at = datetime.now()
        return True
    
    def refine_hypothesis(self, hypothesis_id: str, new_text: str, reason: str) -> Optional[Hypothesis]:
        """
        Refine a hypothesis
        
        Args:
            hypothesis_id: Hypothesis ID
            new_text: New hypothesis text
            reason: Reason for refinement
            
        Returns:
            The refined Hypothesis object or None if not found
        """
        hypothesis = self.get_hypothesis(hypothesis_id)
        if not hypothesis:
            return None
        
        hypothesis.add_refinement(new_text, reason)
        return hypothesis
    
    def compare_hypotheses(self, h1_id: str, h2_id: str, winner_id: Optional[str] = None) -> Tuple[int, int]:
        """
        Compare two hypotheses and update their Elo ratings
        
        Args:
            h1_id: First hypothesis ID
            h2_id: Second hypothesis ID
            winner_id: ID of the winning hypothesis (None for a draw)
            
        Returns:
            Tuple of (new_rating_h1, new_rating_h2)
        """
        h1 = self.get_hypothesis(h1_id)
        h2 = self.get_hypothesis(h2_id)
        
        if not h1 or not h2:
            logger.warning(f"Cannot compare hypotheses: one or both not found ({h1_id}, {h2_id})")
            return (0, 0)
        
        # Elo rating calculation
        r1 = 10 ** (h1.elo_rating / 400)
        r2 = 10 ** (h2.elo_rating / 400)
        
        e1 = r1 / (r1 + r2)  # Expected score for h1
        e2 = r2 / (r1 + r2)  # Expected score for h2
        
        # Actual scores
        if winner_id is None:
            # Draw
            s1, s2 = 0.5, 0.5
        elif winner_id == h1_id:
            # h1 wins
            s1, s2 = 1.0, 0.0
        else:
            # h2 wins
            s1, s2 = 0.0, 1.0
        
        # Update ratings
        new_rating_h1 = h1.elo_rating + ELO_K_FACTOR * (s1 - e1)
        new_rating_h2 = h2.elo_rating + ELO_K_FACTOR * (s2 - e2)
        
        h1.elo_rating = round(new_rating_h1)
        h2.elo_rating = round(new_rating_h2)
        
        # Update rankings
        self._update_rankings()
        
        return (h1.elo_rating, h2.elo_rating)
    
    def run_tournament(self, num_comparisons: int = 10) -> List[Tuple[str, int]]:
        """
        Run a tournament to rank hypotheses
        
        Args:
            num_comparisons: Number of pairwise comparisons to make
            
        Returns:
            List of (hypothesis_id, elo_rating) tuples in ranked order
        """
        hypothesis_ids = list(self.hypotheses.keys())
        
        if len(hypothesis_ids) < 2:
            return [(h_id, self.hypotheses[h_id].elo_rating) for h_id in hypothesis_ids]
        
        # Run pairwise comparisons
        for _ in range(min(num_comparisons, len(hypothesis_ids) * (len(hypothesis_ids) - 1) // 2)):
            # Pick two different hypotheses
            h1_id, h2_id = random.sample(hypothesis_ids, 2)
            
            # Simulate a comparison (in real implementation, this would be done by an agent)
            h1 = self.hypotheses[h1_id]
            h2 = self.hypotheses[h2_id]
            
            # Simple heuristic: higher score wins (in real implementation, this would be more sophisticated)
            if h1.score > h2.score:
                winner_id = h1_id
            elif h2.score > h1.score:
                winner_id = h2_id
            else:
                winner_id = None  # Draw
            
            # Update Elo ratings
            self.compare_hypotheses(h1_id, h2_id, winner_id)
        
        # Update rankings
        self._update_rankings()
        
        # Return ranked list of (hypothesis_id, elo_rating) tuples
        return [(h_id, self.hypotheses[h_id].elo_rating) for h_id in self.ranked_ids]
    
    def _update_rankings(self):
        """Update the ranked_ids list based on Elo ratings"""
        self.ranked_ids = sorted(
            self.hypotheses.keys(),
            key=lambda h_id: self.hypotheses[h_id].elo_rating,
            reverse=True
        )
    
    def _trim_hypotheses(self):
        """Remove lowest-ranked hypotheses to maintain maximum count"""
        # Remove hypotheses from the bottom of the ranking
        to_remove = len(self.hypotheses) - self.max_hypotheses
        
        if to_remove <= 0:
            return
        
        # Get the lowest-ranked hypothesis IDs
        lowest_ranked = self.ranked_ids[-to_remove:]
        
        # Remove from hypotheses dictionary and ranked_ids list
        for h_id in lowest_ranked:
            if h_id in self.hypotheses:
                logger.info(f"Removing hypothesis: {self.hypotheses[h_id].text[:50]}...")
                del self.hypotheses[h_id]
        
        # Update rankings
        self._update_rankings()
