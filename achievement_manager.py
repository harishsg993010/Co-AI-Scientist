import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from models import (
    Achievement, 
    UserAchievement, 
    ResearchProfile, 
    AchievementCategory
)

logger = logging.getLogger(__name__)

# Default achievements
DEFAULT_ACHIEVEMENTS = [
    # Novice achievements (Getting started)
    Achievement(
        id="first_research",
        title="First Steps",
        description="Start your first research cycle",
        category=AchievementCategory.NOVICE,
        icon_name="lightbulb",
        points=10,
    ),
    Achievement(
        id="first_hypothesis",
        title="Hypothesis Formed",
        description="Generate your first research hypothesis",
        category=AchievementCategory.NOVICE,
        icon_name="flask",
        points=15,
    ),
    Achievement(
        id="first_paper",
        title="Published Author",
        description="Generate your first research paper",
        category=AchievementCategory.NOVICE,
        icon_name="file-text",
        points=25,
    ),
    
    # Explorer achievements (Discovering)
    Achievement(
        id="topic_variety",
        title="Curious Mind",
        description="Research 5 different topics",
        category=AchievementCategory.EXPLORER,
        icon_name="compass",
        points=30,
        required_count=5,
    ),
    Achievement(
        id="weekend_researcher",
        title="Weekend Scientist",
        description="Conduct research on a weekend",
        category=AchievementCategory.EXPLORER,
        icon_name="calendar",
        points=20,
    ),
    Achievement(
        id="midnight_inspiration",
        title="Midnight Inspiration",
        description="Start research between midnight and 5 AM",
        category=AchievementCategory.EXPLORER,
        icon_name="moon",
        points=25,
        is_hidden=True,
    ),
    
    # Analyst achievements
    Achievement(
        id="hypothesis_refiner",
        title="Hypothesis Refiner",
        description="Refine a hypothesis 3 times",
        category=AchievementCategory.ANALYST,
        icon_name="edit-3",
        points=35,
        required_count=3,
    ),
    Achievement(
        id="evidence_collector",
        title="Evidence Collector",
        description="Collect evidence for 10 hypotheses",
        category=AchievementCategory.ANALYST,
        icon_name="search",
        points=40,
        required_count=10,
    ),
    Achievement(
        id="mythbuster",
        title="Mythbuster",
        description="Reject 5 invalid hypotheses",
        category=AchievementCategory.ANALYST,
        icon_name="x-circle",
        points=30,
        required_count=5,
    ),
    
    # Inventor achievements
    Achievement(
        id="first_invention",
        title="Inventor's Spark",
        description="Generate your first novel invention concept",
        category=AchievementCategory.INVENTOR,
        icon_name="zap",
        points=50,
    ),
    Achievement(
        id="prolific_inventor",
        title="Prolific Inventor",
        description="Generate 10 invention concepts",
        category=AchievementCategory.INVENTOR,
        icon_name="bulb",
        points=100,
        required_count=10,
    ),
    Achievement(
        id="cross_domain_innovator",
        title="Cross-Domain Innovator",
        description="Create invention concepts spanning 3 different domains",
        category=AchievementCategory.INVENTOR,
        icon_name="link",
        points=75,
        required_count=3,
    ),
    
    # Expert achievements
    Achievement(
        id="research_streak",
        title="Dedicated Researcher",
        description="Conduct research for 7 consecutive days",
        category=AchievementCategory.EXPERT,
        icon_name="trending-up",
        points=100,
        required_count=7,
        prerequisites=["first_research"],
    ),
    Achievement(
        id="citation_master",
        title="Citation Master",
        description="Include 50 citations across all papers",
        category=AchievementCategory.EXPERT,
        icon_name="book",
        points=75,
        required_count=50,
        prerequisites=["first_paper"],
    ),
    Achievement(
        id="breakthrough",
        title="Scientific Breakthrough",
        description="Have a hypothesis validated with overwhelming evidence",
        category=AchievementCategory.EXPERT,
        icon_name="award",
        points=150,
    ),
    
    # Master achievements
    Achievement(
        id="research_level_10",
        title="Senior Scientist",
        description="Reach research level 10",
        category=AchievementCategory.MASTER,
        icon_name="star",
        points=200,
        prerequisites=["research_streak"],
    ),
    Achievement(
        id="paper_collection",
        title="Research Library",
        description="Generate 25 research papers",
        category=AchievementCategory.MASTER,
        icon_name="book-open",
        points=250,
        required_count=25,
        prerequisites=["first_paper"],
    ),
    Achievement(
        id="invention_portfolio",
        title="Invention Portfolio",
        description="Create an invention portfolio with 20 validated ideas",
        category=AchievementCategory.MASTER,
        icon_name="briefcase",
        points=300,
        required_count=20,
        prerequisites=["first_invention"],
    ),
    
    # Legendary achievements
    Achievement(
        id="research_level_25",
        title="Distinguished Researcher",
        description="Reach research level 25",
        category=AchievementCategory.LEGENDARY,
        icon_name="award",
        points=500,
        prerequisites=["research_level_10"],
    ),
    Achievement(
        id="polymathic_genius",
        title="Polymathic Genius",
        description="Generate validated hypotheses across 10 different scientific domains",
        category=AchievementCategory.LEGENDARY,
        icon_name="globe",
        points=1000,
        required_count=10,
        prerequisites=["breakthrough"],
    ),
    Achievement(
        id="year_of_discovery",
        title="Year of Discovery",
        description="Maintain research activity for an entire year",
        category=AchievementCategory.LEGENDARY,
        icon_name="calendar",
        points=1000,
        required_count=365,
        prerequisites=["research_streak"],
    ),
]


class AchievementManager:
    """Manager for handling achievement tracking and awarding"""
    
    def __init__(self):
        """Initialize the achievement manager with default achievements"""
        self.achievements: Dict[str, Achievement] = {a.id: a for a in DEFAULT_ACHIEVEMENTS}
        self.profiles: Dict[str, ResearchProfile] = {}
        self.user_achievements: Dict[str, Dict[str, UserAchievement]] = {}
    
    def get_profile(self, user_id: str) -> ResearchProfile:
        """Get or create a research profile for a user"""
        if user_id not in self.profiles:
            self.profiles[user_id] = ResearchProfile(user_id=user_id)
            self.user_achievements[user_id] = {}
        return self.profiles[user_id]
    
    def get_achievement(self, achievement_id: str) -> Optional[Achievement]:
        """Get an achievement by ID"""
        return self.achievements.get(achievement_id)
    
    def get_all_achievements(self) -> List[Achievement]:
        """Get all available achievements"""
        return list(self.achievements.values())
    
    def get_user_achievements(self, user_id: str) -> Dict[str, UserAchievement]:
        """Get all achievements for a user"""
        if user_id not in self.user_achievements:
            self.user_achievements[user_id] = {}
        return self.user_achievements[user_id]
    
    def get_visible_achievements(self, user_id: str) -> List[Tuple[Achievement, Optional[UserAchievement]]]:
        """Get all visible achievements for a user with their progress"""
        user_achievements = self.get_user_achievements(user_id)
        visible_achievements = []
        
        for achievement in self.achievements.values():
            # Skip hidden achievements that haven't been earned
            if achievement.is_hidden and achievement.id not in user_achievements:
                continue
                
            user_achievement = user_achievements.get(achievement.id)
            visible_achievements.append((achievement, user_achievement))
            
        return visible_achievements
    
    def track_action(self, user_id: str, action_type: str, action_data: Dict[str, Any] = None) -> List[Achievement]:
        """
        Track a user action and award any relevant achievements
        
        Args:
            user_id: User ID
            action_type: Type of action (e.g., "start_research", "generate_hypothesis")
            action_data: Additional data about the action
            
        Returns:
            List of newly awarded achievements
        """
        if action_data is None:
            action_data = {}
            
        profile = self.get_profile(user_id)
        profile.last_active = datetime.now()
        
        # Update streak if within 24 hours of last activity
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        
        if profile.last_active.date() > yesterday.date():
            # Already logged in today, nothing to update
            pass
        elif profile.last_active.date() == yesterday.date():
            # Consecutive login
            profile.streak_days += 1
        else:
            # Streak broken
            profile.streak_days = 1
        
        # Update specific stats based on action
        if action_type == "start_research":
            topic = action_data.get("topic", "")
            if topic and topic not in profile.favorite_topics:
                profile.favorite_topics.append(topic)
                
        elif action_type == "complete_research":
            profile.research_cycles_completed += 1
            
        elif action_type == "generate_hypothesis":
            profile.hypotheses_generated += 1
            
        elif action_type == "publish_paper":
            profile.papers_published += 1
            
        elif action_type == "create_invention":
            profile.inventions_created += 1
        
        # Check for achievements
        newly_awarded = self._check_achievements(user_id, action_type, action_data)
        
        return newly_awarded
    
    def award_achievement(self, user_id: str, achievement_id: str) -> Optional[Achievement]:
        """
        Manually award an achievement to a user
        
        Args:
            user_id: User ID
            achievement_id: Achievement ID
            
        Returns:
            The awarded Achievement or None if not found
        """
        achievement = self.get_achievement(achievement_id)
        if not achievement:
            return None
            
        profile = self.get_profile(user_id)
        user_achievements = self.get_user_achievements(user_id)
        
        if achievement_id not in user_achievements:
            user_achievement = UserAchievement(
                user_id=user_id,
                achievement_id=achievement_id,
                progress=achievement.required_count,
                completed=True
            )
            user_achievements[achievement_id] = user_achievement
            profile.add_points(achievement.points)
            return achievement
        
        return None
    
    def _check_achievements(self, user_id: str, action_type: str, action_data: Dict[str, Any]) -> List[Achievement]:
        """Check if any achievements should be awarded based on an action"""
        profile = self.get_profile(user_id)
        user_achievements = self.get_user_achievements(user_id)
        newly_awarded = []
        
        # Check each achievement
        for achievement in self.achievements.values():
            # Skip already completed achievements
            if achievement.id in user_achievements and user_achievements[achievement.id].completed:
                continue
                
            # Check prerequisites
            prerequisites_met = True
            for prereq_id in achievement.prerequisites:
                if prereq_id not in user_achievements or not user_achievements[prereq_id].completed:
                    prerequisites_met = False
                    break
                    
            if not prerequisites_met:
                continue
                
            # Get or create user achievement progress
            if achievement.id not in user_achievements:
                user_achievements[achievement.id] = UserAchievement(
                    user_id=user_id,
                    achievement_id=achievement.id
                )
            
            user_achievement = user_achievements[achievement.id]
            
            # Check specific achievement conditions
            progress_updated = False
            
            # Action-based achievements
            if action_type == "start_research" and achievement.id == "first_research":
                user_achievement.progress = achievement.required_count
                progress_updated = True
                
            elif action_type == "generate_hypothesis" and achievement.id == "first_hypothesis":
                user_achievement.progress = achievement.required_count
                progress_updated = True
                
            elif action_type == "publish_paper" and achievement.id == "first_paper":
                user_achievement.progress = achievement.required_count
                progress_updated = True
                
            elif action_type == "create_invention" and achievement.id == "first_invention":
                user_achievement.progress = achievement.required_count
                progress_updated = True
                
            # Count-based achievements
            elif action_type == "start_research" and achievement.id == "topic_variety":
                topic = action_data.get("topic", "")
                if topic and len(set(profile.favorite_topics)) >= user_achievement.progress:
                    user_achievement.progress = len(set(profile.favorite_topics))
                    progress_updated = True
                    
            elif action_type == "complete_research" and achievement.id == "topic_variety":
                if len(set(profile.favorite_topics)) > user_achievement.progress:
                    user_achievement.progress = len(set(profile.favorite_topics))
                    progress_updated = True
                    
            elif action_type == "generate_hypothesis" and achievement.id == "hypothesis_refiner":
                if action_data.get("is_refinement", False):
                    user_achievement.progress += 1
                    progress_updated = True
                    
            elif action_type == "generate_hypothesis" and achievement.id == "evidence_collector":
                if action_data.get("has_evidence", False):
                    user_achievement.progress += 1
                    progress_updated = True
                    
            elif action_type == "evaluate_hypothesis" and achievement.id == "mythbuster":
                if action_data.get("status") == "rejected":
                    user_achievement.progress += 1
                    progress_updated = True
                    
            elif action_type == "create_invention" and achievement.id == "prolific_inventor":
                user_achievement.progress += 1
                progress_updated = True
                
            elif action_type == "create_invention" and achievement.id == "cross_domain_innovator":
                domain = action_data.get("domain", "")
                current_domains = action_data.get("domains", [])
                if domain and domain not in current_domains:
                    user_achievement.progress += 1
                    progress_updated = True
                    
            # Special time-based achievements
            elif achievement.id == "weekend_researcher":
                # Check if today is weekend
                if datetime.now().weekday() >= 5:  # 5=Saturday, 6=Sunday
                    user_achievement.progress = achievement.required_count
                    progress_updated = True
                    
            elif achievement.id == "midnight_inspiration":
                # Check if current hour is between midnight and 5 AM
                current_hour = datetime.now().hour
                if 0 <= current_hour < 5:
                    user_achievement.progress = achievement.required_count
                    progress_updated = True
                    
            # Level-based achievements
            elif achievement.id == "research_level_10" and profile.research_level >= 10:
                user_achievement.progress = achievement.required_count
                progress_updated = True
                
            elif achievement.id == "research_level_25" and profile.research_level >= 25:
                user_achievement.progress = achievement.required_count
                progress_updated = True
                
            # Streak-based achievements
            elif achievement.id == "research_streak" and profile.streak_days >= user_achievement.progress:
                user_achievement.progress = profile.streak_days
                progress_updated = True
                
            elif achievement.id == "year_of_discovery" and profile.streak_days >= user_achievement.progress:
                user_achievement.progress = profile.streak_days
                progress_updated = True
                
            # Collection-based achievements
            elif achievement.id == "paper_collection" and profile.papers_published >= user_achievement.progress:
                user_achievement.progress = profile.papers_published
                progress_updated = True
                
            elif achievement.id == "invention_portfolio" and profile.inventions_created >= user_achievement.progress:
                user_achievement.progress = profile.inventions_created
                progress_updated = True
                
            # Check if achievement is now completed
            if (progress_updated and 
                not user_achievement.completed and 
                user_achievement.progress >= achievement.required_count):
                
                user_achievement.completed = True
                user_achievement.earned_at = datetime.now()
                profile.add_points(achievement.points)
                newly_awarded.append(achievement)
                
                logger.info(f"User {user_id} earned achievement: {achievement.title}")
        
        return newly_awarded


# Singleton instance
achievement_manager = AchievementManager()