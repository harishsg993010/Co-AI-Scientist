from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import json
import sqlite3
import os


class HypothesisStatus(Enum):
    GENERATED = "generated"
    EVALUATING = "evaluating"
    REFINED = "refined"
    VALIDATED = "validated"
    REJECTED = "rejected"
    

@dataclass
class Source:
    """Represents a source of information used in research"""
    url: str
    title: str
    content: str
    accessed_at: datetime = field(default_factory=datetime.now)
    type: str = "web"  # web, pdf, api, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    """Represents a research hypothesis"""
    id: str
    text: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: HypothesisStatus = HypothesisStatus.GENERATED
    score: float = 0.0
    elo_rating: int = 1500  # Starting ELO rating
    sources: List[Source] = field(default_factory=list)
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    refinement_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_refinement(self, new_text: str, reason: str):
        """Add a refinement to the hypothesis history"""
        old_text = self.text
        self.refinement_history.append({
            "previous_text": old_text,
            "new_text": new_text,
            "reason": reason,
            "timestamp": datetime.now(),
        })
        self.text = new_text
        self.updated_at = datetime.now()
        self.status = HypothesisStatus.REFINED


@dataclass
class ResearchCycle:
    """Represents a complete research cycle"""
    id: str
    topic: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    hypotheses: List[Hypothesis] = field(default_factory=list)
    cycle_number: int = 1
    notes: str = ""
    status: str = "in_progress"  # in_progress, completed, failed
    
    def complete(self, notes: str = ""):
        """Mark the research cycle as complete"""
        self.end_time = datetime.now()
        self.status = "completed"
        if notes:
            self.notes = notes


@dataclass
class ResearchReport:
    """Represents a final research report"""
    id: str
    title: str
    topic: str
    summary: str
    content: str
    hypotheses: List[Hypothesis]
    created_at: datetime = field(default_factory=datetime.now)
    sources: List[Source] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchPaper:
    """Represents an academic research paper"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    keywords: List[str]
    introduction: str
    literature_review: str
    methodology: str
    results: str
    discussion: str
    conclusion: str
    references: List[Dict[str, str]]
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    version: int = 1
    status: str = "draft"  # draft, review, final
    citation_style: str = "APA"  # APA, MLA, Chicago, etc.
    word_count: int = 0
    figures: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    
    def calculate_word_count(self) -> int:
        """Calculate and update the paper's word count"""
        sections = [
            self.abstract, self.introduction, self.literature_review,
            self.methodology, self.results, self.discussion, self.conclusion
        ]
        total_words = sum(len(section.split()) for section in sections)
        self.word_count = total_words
        return total_words
    
    def update_version(self):
        """Increment the paper version and update timestamp"""
        self.version += 1
        self.last_updated = datetime.now()
        
    def format_citation(self, reference: Dict[str, str]) -> str:
        """Format a reference according to the chosen citation style"""
        if self.citation_style == "APA":
            # Basic APA formatting
            if reference.get("type") == "article":
                return f"{reference.get('authors', 'Unknown')}. ({reference.get('year', 'n.d.')}). {reference.get('title', 'Untitled')}. {reference.get('journal', '')}, {reference.get('volume', '')}{reference.get('issue', '')}, {reference.get('pages', '')}."
            else:
                return f"{reference.get('authors', 'Unknown')}. ({reference.get('year', 'n.d.')}). {reference.get('title', 'Untitled')}. {reference.get('publisher', '')}"
        else:
            # Default simple format
            return f"{reference.get('authors', 'Unknown')}. {reference.get('title', 'Untitled')}. {reference.get('year', 'n.d.')}"


@dataclass
class AgentAction:
    """Represents an action taken by an agent"""
    agent_id: str
    agent_role: str
    action_type: str
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    status: str = "success"  # success, failed
    error_message: str = ""


class AchievementCategory(Enum):
    """Categories for research achievements"""
    NOVICE = "Novice"          # Basic initial achievements
    EXPLORER = "Explorer"      # Discovering new areas
    ANALYST = "Analyst"        # Analysis achievements
    INVENTOR = "Inventor"      # Related to invention and creativity
    EXPERT = "Expert"          # Advanced achievements
    MASTER = "Master"          # High-level achievements
    LEGENDARY = "Legendary"    # Extremely rare achievements


@dataclass
class Achievement:
    """Represents a research achievement that can be earned"""
    id: str
    title: str
    description: str
    category: AchievementCategory
    icon_name: str
    points: int = 10
    required_count: int = 1    # How many times the action must be performed
    is_hidden: bool = False    # Whether this achievement is visible before unlocking
    prerequisites: List[str] = field(default_factory=list)  # IDs of required achievements
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class UserAchievement:
    """Represents an achievement earned by a user"""
    user_id: str
    achievement_id: str
    earned_at: datetime = field(default_factory=datetime.now)
    progress: int = 0          # Current progress toward required_count
    completed: bool = False    # Whether the achievement is fully earned


@dataclass
class ResearchProfile:
    """Represents a user's research profile with achievements and stats"""
    user_id: str
    total_points: int = 0
    research_level: int = 1
    achievements: List[UserAchievement] = field(default_factory=list)
    research_cycles_completed: int = 0
    hypotheses_generated: int = 0
    papers_published: int = 0
    inventions_created: int = 0
    last_active: datetime = field(default_factory=datetime.now)
    registration_date: datetime = field(default_factory=datetime.now)
    streak_days: int = 0       # Consecutive days of research activity
    favorite_topics: List[str] = field(default_factory=list)
    
    def add_points(self, points: int) -> int:
        """Add points to the research profile and update level"""
        self.total_points += points
        
        # Simple level calculation - can be adjusted for game balance
        new_level = max(1, int((self.total_points / 100) + 1))
        level_changed = new_level > self.research_level
        self.research_level = new_level
        
        return self.research_level


class SessionStore:
    """SQLite-based persistent storage for research sessions"""
    
    DB_FILE = "research_sessions.db"
    
    def __init__(self):
        """Initialize the session store and create tables if they don't exist"""
        self._init_db()
        print("Running database migrations...")
        self.run_migrations()
    
    def _init_db(self):
        """Initialize the database and create tables if needed"""
        conn = sqlite3.connect(self.DB_FILE)
        cursor = conn.cursor()
        
        # Create research_sessions table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS research_sessions (
            session_id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            status TEXT NOT NULL,
            start_time TEXT NOT NULL,
            progress REAL DEFAULT 0.0,
            results TEXT,
            user_id TEXT,
            achievements TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            current_agent TEXT,
            current_action TEXT,
            crew_initiated BOOLEAN DEFAULT 0,
            error TEXT
        )
        ''')
        
        # Check for and add missing columns if needed
        self._add_column_if_not_exists(cursor, "research_sessions", "current_agent", "TEXT")
        self._add_column_if_not_exists(cursor, "research_sessions", "current_action", "TEXT")
        self._add_column_if_not_exists(cursor, "research_sessions", "crew_initiated", "BOOLEAN DEFAULT 0")
        self._add_column_if_not_exists(cursor, "research_sessions", "error", "TEXT")
        
        conn.commit()
        conn.close()
        
    def _add_column_if_not_exists(self, cursor, table_name, column_name, column_type):
        """Add a column to a table if it doesn't exist already"""
        try:
            # Check if the column exists
            cursor.execute(f"SELECT {column_name} FROM {table_name} LIMIT 1")
        except sqlite3.OperationalError:
            # Column doesn't exist, so add it
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
            print(f"Added column {column_name} to {table_name}")
        except Exception as e:
            print(f"Error checking/adding column {column_name}: {e}")
            
    def run_migrations(self):
        """
        Run all database migrations
        This will explicitly add any columns that are missing from existing tables
        """
        try:
            print("Starting database migrations...")
            conn = sqlite3.connect(self.DB_FILE)
            cursor = conn.cursor()
            
            # Get existing column names from the research_sessions table
            cursor.execute("PRAGMA table_info(research_sessions)")
            existing_columns = [row[1] for row in cursor.fetchall()]
            print(f"Existing columns: {existing_columns}")
            
            # List of columns that should exist
            required_columns = {
                "current_agent": "TEXT",
                "current_action": "TEXT",
                "crew_initiated": "BOOLEAN DEFAULT 0",
                "error": "TEXT"
            }
            
            # Add any missing columns
            for col_name, col_type in required_columns.items():
                if col_name not in existing_columns:
                    try:
                        cursor.execute(f"ALTER TABLE research_sessions ADD COLUMN {col_name} {col_type}")
                        conn.commit()
                        print(f"Migration added column {col_name} to research_sessions")
                    except Exception as e:
                        print(f"Error adding column {col_name}: {e}")
            
            conn.close()
            print("Database migrations completed")
        except Exception as e:
            print(f"Error running migrations: {e}")
    
    def create_session(self, session_id: str, topic: str, user_id: str) -> bool:
        """
        Create a new research session
        
        Args:
            session_id: Session ID
            topic: Research topic
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.DB_FILE)
            cursor = conn.cursor()
            
            cursor.execute(
                """INSERT INTO research_sessions 
                   (session_id, topic, status, start_time, user_id, current_agent, current_action) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, topic, "initializing", datetime.now().isoformat(), user_id, 
                 "System", "Initializing research environment")
            )
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error creating session: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a research session by ID
        
        Args:
            session_id: Session ID
            
        Returns:
            Dictionary representation of the session or None if not found
        """
        try:
            conn = sqlite3.connect(self.DB_FILE)
            conn.row_factory = sqlite3.Row  # This enables column access by name
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM research_sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            # Convert row to dictionary
            session_data = dict(row)
            
            # Parse JSON fields if they exist
            if session_data.get('results'):
                try:
                    session_data['results'] = json.loads(session_data['results'])
                except:
                    pass  # Keep as string if JSON parsing fails
            
            conn.close()
            return session_data
        except Exception as e:
            print(f"Error getting session: {e}")
            return None
    
    def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """
        Update a research session
        
        Args:
            session_id: Session ID
            data: Dictionary of data to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.DB_FILE)
            cursor = conn.cursor()
            
            # Build the update query dynamically based on the provided data
            update_fields = []
            values = []
            
            # Debug information
            print(f"Update request for session {session_id} with data: {data.keys()}")
            
            for key, value in data.items():
                # Skip session_id as it's used in the WHERE clause
                if key == 'session_id':
                    continue
                
                # Convert complex objects to JSON
                if key == 'results' and isinstance(value, dict):
                    value = json.dumps(value)
                
                update_fields.append(f"{key} = ?")
                values.append(value)
            
            # Add session_id for the WHERE clause
            values.append(session_id)
            
            if not update_fields:
                print(f"Warning: No fields to update for session {session_id}")
                return True
            
            query = f"UPDATE research_sessions SET {', '.join(update_fields)} WHERE session_id = ?"
            print(f"Executing query: {query}")
            print(f"With values: {values}")
            
            cursor.execute(query, values)
            
            conn.commit()
            conn.close()
            return cursor.rowcount > 0
        except sqlite3.OperationalError as e:
            print(f"SQLite operational error updating session: {e}")
            print(f"Session ID: {session_id}, Data keys: {list(data.keys())}")
            # Attempt to identify which column is causing the issue
            if "no such column" in str(e):
                column_name = str(e).split("no such column: ")[1].strip()
                print(f"Missing column detected: {column_name}")
                try:
                    # Try to add the missing column dynamically
                    conn = sqlite3.connect(self.DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute(f"ALTER TABLE research_sessions ADD COLUMN {column_name} TEXT")
                    print(f"Added missing column {column_name} to database")
                    conn.commit()
                    conn.close()
                    
                    # Retry the update now that we've added the column
                    return self.update_session(session_id, data)
                except Exception as add_error:
                    print(f"Failed to add missing column {column_name}: {add_error}")
            return False
        except Exception as e:
            print(f"Error updating session: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a research session
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.DB_FILE)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM research_sessions WHERE session_id = ?", (session_id,))
            
            conn.commit()
            conn.close()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False
    
    def get_all_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all research sessions, optionally filtered by user ID
        
        Args:
            user_id: Optional user ID filter
            
        Returns:
            List of session dictionaries
        """
        try:
            conn = sqlite3.connect(self.DB_FILE)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute("SELECT * FROM research_sessions WHERE user_id = ? ORDER BY start_time DESC", (user_id,))
            else:
                cursor.execute("SELECT * FROM research_sessions ORDER BY start_time DESC")
            
            rows = cursor.fetchall()
            
            # Convert rows to dictionaries
            sessions = []
            for row in rows:
                session_data = dict(row)
                
                # Parse JSON fields if they exist
                if session_data.get('results'):
                    try:
                        session_data['results'] = json.loads(session_data['results'])
                    except:
                        pass  # Keep as string if JSON parsing fails
                
                sessions.append(session_data)
            
            conn.close()
            return sessions
        except Exception as e:
            print(f"Error getting all sessions: {e}")
            return []
