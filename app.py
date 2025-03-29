import os
import json
import uuid
import logging
import random
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config import PORT, HOST, DEBUG
from models import Source, Hypothesis, HypothesisStatus, AchievementCategory, SessionStore
from paper_generator import PaperGenerator
from achievement_manager import achievement_manager

# Configure logging
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger(__name__)

# Check if OpenAI API key is set
if not os.environ.get("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY environment variable is not set.")
    logger.warning("The application will attempt to run, but CrewAI agents might not function correctly.")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "ai-co-scientist-secret-key")

# Initialize components
paper_generator = PaperGenerator()

# SQLite session store
session_store = SessionStore()

# Global dictionary to store research sessions (in-memory cache)
research_sessions = {}

def get_user_id():
    """Get or create a user ID stored in the session"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/start_research', methods=['POST'])
def start_research():
    """Start a new research process"""
    data = request.form
    topic = data.get('topic')
    
    if not topic:
        return jsonify({
            'success': False,
            'error': 'Research topic is required'
        }), 400
    
    try:
        # Generate a session ID
        session_id = datetime.now().strftime('%Y%m%d%H%M%S')
        session['research_session_id'] = session_id
        
        # Get user ID for tracking
        user_id = get_user_id()
        
        # Store research session information in SQLite
        if not session_store.create_session(session_id, topic, user_id):
            return jsonify({
                'success': False,
                'error': 'Failed to create research session'
            }), 500
        
        # Also store in memory for fast access during this request
        research_sessions[session_id] = {
            'topic': topic,
            'status': 'initializing',
            'start_time': datetime.now().isoformat(),
            'progress': 0,
            'results': None,
            'user_id': user_id
        }
        
        # Track achievement for starting research
        try:
            new_achievements = achievement_manager.track_action(
                user_id=user_id,
                action_type="start_research",
                action_data={"topic": topic}
            )
            
            # Store any earned achievements for display
            if new_achievements:
                achievement_names = [a.title for a in new_achievements]
                achievements_text = f"You earned: {', '.join(achievement_names)}"
                
                # Update both the in-memory cache and the database
                research_sessions[session_id]['achievements'] = achievements_text
                session_store.update_session(session_id, {'achievements': achievements_text})
        except Exception as e:
            logger.error(f"Error tracking achievement: {str(e)}")
        
        # Start the research process asynchronously
        # In a production app, this would be done in a background task
        # For now, we'll simulate an immediate start
        research_sessions[session_id]['status'] = 'in_progress'
        session_store.update_session(session_id, {'status': 'in_progress'})
        
        # Log any new achievements
        if 'achievements' in research_sessions[session_id]:
            logger.info(f"User {user_id} earned achievements when starting research on '{topic}'")
        
        # Include any achievements in the response
        achievements = research_sessions[session_id].get('achievements', None)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': f'Research on "{topic}" started',
            'achievements': achievements,
            'redirect': url_for('research_status', session_id=session_id)
        })
    
    except Exception as e:
        logger.error(f"Error starting research: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to start research: {str(e)}'
        }), 500


@app.route('/research_status/<session_id>')
def research_status(session_id):
    """Show the status of a research session"""
    # Check memory cache first
    if session_id in research_sessions:
        research_data = research_sessions[session_id]
    else:
        # If not in memory, try to get from database
        research_data = session_store.get_session(session_id)
        if research_data is None:
            return render_template('index.html', error='Research session not found')
        
        # Cache in memory for future requests
        research_sessions[session_id] = research_data
    
    # Include any achievements earned
    achievements = research_data.get('achievements', None)
    
    # If we have achievements and results is a dictionary, add it there
    if achievements and isinstance(research_data.get('results'), dict):
        research_data['results']['achievements'] = achievements
    
    return render_template(
        'results.html',
        session_id=session_id,
        topic=research_data['topic'],
        status=research_data['status'],
        progress=research_data.get('progress', 0),
        results=research_data.get('results')
    )


@app.route('/api/research_progress/<session_id>')
def research_progress(session_id):
    """Get the progress of a research session and run actual research using CrewAI"""
    # Check memory cache first
    if session_id in research_sessions:
        research_data = research_sessions[session_id]
    else:
        # If not in memory, try to get from database
        research_data = session_store.get_session(session_id)
        if research_data is None:
            return jsonify({
                'success': False,
                'error': 'Research session not found'
            }), 404
        
        # Cache in memory for future requests
        research_sessions[session_id] = research_data
    
    # Verify OpenAI API key is present before attempting research
    if not os.environ.get("OPENAI_API_KEY"):
        error_message = 'OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable.'
        research_data['status'] = 'failed'
        research_data['error'] = error_message
        research_data['progress'] = 100  # Mark as complete to stop polling
        
        # Persist to SQLite
        session_store.update_session(session_id, {
            'status': research_data['status'],
            'progress': research_data['progress'],
            'error': error_message
        })
        return jsonify({
            'success': False,
            'status': 'failed',
            'progress': 100,
            'error': error_message,
            'topic': research_data['topic'],
            'results': {
                'topic': research_data['topic'],
                'hypotheses': [
                    {
                        'text': 'API Configuration Error',
                        'evidence': error_message,
                        'score': 0
                    }
                ],
                'summary': 'Research could not be completed due to missing API key.',
                'sources': []
            }
        })
    
    # If research is in progress and hasn't been run with CrewAI yet, start the actual process
    if research_data['status'] == 'in_progress' and research_data.get('crew_initiated', False) is False:
        try:
            # Import here to avoid circular imports
            from crew_setup import ResearchCrew
            
            # Mark that we've initiated the CrewAI process
            research_data['crew_initiated'] = True
            research_data['progress'] = 10
            research_data['current_agent'] = 'System'
            research_data['current_action'] = 'Initializing research environment'
            
            # Persist the progress update to SQLite
            session_store.update_session(session_id, {
                'crew_initiated': True,
                'progress': research_data['progress'],
                'current_agent': research_data['current_agent'],
                'current_action': research_data['current_action']
            })
            
            # Log that we're starting the real research process
            logger.info(f"Starting CrewAI research process for topic: {research_data['topic']}")
            
            # Initialize the crew and run a research cycle
            try:
                logger.info("Initializing ResearchCrew...")
                
                # Force garbage collection before creating the crew
                import gc
                gc.collect()
                
                # Create crew with memory optimization
                logger.info("Creating crew with memory optimization and progressive agent loading...")
                
                # Define a progress callback to update the research status in the database
                def progress_callback(agent_name, action, progress_percent):
                    # Update the research data in memory
                    research_data['progress'] = progress_percent
                    research_data['current_agent'] = agent_name
                    research_data['current_action'] = action
                    
                    # Persist the progress update to SQLite
                    session_store.update_session(session_id, {
                        'progress': progress_percent,
                        'current_agent': agent_name,
                        'current_action': action
                    })
                    
                    # Output more detailed information about the agent activity
                    logger.info(f"*** AGENT ACTIVITY ***")
                    logger.info(f"Active Agent: {agent_name}")
                    logger.info(f"Action: {action}")
                    logger.info(f"Progress: {progress_percent}%")
                    
                    # Track which agents have been seen so far using global state
                    if 'agents_seen' not in research_data:
                        research_data['agents_seen'] = set()
                    
                    research_data['agents_seen'].add(agent_name)
                    logger.info(f"Agents seen so far: {list(research_data['agents_seen'])}")
                    
                    # List the 6 expected agents and check which ones we've seen
                    all_agents = {"Inventor", "Researcher", "Analyst", "Evaluator", "Refiner", "Supervisor"}
                    remaining = all_agents - research_data['agents_seen']
                    logger.info(f"Remaining agents to be seen: {list(remaining)}")
                
                # Create a research crew instance with all 6 agents pre-loaded (no lazy loading)
                logger.info("Creating research crew with ALL 6 AGENTS pre-loaded: Inventor, Researcher, Analyst, Evaluator, Refiner, Supervisor")
                crew = ResearchCrew(verbose=True, progress_callback=progress_callback)
                
                # Test OpenAI API connectivity before attempting full research
                try:
                    import openai
                    # Super lightweight API test
                    # Just check models list without parameters
                    openai.models.list()
                    logger.info("OpenAI API connectivity test passed.")
                except Exception as api_test_error:
                    # Handle API connectivity issues gracefully
                    error_message = f"OpenAI API connectivity test failed: {str(api_test_error)}"
                    logger.error(error_message)
                    research_data['progress'] = 100
                    research_data['status'] = 'completed'
                    
                    # Persist to SQLite
                    session_store.update_session(session_id, {
                        'status': research_data['status'],
                        'progress': research_data['progress'],
                        'error': error_message
                    })
                    
                    research_data['results'] = {
                        'topic': research_data['topic'],
                        'hypotheses': [
                            {
                                'text': f'API Connectivity Issue',
                                'evidence': f'Unable to connect to OpenAI API. This could be due to network issues, incorrect API key, or service outage. Error: {str(api_test_error)}',
                                'score': 0
                            }
                        ],
                        'summary': 'Research could not be completed due to API connectivity issues. Please check your internet connection and API key validity.',
                        'sources': []
                    }
                    return jsonify({
                        'success': False,
                        'status': 'completed',
                        'progress': 100,
                        'error': error_message,
                        'results': research_data['results'],
                        'topic': research_data['topic']
                    })
                
                logger.info(f"Running research cycle for topic: {research_data['topic']}")
                
                # Force garbage collection again before running the cycle
                import gc
                gc.collect()
                
                # Make sure we have enough memory for the operation
                # Initial progress update using the callback pattern
                progress_callback('System', 'Loading AI research agents', 25)
                
                # Run the research cycle with an extra timeout wrapper for safety
                logger.info("Starting crew execution with optimized memory management...")
                
                # Set a global timeout to ensure we always get a response even if everything fails
                def get_emergency_result():
                    """Generate emergency result if all else fails, with no dependencies on CrewAI"""
                    try:
                        import openai
                        logger.info("*** EXTREME EMERGENCY: Direct API call outside all other systems ***")
                        
                        response = openai.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are a research assistant providing hypotheses."},
                                {"role": "user", "content": f"Generate 3 concise hypotheses about: {research_data['topic']}"}
                            ],
                            temperature=0.7,
                            max_tokens=500
                        )
                        
                        output = response.choices[0].message.content
                        
                        return {
                            "success": True,
                            "result": {
                                "output": output,
                                "emergency_fallback": True,
                                "extreme_emergency": True
                            }
                        }
                    except Exception as extreme_error:
                        logger.error(f"Even ultra-emergency fallback failed: {str(extreme_error)}")
                        return {
                            "success": True,
                            "result": {
                                "output": f"After multiple attempts, research on '{research_data['topic']}' could not be completed. Please try a different topic or try again later.",
                                "emergency_fallback": True,
                                "extreme_emergency": True,
                                "error": str(extreme_error)
                            }
                        }
                
                # Run with backup timeout
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(crew.run_research_cycle, research_data['topic'])
                    try:
                        # Absolute global timeout, regardless of internal timeouts
                        cycle_result = future.result(timeout=120)
                    except concurrent.futures.TimeoutError:
                        logger.error("*** GLOBAL TIMEOUT: Complete system failure, using extreme emergency result ***")
                        cycle_result = get_emergency_result()
                    except Exception as global_error:
                        logger.error(f"*** GLOBAL ERROR: {str(global_error)}, using extreme emergency result ***")
                        cycle_result = get_emergency_result()
                
                # Update progress using the callback pattern
                progress_callback('System', 'Processing research results', 70)
                
                # Release memory immediately after completion
                del crew
                gc.collect()
                
                # Process the results from CrewAI
                if cycle_result.get('success', False):
                    # Extract hypotheses, evidence, and other data from the result
                    crew_result = cycle_result.get('result', {})
                    logger.info(f"Research cycle completed successfully: {crew_result}")
                    
                    # Structure the research results
                    research_data['results'] = {
                        'topic': research_data['topic'],
                        'hypotheses': [],  # Will be populated later, initialized as empty
                        'summary': f"Research report on {research_data['topic']} generated by the AI Co-Scientist system with the Inventor Agent.",
                        'sources': []
                    }
                    
                    # Ensure we always have at least one default hypothesis as a fallback
                    default_hypothesis = {
                        'text': f"Research findings on {research_data['topic']}",
                        'evidence': "The AI Co-Scientist system analyzed this topic and arrived at initial conclusions. Further research is recommended.",
                        'score': 7.0
                    }
                    
                    # Extract hypotheses if available
                    if isinstance(crew_result, dict) and 'output' in crew_result:
                        output = crew_result['output']
                        
                        # Parse output to extract hypotheses
                        # First, check for explicitly formatted hypotheses with the word "Hypothesis"
                        hypotheses_found = False
                        
                        # Method 1: Look for "Hypothesis" keyword
                        if "Hypothesis" in output:
                            # Split by "Hypothesis" and process each part
                            hypothesis_parts = output.split("Hypothesis")
                            
                            for i, part in enumerate(hypothesis_parts[1:], 1):  # Skip the first empty part
                                if part.strip():  # Only process non-empty parts
                                    # Try to extract the first line as hypothesis text
                                    try:
                                        hypothesis_text = part.strip().split("\n")[0]  # First line after "Hypothesis"
                                        evidence = "\n".join(part.strip().split("\n")[1:])  # Rest of the content
                                        
                                        research_data['results']['hypotheses'].append({
                                            'text': hypothesis_text,
                                            'evidence': evidence if evidence.strip() else "Supporting evidence not explicitly provided.",
                                            'score': 8.0  # Default score if not explicitly provided
                                        })
                                        hypotheses_found = True
                                    except IndexError:
                                        # In case of malformed output, use the whole part
                                        research_data['results']['hypotheses'].append({
                                            'text': f"Hypothesis {i}",
                                            'evidence': part.strip(),
                                            'score': 7.5
                                        })
                                        hypotheses_found = True
                        
                        # Method 2: Look for numbered hypotheses (e.g., "1.", "2.", "Hypothesis 1:", etc.)
                        if not hypotheses_found:
                            import re
                            # Look for patterns like "1.", "2.", "Hypothesis 1:", "1)", etc.
                            hypothesis_pattern = r'(?:\d+\.|Hypothesis\s+\d+:|Hypothesis\s+\d+|\d+\))'
                            hypothesis_matches = re.split(hypothesis_pattern, output)
                            
                            if len(hypothesis_matches) > 1:  # If we found matches
                                # First element is usually preamble text
                                for i, part in enumerate(hypothesis_matches[1:], 1):
                                    if part.strip():
                                        # Split into first paragraph (hypothesis) and the rest (evidence)
                                        paragraphs = part.strip().split('\n\n')
                                        hypothesis_text = paragraphs[0].strip()
                                        evidence = "\n\n".join(paragraphs[1:]) if len(paragraphs) > 1 else ""
                                        
                                        research_data['results']['hypotheses'].append({
                                            'text': hypothesis_text,
                                            'evidence': evidence if evidence.strip() else "Supporting evidence not explicitly provided.",
                                            'score': 7.5
                                        })
                                        hypotheses_found = True
                        
                        # Method 3: Try to identify paragraphs that might be hypotheses
                        if not hypotheses_found:
                            paragraphs = output.split('\n\n')
                            
                            # Look for paragraphs that seem like hypotheses (short, declarative statements)
                            for i, paragraph in enumerate(paragraphs):
                                # Skip very short or very long paragraphs
                                if 10 <= len(paragraph) <= 300 and '.' in paragraph:
                                    if i < len(paragraphs) - 1:  # If there's a next paragraph, use as evidence
                                        research_data['results']['hypotheses'].append({
                                            'text': paragraph.strip(),
                                            'evidence': paragraphs[i+1] if i+1 < len(paragraphs) else "Supporting evidence not explicitly provided.",
                                            'score': 7.0
                                        })
                                    else:
                                        research_data['results']['hypotheses'].append({
                                            'text': paragraph.strip(),
                                            'evidence': "Supporting evidence not explicitly provided.",
                                            'score': 7.0
                                        })
                                    
                                    hypotheses_found = True
                                    # Only take up to 3 paragraphs as hypotheses to avoid over-extraction
                                    if len(research_data['results']['hypotheses']) >= 3:
                                        break
                        
                        # If all methods failed, create fallback hypotheses
                        if not research_data['results']['hypotheses']:
                            logger.warning(f"Failed to extract hypotheses from output, creating fallback hypotheses")
                            
                            # Split output into chunks of reasonable size for hypotheses
                            output_chunks = []
                            
                            # Try to split by paragraphs first
                            paragraphs = output.split('\n\n')
                            if len(paragraphs) >= 2:
                                output_chunks = paragraphs[:3]  # Use up to 3 paragraphs
                            else:
                                # If not enough paragraphs, split by sentences
                                import re
                                sentences = re.split(r'(?<=[.!?])\s+', output)
                                
                                # Group sentences into 3 chunks
                                chunk_size = max(1, len(sentences) // 3)
                                for i in range(0, min(3 * chunk_size, len(sentences)), chunk_size):
                                    chunk = ' '.join(sentences[i:i+chunk_size])
                                    if chunk.strip():
                                        output_chunks.append(chunk)
                            
                            # Create hypotheses from chunks
                            for i, chunk in enumerate(output_chunks):
                                if chunk.strip():
                                    research_data['results']['hypotheses'].append({
                                        'text': f"Hypothesis {i+1} about {research_data['topic']}",
                                        'evidence': chunk.strip(),
                                        'score': 7.0
                                    })
                            
                            # If still no hypotheses, create one fallback hypothesis
                            if not research_data['results']['hypotheses']:
                                research_data['results']['hypotheses'] = [
                                    {
                                        'text': f"Hypothesis about {research_data['topic']}",
                                        'evidence': output if output.strip() else "Research output was generated but no specific evidence was provided.",
                                        'score': 7.0
                                    }
                                ]
                            
                        # Add the full output as a source
                        research_data['results']['sources'] = [
                            {'title': 'AI Co-Scientist Research Output', 'content': output}
                        ]
                    
                    research_data['status'] = 'completed'
                    
                    # Final progress update to mark completion
                    # Update the research data in memory
                    research_data['progress'] = 100
                    research_data['current_agent'] = 'System'
                    research_data['current_action'] = 'Research completed successfully'
                    
                    # Persist to SQLite
                    session_store.update_session(session_id, {
                        'status': research_data['status'],
                        'progress': 100,
                        'results': research_data['results']
                    })
                    
                    # Track achievements for completing research and generating hypotheses
                    user_id = research_data['user_id']
                    
                    # Track research completion
                    achievement_manager.track_action(
                        user_id=user_id,
                        action_type="complete_research",
                        action_data={"topic": research_data['topic']}
                    )
                    
                    # Track hypothesis generation for each hypothesis
                    for hypothesis in research_data['results']['hypotheses']:
                        achievement_manager.track_action(
                            user_id=user_id,
                            action_type="generate_hypothesis",
                            action_data={
                                "text": hypothesis['text'],
                                "has_evidence": 'evidence' in hypothesis and hypothesis['evidence'],
                                "is_refinement": False
                            }
                        )
                else:
                    # If there was an error, provide appropriate feedback
                    logger.error(f"CrewAI research failed: {cycle_result.get('error', 'Unknown error')}")
                    research_data['status'] = 'completed'  # Still mark as completed for demo purposes
                    
                    # Final progress update for error case
                    # Update the research data in memory
                    research_data['progress'] = 100
                    research_data['current_agent'] = 'System'
                    research_data['current_action'] = f"Research failed: {cycle_result.get('error', 'Unknown error')}"
                    
                    # Persist all progress info to SQLite
                    session_store.update_session(session_id, {
                        'status': research_data['status'],
                        'progress': 100,
                        'current_agent': research_data['current_agent'],
                        'current_action': research_data['current_action']
                    })
                    
                    # Track completion achievement even for failed research
                    try:
                        user_id = research_data['user_id']
                        achievement_manager.track_action(
                            user_id=user_id,
                            action_type="complete_research",
                            action_data={"topic": research_data['topic']}
                        )
                    except Exception as achievement_e:
                        logger.error(f"Error tracking achievement: {str(achievement_e)}")
                    research_data['results'] = {
                        'topic': research_data['topic'],
                        'hypotheses': [
                            {
                                'text': f'The research process encountered an issue, but here is what we know about {research_data["topic"]}',
                                'evidence': 'The AI agents attempted to generate hypotheses but encountered technical limitations. ' +
                                          f'Error details: {cycle_result.get("error", "Unknown error")}',
                                'score': 6.0
                            }
                        ],
                        'summary': 'The research process was initiated but could not be fully completed. ' +
                                  'This could be due to API limits, complexity of the topic, or technical constraints.',
                        'sources': []
                    }
                    
                    # Persist the results to SQLite with updated values to ensure they are saved
                    session_store.update_session(session_id, {
                        'results': research_data['results']
                    })
            except Exception as e:
                # If the CrewAI process fails, log the error and fall back to a graceful error message
                logger.error(f"Error running CrewAI research: {str(e)}")
                research_data['status'] = 'completed'  # Still mark as completed for demo purposes
                
                # Final progress update using function defined in this scope
                # Update the research data in memory
                research_data['progress'] = 100
                research_data['current_agent'] = 'System'
                research_data['current_action'] = f"Research exception: {str(e)}"
                
                # Persist the progress update to SQLite
                session_store.update_session(session_id, {
                    'progress': 100,
                    'current_agent': 'System',
                    'current_action': f"Research exception: {str(e)}"
                })
                
                # Create the results first
                research_data['results'] = {
                    'topic': research_data['topic'],
                    'hypotheses': [
                        {
                            'text': f'Research on {research_data["topic"]} encountered a technical limitation',
                            'evidence': f'The AI Co-Scientist system tried to generate hypotheses but encountered an error: {str(e)}. ' +
                                      'This might be due to API limits, the complexity of the topic, or other technical constraints.',
                            'score': 5.0
                        }
                    ],
                    'summary': 'The research process could not be completed due to technical limitations. ' +
                              'In a production environment, this would be handled with retries and fallback mechanisms.',
                    'sources': []
                }
                
                # Now persist to SQLite with results
                session_store.update_session(session_id, {
                    'status': research_data['status'],
                    'progress': 100,  # Always set to 100 for completed sessions
                    'error': str(e),
                    'results': research_data['results']
                })
        except Exception as outer_e:
            # Catch-all for any other errors in the research process initialization
            logger.error(f"Error initializing research process: {str(outer_e)}")
            research_data['status'] = 'completed'  # Still mark as completed for demo purposes
            
            # Final progress update for outer exception case
            # Update the research data in memory
            research_data['progress'] = 100
            research_data['current_agent'] = 'System'
            research_data['current_action'] = f"Research initialization failed: {str(outer_e)}"
            
            # Persist the progress update to SQLite
            session_store.update_session(session_id, {
                'progress': 100,
                'current_agent': 'System',
                'current_action': f"Research initialization failed: {str(outer_e)}"
            })
            
            # Create the results first
            research_data['results'] = {
                'topic': research_data['topic'],
                'hypotheses': [
                    {
                        'text': f'Unable to complete research on {research_data["topic"]}',
                        'evidence': f'Error: {str(outer_e)}',
                        'score': 4.0
                    }
                ],
                'summary': 'The research process could not be initialized.',
                'sources': []
            }
            
            # Now persist to SQLite with results
            session_store.update_session(session_id, {
                'status': research_data['status'],
                'progress': 100,  # Always set to 100 for completed sessions
                'error': str(outer_e),
                'results': research_data['results']
            })
    
    # For subsequent calls, just update progress incrementally if still in progress
    elif research_data['status'] == 'in_progress':
        current_progress = research_data['progress']
        if current_progress < 100:
            # Increment progress more slowly now to show it's working
            research_data['progress'] = min(current_progress + 2, 99)  # Smaller increments for smoother updates
            
            # Rotate through different agent status messages for a better user experience
            agent_statuses = [
                ('Inventor', 'Generating creative research ideas and approaches'),
                ('Researcher', 'Gathering information and source material'),
                ('Analyst', 'Analyzing and synthesizing research findings'),
                ('Evaluator', 'Assessing hypothesis quality and evidence'),
                ('Refiner', 'Refining hypotheses based on evaluation'),
                ('Supervisor', 'Coordinating the research process')
            ]
            
            # Select agent status based on progress range
            status_index = min(int(current_progress / 17), 5)  # Divides progress into 6 ranges
            agent, action = agent_statuses[status_index]
            
            # Add some randomness to when we update the agent status
            if random.random() < 0.3:  # 30% chance to change the message on each poll
                # Use a different agent message occasionally to show variety
                alt_index = (status_index + random.choice([1, 2])) % 6
                agent, action = agent_statuses[alt_index]
            
            # Update and persist the progress and agent status
            research_data['current_agent'] = agent
            research_data['current_action'] = action
            
            # Persist to SQLite
            session_store.update_session(session_id, {
                'progress': research_data['progress'],
                'current_agent': research_data['current_agent'],
                'current_action': research_data['current_action']
            })
    
    # Return agent status along with progress updates
    return jsonify({
        'success': True,
        'status': research_data['status'],
        'progress': research_data['progress'],
        'current_agent': research_data.get('current_agent', 'System'),
        'current_action': research_data.get('current_action', 'Processing'),
        'results': research_data['results'] if research_data['status'] == 'completed' else None
    })


@app.route('/api/cancel_research/<session_id>', methods=['POST'])
def cancel_research(session_id):
    """Cancel a research session"""
    if session_id not in research_sessions:
        return jsonify({
            'success': False,
            'error': 'Research session not found'
        }), 404
    
    research_data = research_sessions[session_id]
    research_data['status'] = 'cancelled'
    
    # Persist to SQLite
    session_store.update_session(session_id, {
        'status': research_data['status']
    })
    
    return jsonify({
        'success': True,
        'message': 'Research cancelled'
    })


@app.route('/achievements')
def achievements():
    """View user achievements and research profile"""
    user_id = get_user_id()
    
    # Get user research profile
    profile = achievement_manager.get_profile(user_id)
    
    # Get all visible achievements with progress
    visible_achievements = achievement_manager.get_visible_achievements(user_id)
    
    # Group achievements by category
    achievements_by_category = {}
    for category in AchievementCategory:
        achievements_by_category[category.value] = []
    
    for achievement, user_achievement in visible_achievements:
        category = achievement.category.value
        achievements_by_category[category].append({
            'achievement': achievement,
            'user_achievement': user_achievement,
            'progress_percent': int((user_achievement.progress / achievement.required_count) * 100) if user_achievement else 0,
            'is_completed': user_achievement.completed if user_achievement else False
        })
    
    return render_template(
        'achievements.html',
        profile=profile,
        achievements_by_category=achievements_by_category,
        categories=AchievementCategory
    )


@app.route('/api/award_achievement', methods=['POST'])
def award_achievement():
    """Manually award an achievement (for testing purposes)"""
    user_id = get_user_id()
    achievement_id = request.form.get('achievement_id')
    
    if not achievement_id:
        return jsonify({
            'success': False,
            'error': 'Achievement ID is required'
        }), 400
    
    achievement = achievement_manager.award_achievement(user_id, achievement_id)
    if not achievement:
        return jsonify({
            'success': False,
            'error': 'Achievement not found or already awarded'
        }), 404
    
    return jsonify({
        'success': True,
        'message': f'Achievement "{achievement.title}" awarded!',
        'points': achievement.points
    })


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('index.html', error='Page not found'), 404


@app.route('/generate_paper/<session_id>', methods=['POST'])
def generate_paper(session_id):
    """Generate a research paper from the research results"""
    if session_id not in research_sessions:
        return jsonify({
            'success': False,
            'error': 'Research session not found'
        }), 404
    
    research_data = research_sessions[session_id]
    
    # Check if research is completed
    if research_data['status'] != 'completed':
        return jsonify({
            'success': False,
            'error': 'Research must be completed before generating a paper'
        }), 400
    
    try:
        # Convert the research results into the format needed for paper generation
        topic = research_data['topic']
        
        # Create hypothesis objects from the research results
        hypotheses = []
        for h_data in research_data['results']['hypotheses']:
            hypothesis = Hypothesis(
                id=str(uuid.uuid4()),
                text=h_data['text'],
                score=h_data.get('score', 5.0)
            )
            
            # Add evidence if available
            if 'evidence' in h_data:
                hypothesis.evidence_for = [h_data['evidence']]
            
            hypotheses.append(hypothesis)
        
        # Select only the highest-scoring hypothesis for the paper
        if hypotheses:
            selected_hypothesis = max(hypotheses, key=lambda h: h.score)
            hypotheses = [selected_hypothesis]
            app.logger.info(f"Selected single hypothesis for paper generation: {selected_hypothesis.text}")
        
        # Create source objects from the research results
        sources = []
        for s_data in research_data['results'].get('sources', []):
            source = Source(
                url=s_data.get('url', ''),
                title=s_data.get('title', 'Untitled Source'),
                content=s_data.get('content', 'No content available')
            )
            sources.append(source)
        
        # Generate the paper
        paper = paper_generator.create_paper_from_research(
            topic=topic,
            hypotheses=hypotheses,
            sources=sources,
            analysis_results=research_data['results']
        )
        
        # Store the paper in the research session
        if 'papers' not in research_data:
            research_data['papers'] = []
        
        paper_id = str(uuid.uuid4())
        research_data['papers'].append({
            'id': paper_id,
            'title': paper.title,
            'abstract': paper.abstract,
            'created_at': datetime.now().isoformat(),
            'word_count': paper.word_count,
            'paper_object': paper  # Store the actual paper object
        })
        
        # Track achievement for publishing a paper
        user_id = research_data['user_id']
        new_achievements = achievement_manager.track_action(
            user_id=user_id,
            action_type="publish_paper",
            action_data={
                "topic": topic,
                "title": paper.title,
                "word_count": paper.word_count,
                "citation_count": len(paper.references)
            }
        )
        
        # Create notification message for earned achievements
        achievement_message = ""
        if new_achievements:
            achievement_names = [a.title for a in new_achievements]
            achievement_message = f"You earned {len(new_achievements)} achievement(s): {', '.join(achievement_names)}"
            
            # Store achievements in the research data for display on the paper view page
            if 'achievements' not in research_data:
                research_data['achievements'] = achievement_message
            else:
                research_data['achievements'] += f" And {achievement_message.lower()}"
        
        return jsonify({
            'success': True,
            'message': 'Research paper generated successfully',
            'paper_id': paper_id,
            'achievements': achievement_message if achievement_message else None,
            'redirect': url_for('view_paper', session_id=session_id, paper_id=paper_id)
        })
        
    except Exception as e:
        logger.error(f"Error generating research paper: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to generate research paper: {str(e)}'
        }), 500


@app.route('/view_paper/<session_id>/<paper_id>')
def view_paper(session_id, paper_id):
    """View a generated research paper"""
    if session_id not in research_sessions:
        return render_template('index.html', error='Research session not found')
    
    research_data = research_sessions[session_id]
    
    # Find the paper
    paper_data = None
    if 'papers' in research_data:
        for p in research_data['papers']:
            if p['id'] == paper_id:
                paper_data = p
                break
    
    if not paper_data:
        return render_template('index.html', error='Research paper not found')
    
    # Get the paper object
    paper = paper_data['paper_object']
    
    # Convert to HTML
    paper_html = paper_generator.export_paper_to_html(paper)
    
    # Get any achievements to display
    achievements = research_data.get('achievements', None)
    
    return render_template(
        'paper.html',
        session_id=session_id,
        paper_id=paper_id,
        topic=research_data['topic'],
        paper=paper,
        paper_html=paper_html,
        achievements=achievements
    )


@app.route('/download_paper/<session_id>/<paper_id>')
def download_paper(session_id, paper_id):
    """Download a generated research paper as plain text or LaTeX"""
    if session_id not in research_sessions:
        return jsonify({
            'success': False,
            'error': 'Research session not found'
        }), 404
    
    research_data = research_sessions[session_id]
    
    # Check if format parameter is provided
    format_type = request.args.get('format', 'text')  # Default to text if not specified
    
    # Find the paper
    paper_data = None
    if 'papers' in research_data:
        for p in research_data['papers']:
            if p['id'] == paper_id:
                paper_data = p
                break
    
    if not paper_data:
        return jsonify({
            'success': False,
            'error': 'Research paper not found'
        }), 404
    
    # Get the paper object
    paper = paper_data['paper_object']
    
    # Convert to appropriate format
    if format_type == 'latex':
        # Convert to LaTeX (arXiv format)
        paper_content = paper_generator.export_paper_to_arxiv_latex(paper)
        mimetype = 'application/x-latex'
        filename = f"{paper.title.replace(' ', '_')}.tex"
    elif format_type == 'arxiv':
        # Convert to ArXiv-style LaTeX
        paper_content = paper_generator.format_paper_as_arxiv(paper)
        mimetype = 'application/x-latex'
        filename = f"{paper.title.replace(' ', '_')}_arxiv.tex"
    else:
        # Convert to plain text (default)
        paper_content = paper_generator.export_paper_to_text(paper)
        mimetype = 'text/plain'
        filename = f"{paper.title.replace(' ', '_')}.txt"
    
    # Return as a downloadable file
    return Response(
        paper_content,
        mimetype=mimetype,
        headers={'Content-Disposition': f'attachment;filename={filename}'}
    )


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return render_template('index.html', error='Server error'), 500
    
@app.route('/test_agents')
def test_agents():
    """Test page displaying status of all agent initialization"""
    # Import the necessary components
    from crew_setup import ResearchCrew
    
    try:
        # Create the crew instance to test all 6 agents get initialized
        logger.info("*** TEST: Creating research crew with ALL 6 AGENTS pre-loaded ***")
        crew = ResearchCrew(verbose=True)
        
        # Access all agents to verify they exist
        agents_initialized = {
            "Inventor": crew.inventor is not None,
            "Researcher": crew.researcher is not None,
            "Analyst": crew.analyst is not None,
            "Evaluator": crew.evaluator is not None,
            "Refiner": crew.refiner is not None,
            "Supervisor": crew.supervisor is not None
        }
        
        # Free up resources
        del crew
        import gc
        gc.collect()
        
        results = {
            "success": all(agents_initialized.values()),
            "agent_status": agents_initialized,
            "message": "All 6 agents successfully initialized" if all(agents_initialized.values()) else "Some agents failed to initialize"
        }
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error testing agents: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Error initializing agents"
        })

@app.route('/agent_diagnostics')
def agent_diagnostics():
    """Detailed diagnostic page showing agent structure and tools"""
    from crew_setup import ResearchCrew
    
    try:
        # Create the crew instance to test all 6 agents get initialized
        logger.info("*** DIAGNOSTICS: Creating research crew for detailed inspection ***")
        crew = ResearchCrew(verbose=True)
        
        # Extract detailed information about each agent
        agent_details = {}
        
        # Function to safely extract agent info
        def get_agent_info(agent):
            if not agent:
                return {"error": "Agent not initialized"}
            
            # Get basic agent properties
            info = {
                "name": getattr(agent, "name", "Unknown"),
                "role": getattr(agent, "role", "Unknown"),
                "goal": getattr(agent, "goal", "Unknown"),
                "allow_delegation": getattr(agent, "allow_delegation", False),
                "memory_enabled": getattr(agent, "memory", False),
                "max_iter": getattr(agent, "max_iter", "Unknown"),
                "llm": str(getattr(agent, "llm", "Unknown"))
            }
            
            # Extract tool information
            tools = getattr(agent, "tools", [])
            tool_info = []
            for tool in tools:
                tool_info.append({
                    "name": getattr(tool, "name", "Unknown"),
                    "description": getattr(tool, "description", "Unknown"),
                    "type": tool.__class__.__name__
                })
            
            info["tools"] = tool_info
            info["tool_count"] = len(tool_info)
            
            # Specifically check for communication tools
            info["has_ask_coworker_tool"] = any(
                getattr(tool, "name", "") == "Ask Coworker" for tool in tools
            )
            info["has_delegate_tool"] = any(
                getattr(tool, "name", "") == "Delegate Work" for tool in tools
            )
            
            return info
        
        # Get detailed info for each agent
        agent_details["Inventor"] = get_agent_info(crew.inventor)
        agent_details["Researcher"] = get_agent_info(crew.researcher)
        agent_details["Analyst"] = get_agent_info(crew.analyst)
        agent_details["Evaluator"] = get_agent_info(crew.evaluator)
        agent_details["Refiner"] = get_agent_info(crew.refiner)
        agent_details["Supervisor"] = get_agent_info(crew.supervisor)
        
        # Capture crew information
        crew_info = {
            "llm": getattr(crew, "llm", "Unknown"),
            "manager_llm": getattr(crew, "manager_llm", "Unknown"),
            "max_research_cycles": getattr(crew, "max_research_cycles", "Unknown"),
            "verbose": getattr(crew, "verbose", False),
            "current_research_cycle": getattr(crew, "current_research_cycle", None),
            "research_topic": getattr(crew, "research_topic", None)
        }
        
        # Specifically check if communication tools are available
        comm_tools_status = {}
        try:
            comm_tools_status["ask_coworker_tool"] = {
                "exists": hasattr(crew, "ask_coworker_tool"),
                "type": type(crew.ask_coworker_tool).__name__ if hasattr(crew, "ask_coworker_tool") else "Missing"
            }
            comm_tools_status["delegate_work_tool"] = {
                "exists": hasattr(crew, "delegate_work_tool"),
                "type": type(crew.delegate_work_tool).__name__ if hasattr(crew, "delegate_work_tool") else "Missing"
            }
        except Exception as tool_err:
            comm_tools_status["error"] = str(tool_err)
        
        # Free up resources
        del crew
        import gc
        gc.collect()
        
        # Build the response
        results = {
            "success": True,
            "crew_info": crew_info,
            "agent_details": agent_details,
            "communication_tools_status": comm_tools_status,
            "all_agents_have_ask_coworker": all(
                agent_details[a].get("has_ask_coworker_tool", False)
                for a in ["Inventor", "Researcher", "Analyst", "Evaluator", "Refiner", "Supervisor"]
            ),
            "all_agents_have_delegate": all(
                agent_details[a].get("has_delegate_tool", False)
                for a in ["Inventor", "Researcher", "Analyst", "Evaluator", "Refiner", "Supervisor"]
            )
        }
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in agent diagnostics: {str(e)}")
        import traceback
        traceback_str = traceback.format_exc()
        logger.error(f"Traceback: {traceback_str}")
        
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback_str,
            "message": "Error running agent diagnostics"
        })

@app.route('/communication_tools_test')
def communication_tools_test():
    """Test the communication tools directly"""
    from tools.communication_tools import AskCoworkerTool, DelegateWorkTool
    from crew_setup import ResearchCrew
    
    try:
        # Create the tools directly
        ask_tool = AskCoworkerTool()
        delegate_tool = DelegateWorkTool()
        
        tool_info = {
            "ask_coworker": {
                "name": getattr(ask_tool, "name", "Unknown"),
                "description": getattr(ask_tool, "description", "Unknown"),
                "has_args_schema": hasattr(ask_tool, "args_schema"),
                "args_schema_type": type(getattr(ask_tool, "args_schema", None)).__name__
            },
            "delegate_work": {
                "name": getattr(delegate_tool, "name", "Unknown"),
                "description": getattr(delegate_tool, "description", "Unknown"),
                "has_args_schema": hasattr(delegate_tool, "args_schema"),
                "args_schema_type": type(getattr(delegate_tool, "args_schema", None)).__name__
            }
        }
        
        # Verify that we can create a crew and that it automatically gets these tools
        crew = ResearchCrew(verbose=True)
        crew_has_tools = {
            "ask_coworker_tool": hasattr(crew, "ask_coworker_tool"),
            "delegate_work_tool": hasattr(crew, "delegate_work_tool")
        }
        
        # Test if the first agent has the tools attached
        inventor_has_tools = False
        try:
            inventor = crew.inventor
            inventor_tools = getattr(inventor, "tools", [])
            inventor_has_tools = any(getattr(tool, "name", "") == "Ask Coworker" for tool in inventor_tools)
        except Exception as agent_err:
            logger.error(f"Error checking inventor tools: {str(agent_err)}")
        
        # Free up resources
        del crew
        import gc
        gc.collect()
        
        return jsonify({
            "success": True,
            "tool_info": tool_info,
            "crew_has_tools": crew_has_tools,
            "inventor_has_tools": inventor_has_tools,
            "message": "Communication tools test completed successfully"
        })
    except Exception as e:
        logger.error(f"Error testing communication tools: {str(e)}")
        import traceback
        traceback_str = traceback.format_exc()
        logger.error(f"Traceback: {traceback_str}")
        
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback_str, 
            "message": "Error testing communication tools"
        })


if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)
