import os
import subprocess
import sys

def run_command(command):
    """Run a shell command and capture output"""
    print(f"Running: {command}")
    try:
        process = subprocess.run(
            command, shell=True, check=True, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True
        )
        print(process.stdout)
        return True, process.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Error details: {e.stderr}")
        return False, e.stderr

def fix_git_state():
    """Fix the git state by aborting rebase and checking out main branch"""
    
    # First, try to abort any rebase in progress
    run_command("git rebase --abort")
    
    # Check current branch status
    success, output = run_command("git branch")
    print(f"Current branch status: {output}")
    
    # Checkout the main branch
    run_command("git checkout main")
    
    # Check status again
    success, output = run_command("git branch")
    print(f"Updated branch status: {output}")
    
    # Pull latest changes with allow-unrelated-histories
    github_token = os.environ.get("GITHUB_TOKEN")
    github_user = "harishsg993010"
    github_repo = "Co-AI-Scientist"
    
    if github_token:
        remote_url = f"https://{github_user}:{github_token}@github.com/{github_user}/{github_repo}.git"
        run_command(f"git remote set-url origin {remote_url}")
        run_command("git pull origin main --allow-unrelated-histories")
    else:
        print("GITHUB_TOKEN not found in environment variables")
    
    # Add all files
    run_command("git add -A")
    
    # Commit changes
    run_command('git commit -m "Fix repository state and sync with remote"')
    
    # Try pushing
    if github_token:
        success, output = run_command("git push origin main")
        if success:
            print("Successfully pushed changes to GitHub")
        else:
            print("Failed to push changes to GitHub")
    
    return "Git repository state has been fixed."

if __name__ == "__main__":
    result = fix_git_state()
    print(result)