#!/bin/bash
# Script to cleanly push code to GitHub using the GITHUB_TOKEN secret

echo "Pushing code to GitHub using clean approach..."

# Repository information
GITHUB_USER="harishsg993010"
GITHUB_REPO="Co-AI-Scientist"

# Configure Git
echo "Configuring Git..."
git config --global user.name "harishsg993010"
git config --global user.email "harishsg993010@gmail.com"

# Setup the remote URL with authentication using the environment secret
echo "Setting up remote with authentication..."
if [ -z "$GITHUB_TOKEN" ]; then
  echo "Error: GITHUB_TOKEN environment variable is not set."
  echo "Please set the GITHUB_TOKEN environment variable."
  exit 1
fi

REMOTE_URL="https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${GITHUB_REPO}.git"

# Check if the remote exists
if git remote | grep -q "origin"; then
  echo "Remote 'origin' exists, removing it..."
  git remote remove origin
fi

echo "Adding new remote 'origin'..."
git remote add origin ${REMOTE_URL}

# Force-create a new branch and checkout
echo "Creating a new clean branch..."
git checkout --orphan temp_branch

# Add all files
echo "Adding all files to git..."
git add -A

# Commit changes
echo "Committing changes..."
TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
git commit -m "Initial Co-AI-Scientist commit ${TIMESTAMP}" || {
  echo "No changes to commit."
  exit 0
}

# Force push to GitHub as new main branch
echo "Force pushing to GitHub repository..."
if git push -f origin temp_branch:main; then
  echo "Success! All code has been pushed to GitHub."
  echo "Repository URL: https://github.com/${GITHUB_USER}/${GITHUB_REPO}"
  
  # Clean up - delete temporary branch
  git branch -D temp_branch
  git checkout -b main
  git branch --set-upstream-to=origin/main main
  
  echo "Repository is now in a clean state."
else
  echo "Error: Failed to push code to GitHub."
  echo "Please check your token and repository configuration."
fi