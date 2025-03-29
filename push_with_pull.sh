#!/bin/bash
# Script to pull latest changes and then push code to GitHub using the GITHUB_TOKEN secret

echo "Pushing code to GitHub using stored secret token (with pull first)..."

# Repository information
GITHUB_USER="harishsg993010"
GITHUB_REPO="Co-AI-Scientist"

# Configure Git
echo "Configuring Git..."
git config --global user.name "harishsg993010"
git config --global user.email "harishsg993010@gmail.com"

# Setup the remote URL with authentication using the environment secret
echo "Setting up remote with authentication..."
REMOTE_URL="https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${GITHUB_REPO}.git"
git remote set-url origin ${REMOTE_URL}

# First pull the latest changes
echo "Pulling latest changes from repository..."
git pull origin main --allow-unrelated-histories

# Add all files
echo "Adding all files to git..."
git add -A

# Commit changes
echo "Committing changes..."
TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
git commit -m "Update Co-AI-Scientist ${TIMESTAMP}" || {
    echo "No changes to commit."
    exit 0
}

# Push to GitHub
echo "Pushing to GitHub repository..."
if git push origin main; then
    echo "Success! All code has been pushed to GitHub."
    echo "Repository URL: https://github.com/${GITHUB_USER}/${GITHUB_REPO}"
else
    echo "Error: Failed to push code to GitHub."
    echo "Please check your token and repository configuration."
fi