# Instructions for Uploading to GitHub

Follow these steps to upload this replication repository to GitHub:

## 1. Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Choose a repository name (e.g., `llm-interpretation-replication`)
5. Add a description: "Replication code for 'Large Language Models as Unreliable Judges'"
6. Set visibility (Public or Private)
7. **DO NOT** initialize with README, .gitignore, or license (we already have these)
8. Click "Create repository"

## 2. Configure Git (First Time Only)

If you haven't configured git on your system:

```bash
git config --global user.email "your-email@example.com"
git config --global user.name "Your Name"
```

## 3. Navigate to the Repository

```bash
cd /home/thechoipolloi/claude_projects/llm_interpretation/llm_interpretation_replication
```

## 4. Initialize and Commit (if not already done)

```bash
# If not already initialized
git init
git branch -m main

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: LLM interpretation replication code"
```

## 5. Connect to GitHub and Push

Replace `YOUR-USERNAME` and `YOUR-REPO-NAME` with your actual GitHub username and repository name:

```bash
# Add remote origin
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git

# Push to GitHub
git push -u origin main
```

If using SSH instead of HTTPS:
```bash
git remote add origin git@github.com:YOUR-USERNAME/YOUR-REPO-NAME.git
git push -u origin main
```

## 6. Authentication

You may be prompted for authentication:

### For HTTPS:
- Username: Your GitHub username
- Password: Your GitHub Personal Access Token (not your password!)
  - To create a token: GitHub Settings → Developer settings → Personal access tokens → Generate new token
  - Give it `repo` permissions

### For SSH:
- Make sure you have SSH keys set up with GitHub
- See: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

## 7. Verify Upload

1. Go to your GitHub repository page
2. Verify all files are uploaded
3. Check that the README displays correctly

## 8. Optional: Add Repository Topics

On your GitHub repository page:
1. Click the gear icon next to "About"
2. Add topics like: `nlp`, `llm`, `replication`, `research`, `legal-tech`, `machine-learning`

## 9. Optional: Create a Release

1. Go to the "Releases" section
2. Click "Create a new release"
3. Tag version: `v1.0.0`
4. Release title: "Initial Release"
5. Describe the release and link to the paper

## Troubleshooting

### If you get a "repository not found" error:
- Check the repository URL is correct
- Ensure you have permissions to push

### If authentication fails:
- For HTTPS: Make sure you're using a Personal Access Token, not your password
- For SSH: Check your SSH key is added to GitHub

### To change remote URL if needed:
```bash
git remote set-url origin NEW-URL
```

### To see current remote:
```bash
git remote -v
```

## Repository Structure Summary

Your repository is now ready with:
- ✅ All analysis code organized
- ✅ README with replication instructions
- ✅ requirements.txt for dependencies
- ✅ .gitignore for clean repository
- ✅ MIT License
- ✅ Clear folder structure

The repository contains everything needed to replicate the paper's results!