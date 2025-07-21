#!/bin/bash
echo "🚀 Initialisation du dépôt Git pour Scale Generator App..."
git init
git remote add origin https://github.com/BobbyPattypan/scale-generator-app.git
git add .
git commit -m "Initial commit"
git branch -M main
git push -u origin main
