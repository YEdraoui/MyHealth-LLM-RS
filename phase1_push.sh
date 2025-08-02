# Create proper .gitignore for the project
cat > .gitignore << 'EOF'
# Dataset files (private/large)
data/
*.csv
*.jpg
*.jpeg
*.png
*.tiff

# Model weights (large files)
*.pth
*.pt
*.bin
*.safetensors

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so

# Virtual environments
ophthalmology_env/
brset_env/
venv/
env/

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Temporary files
tmp/
temp/
*.tmp

# Jupyter checkpoints
.ipynb_checkpoints/

# Environment variables
.env
.env.local

# Large files
*.zip
*.tar.gz
*.rar
EOF

# Remove cached files and reset git
rm -rf .git
git init

# Add only code files (no data, no models, no cache)
git add .gitignore
git add *.py
git add *.md
git add src/*.py
git add vision_model/*.py
git add llm/*.py
git add notebooks/*.py
git add requirements.txt

# Clean commit message
git commit -m "✅ PHASE 1: Ophthalmology LLM Architecture

🧠 Core Components:
- Vision models (ResNet50-based classifiers)
- Clinical LLM integration
- BRSET-compatible data loaders
- Multi-task training pipeline

📁 Structure:
- src/ - Dataset utilities and model components
- vision_model/ - Neural network architectures  
- llm/ - Clinical reasoning modules
- notebooks/ - Analysis scripts

🎯 Ready for Phase 2: Multimodal Fusion"

# Set up remote and push
git remote add origin https://github.com/YEdraoui/MyHealth-LLM-RS.git
git branch -M main

# Force push to overwrite (since we're excluding large files)
git push -f origin main

# Create tag for Phase 1
git tag -a v1.0-phase1 -m "Phase 1: Core Architecture Complete"
git push origin v1.0-phase1

echo "✅ PHASE 1 PUSHED TO GITHUB (clean, no private data)"
echo "🏷️ Tagged as v1.0-phase1"
echo "📁 Repository: https://github.com/YEdraoui/MyHealth-LLM-RS"
echo ""
echo "🧬 Ready for Phase 2 branch creation"
