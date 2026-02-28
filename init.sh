#!/bin/bash
# init.sh - NexusAI Phase 1 Environment Setup

echo "🚀 NexusAI Harness Initialization..."

# Check Python and Conda
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found."
    exit 1
fi
echo "✅ Python present: $(python3 --version)"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found."
    exit 1
fi
echo "✅ Node.js present: $(node -v)"

# Create Directory Structure
mkdir -p backend/app/routers backend/app/services backend/app/models
mkdir -p frontend/src/app frontend/src/components
echo "✅ Directory structure verified."

# Check Qdrant
if ! curl -s http://localhost:6333/ &> /dev/null; then
  echo "⚠️  Qdrant is not responsive at http://localhost:6333. Please ensure Docker container is running."
else
  echo "✅ Qdrant detected."
fi

# Install Backend Dependencies
echo "Installing backend dependencies in daily_3_9..."
pip install -r backend/requirements.txt

echo "Environment check complete. Run 'python backend/app/harness_check.py' for API validation."
