# ============================================================================
# 12. DOCKERFILE
# ============================================================================

dockerfile_content = '''
# ============================================================================
# TRACK&CARE INFERENCE API - DOCKERFILE
# ============================================================================

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api.py .
COPY models/ ./models/

# Create models directory if not exists
RUN mkdir -p models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
'''

with open('Dockerfile', 'w', encoding='utf-8') as f:
    f.write(dockerfile_content)

print("✅ Dockerfile criado")

# Requirements.txt
requirements_content = '''
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.2
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2
xgboost==2.0.0
joblib==1.3.2
python-multipart==0.0.6
'''

with open('requirements.txt', 'w', encoding='utf-8') as f:
    f.write(requirements_content)

print("✅ requirements.txt criado")
