# Dockerfile for ASL Recognition API
# Uses Python 3.10 for compatibility with MediaPipe and dependencies

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for MediaPipe and OpenCV
# Note: libgl1-mesa-glx is replaced with libgl1 in Debian 12+
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY ASL_to_English/requirements_signtalk.txt .
COPY ASL_to_English/requirements.txt .

# Install Python dependencies
# First install API dependencies (FastAPI, uvicorn, etc.)
RUN pip install --no-cache-dir \
    fastapi>=0.100.0 \
    uvicorn>=0.23.0 \
    python-multipart>=0.0.6

# Then install ML dependencies from signtalk requirements
RUN pip install --no-cache-dir -r requirements_signtalk.txt

# Copy the entire project structure
# This preserves the ASL_to_English package structure needed for imports
COPY . .

# Expose port 8000
EXPOSE 8000

# Health check (using curl - install it if needed, or use Python's urllib)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the server
CMD ["python", "server.py"]

