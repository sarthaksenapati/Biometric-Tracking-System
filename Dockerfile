FROM python:3.10-slim

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libcurl4-openssl-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
# Install torch CPU version (for GPU, see comment below)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Install insightface explicitly (not in requirements.txt)
RUN pip install --no-cache-dir insightface

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p embeddings_db unknown_persons_emb datasets

# Expose ports
# 8000 for FastAPI backend
# 8501 for Streamlit dashboard
EXPOSE 8000 8501

# Set default command (can be overridden)
CMD ["bash"]
