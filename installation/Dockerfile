# SIRA Dockerfile
# Built on official Python 3.12 image with Debian base

FROM python:3.12-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Set working directory
WORKDIR /app

# Install system dependencies including Graphviz
# (required by networkx and pygraphviz for system diagrams)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    graphviz \
    graphviz-dev \
    libgraphviz-dev \
    libxml2-dev \
    git \
    curl \
    wget \
    gcc \
    g++ \
    libfreetype6-dev \
    libpng-dev \
    libcairo2-dev \
    libjpeg62-turbo-dev \
    libpango1.0-dev \
    libgif-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements files
COPY installation/constraints.txt ./installation/
COPY installation/requirements-core.txt ./installation/
COPY installation/requirements-dev.txt ./installation/
COPY installation/requirements-viz.txt ./installation/
COPY installation/requirements-docs.txt ./installation/
COPY installation/requirements-diagrams.txt ./installation/

# Install Python dependencies
RUN pip install --no-cache-dir -r ./installation/constraints.txt
RUN pip install --no-cache-dir -r ./installation/requirements-core.txt
RUN pip install --no-cache-dir -r ./installation/requirements-dev.txt
RUN pip install --no-cache-dir -r ./installation/requirements-viz.txt
RUN pip install --no-cache-dir -r ./installation/requirements-docs.txt
RUN pip install --no-cache-dir -r ./installation/requirements-diagrams.txt

# Copy SIRA source code
COPY . .

# Set Python path to include SIRA module
ENV PYTHONPATH=${PYTHONPATH:+${PYTHONPATH}:}/app

# Default command - show SIRA help
CMD ["python", "-m", "sira", "--help"]

# Note: To run a simulation, mount your scenario directory and override CMD:
# docker run -v /path/to/scenario:/scenario sira python -m sira -d /scenario -s
