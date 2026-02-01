# Base Image: Ray 2.33.0 (GPU Version)
FROM rayproject/ray:2.33.0-py310-gpu

# Set working directory
WORKDIR /app

# Switch to root to install system dependencies
USER root
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Switch back to ray user (best practice)
USER ray

# Copy dependency files first to leverage Docker cache
COPY requirements-core.txt .
COPY requirements-ml.txt .

# Install dependencies
# Using --no-cache-dir to keep image size small
RUN pip install --no-cache-dir -r requirements-core.txt \
    && pip install --no-cache-dir -r requirements-ml.txt

# Copy source code
COPY src/ src/
COPY .gitignore . 
COPY serve_config.yaml .

# Expose Ray Serve HTTP port
EXPOSE 8000
# Expose Ray Dashboard port
EXPOSE 8265

# Default Command: Start Ray Head with Dashboard exposed, then run Serve
CMD ["bash", "-c", "ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-gpus=1 && serve run serve_config.yaml"]
