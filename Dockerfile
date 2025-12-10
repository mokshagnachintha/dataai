# ============================================================================
# DOCKERFILE - How to Write It
# ============================================================================
# A Dockerfile is a blueprint to create a Docker image
# Think of it like instructions to build an environment
# ============================================================================

# STEP 1: Start from a base image
# Think: "What OS + pre-installed software do I want?"
FROM python:3.11-slim

# STEP 2: Set working directory inside container
# Think: "Where should files go inside the container?"
WORKDIR /app

# STEP 3: Copy files from your computer to container
# Syntax: COPY <from-your-computer> <to-container>
COPY demo.py .
COPY dataai/ ./dataai/

# STEP 4: Install Python dependencies
# This runs ONCE when building the image
RUN pip install --no-cache-dir \
    openai \
    pandas \
    pydantic \
    jinja2 \
    sqlglot \
    duckdb

# STEP 5: Set environment variables
# These are available inside the container
ENV PYTHONUNBUFFERED=1

# STEP 6: Expose ports (if needed)
# For web apps: EXPOSE 8000

# STEP 7: Run command when container starts
# Think: "What should happen when I start the container?"
CMD ["python", "demo.py"]

# ============================================================================
# HOW TO BUILD & RUN
# ============================================================================
# Build the image:
#   docker build -t dataai:latest .
#
# Run the container:
#   docker run -e OPENAI_API_KEY=sk-xxx -e OPENAI_API_BASE=https://openrouter.ai/api/v1 dataai:latest
#
# Or use .env file:
#   docker run --env-file .env dataai:latest
# ============================================================================
