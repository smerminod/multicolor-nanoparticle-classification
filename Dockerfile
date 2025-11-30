# ===================================================================
# Docker build for nanoparticle classification pipeline
# ===================================================================
FROM python:3.12-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Build argument to control installation of development dependencies
ARG INSTALL_DEV=false

# Production: install only runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development: install development dependencies
COPY requirements-dev.txt .
RUN if [ "$INSTALL_DEV" = "true" ] ; then \
        pip install --no-cache-dir -r requirements-dev.txt ; \
    fi

# Copy the rest of the project's code into the container's working directory
COPY . .

# Install package
RUN pip install --no-cache-dir .

# Create a user for the container for security
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# Set the default command to run when the container starts.
# This makes the container executable and allows passing command-line
# arguments directly to main.py.
ENTRYPOINT ["python", "main.py"]