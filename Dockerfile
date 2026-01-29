FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY pyproject.toml .
COPY mcp_server_odoo/ ./mcp_server_odoo/

# Install the package with HTTP dependencies
RUN pip install --no-cache-dir . uvicorn[standard] fastapi

# Expose port
ENV PORT=8000
EXPOSE 8000

# Run the OpenAI-compatible API server
CMD ["python", "-m", "mcp_server_odoo.openai_api"]
