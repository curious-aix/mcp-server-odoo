FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies directly
RUN pip install --no-cache-dir \
    mcp>=1.9.4 \
    httpx>=0.27.0 \
    python-dotenv>=1.0.0 \
    pydantic>=2.0.0 \
    pydantic-settings>=2.0.0 \
    fastapi \
    uvicorn[standard]

# Copy the application code
COPY mcp_server_odoo/ ./mcp_server_odoo/

# Expose port
ENV PORT=8000
EXPOSE 8000

# Run the OpenAI-compatible API server
CMD ["python", "-m", "uvicorn", "mcp_server_odoo.openai_api:app", "--host", "0.0.0.0", "--port", "8000"]
