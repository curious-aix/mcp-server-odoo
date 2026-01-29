FROM python:3.11-slim
WORKDIR /app

# Install dependencies directly (not from pyproject.toml)
RUN pip install --no-cache-dir \
    mcp>=1.0.0 \
    httpx>=0.28.0 \
    python-dotenv>=1.0.0 \
    pydantic>=2.0.0 \
    pydantic-settings>=2.0.0 \
    uvicorn[standard] \
    fastapi

# Copy only the source files we need
COPY mcp_server_odoo/ ./mcp_server_odoo/
COPY openai_api.py .

ENV PORT=8000
CMD ["uvicorn", "openai_api:app", "--host", "0.0.0.0", "--port", "8000"]
