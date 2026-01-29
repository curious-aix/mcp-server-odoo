"""Standalone OpenAI-compatible API for Odoo ERP.

This is a self-contained FastAPI server that provides OpenAI function-calling
compatible endpoints for Odoo operations.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import uuid
import xmlrpc.client

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="MCP Server Odoo - OpenAI Compatible API",
    description="OpenAI function-calling compatible API for Odoo ERP operations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OdooConnection:
    """Simple Odoo XML-RPC connection handler."""
    
    def __init__(self):
        self.url = os.environ.get("ODOO_URL", "").rstrip("/")
        self.db = os.environ.get("ODOO_DB", "")
        self.username = os.environ.get("ODOO_USERNAME", "")
        self.api_key = os.environ.get("ODOO_API_KEY", "")
        self.uid = None
        self._models = None
        
    def connect(self):
        """Authenticate with Odoo."""
        if not all([self.url, self.db, self.username, self.api_key]):
            raise ValueError("Missing Odoo configuration. Set ODOO_URL, ODOO_DB, ODOO_USERNAME, ODOO_API_KEY")
        
        common = xmlrpc.client.ServerProxy(f"{self.url}/xmlrpc/2/common")
        self.uid = common.authenticate(self.db, self.username, self.api_key, {})
        
        if not self.uid:
            raise ValueError("Failed to authenticate with Odoo")
        
        self._models = xmlrpc.client.ServerProxy(f"{self.url}/xmlrpc/2/object")
        return self.uid
    
    @property
    def models(self):
        if self._models is None:
            self.connect()
        return self._models
    
    def execute(self, model: str, method: str, *args, **kwargs):
        """Execute an Odoo method."""
        if self.uid is None:
            self.connect()
        return self.models.execute_kw(
            self.db, self.uid, self.api_key,
            model, method, list(args), kwargs
        )
    
    def search(self, model: str, domain: list, **kwargs):
        return self.execute(model, "search", domain, **kwargs)
    
    def read(self, model: str, ids: list, fields: list = None):
        if fields:
            return self.execute(model, "read", ids, {"fields": fields})
        return self.execute(model, "read", ids)
    
    def search_read(self, model: str, domain: list, fields: list = None, **kwargs):
        options = kwargs.copy()
        if fields:
            options["fields"] = fields
        return self.execute(model, "search_read", domain, **options)
    
    def create(self, model: str, values: dict):
        return self.execute(model, "create", [values])
    
    def write(self, model: str, ids: list, values: dict):
        return self.execute(model, "write", ids, values)
    
    def unlink(self, model: str, ids: list):
        return self.execute(model, "unlink", ids)
    
    def fields_get(self, model: str, attributes: list = None):
        if attributes is None:
            attributes = ["string", "type", "required", "readonly", "selection"]
        return self.execute(model, "fields_get", [], {"attributes": attributes})


# Global connection
_connection: Optional[OdooConnection] = None


def get_connection() -> OdooConnection:
    global _connection
    if _connection is None:
        _connection = OdooConnection()
        _connection.connect()
    return _connection


# Pydantic models for OpenAI API compatibility
class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "odoo-mcp"
    messages: List[Message]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


# Define available Odoo tools in OpenAI function format
ODOO_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_records",
            "description": "Search for records in an Odoo model with optional filtering, field selection, and pagination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "The Odoo model name (e.g., 'res.partner', 'sale.order', 'product.product')"
                    },
                    "domain": {
                        "type": "string",
                        "description": "Odoo domain filter as JSON string, e.g., \"[['is_company', '=', true]]\""
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of field names to return"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of records to return (default: 10)",
                        "default": 10
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Number of records to skip (default: 0)",
                        "default": 0
                    },
                    "order": {
                        "type": "string",
                        "description": "Sort order (e.g., 'name asc', 'create_date desc')"
                    }
                },
                "required": ["model"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_record",
            "description": "Read a specific record by ID from an Odoo model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "The Odoo model name"},
                    "record_id": {"type": "integer", "description": "The ID of the record to read"},
                    "fields": {"type": "array", "items": {"type": "string"}, "description": "List of field names to return"}
                },
                "required": ["model", "record_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_record",
            "description": "Create a new record in an Odoo model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "The Odoo model name"},
                    "values": {"type": "object", "description": "Dictionary of field values for the new record"}
                },
                "required": ["model", "values"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_record",
            "description": "Update an existing record in an Odoo model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "The Odoo model name"},
                    "record_id": {"type": "integer", "description": "The ID of the record to update"},
                    "values": {"type": "object", "description": "Dictionary of field values to update"}
                },
                "required": ["model", "record_id", "values"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_record",
            "description": "Delete a record from an Odoo model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "The Odoo model name"},
                    "record_id": {"type": "integer", "description": "The ID of the record to delete"}
                },
                "required": ["model", "record_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_fields",
            "description": "Get field definitions for an Odoo model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "The Odoo model name"}
                },
                "required": ["model"]
            }
        }
    }
]


async def execute_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute an Odoo tool and return the result."""
    connection = get_connection()
    
    try:
        if name == "search_records":
            model = arguments["model"]
            domain = arguments.get("domain")
            fields = arguments.get("fields")
            limit = arguments.get("limit", 10)
            offset = arguments.get("offset", 0)
            order = arguments.get("order")
            
            if domain and isinstance(domain, str):
                domain = json.loads(domain)
            
            kwargs = {"limit": limit, "offset": offset}
            if order:
                kwargs["order"] = order
            
            records = connection.search_read(model, domain or [], fields, **kwargs)
            return {"success": True, "count": len(records), "records": records}
            
        elif name == "read_record":
            model = arguments["model"]
            record_id = arguments["record_id"]
            fields = arguments.get("fields")
            
            records = connection.read(model, [record_id], fields)
            if records:
                return {"success": True, "record": records[0]}
            return {"success": False, "error": f"Record {record_id} not found in {model}"}
            
        elif name == "create_record":
            model = arguments["model"]
            values = arguments["values"]
            
            record_id = connection.create(model, values)
            return {"success": True, "record_id": record_id, "message": f"Created record {record_id} in {model}"}
            
        elif name == "update_record":
            model = arguments["model"]
            record_id = arguments["record_id"]
            values = arguments["values"]
            
            connection.write(model, [record_id], values)
            return {"success": True, "message": f"Updated record {record_id} in {model}"}
            
        elif name == "delete_record":
            model = arguments["model"]
            record_id = arguments["record_id"]
            
            connection.unlink(model, [record_id])
            return {"success": True, "message": f"Deleted record {record_id} from {model}"}
            
        elif name == "get_model_fields":
            model = arguments["model"]
            fields_info = connection.fields_get(model)
            return {"success": True, "model": model, "fields": fields_info}
            
        else:
            return {"success": False, "error": f"Unknown tool: {name}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "MCP Server Odoo - OpenAI Compatible API",
        "version": "1.0.0",
        "endpoints": {
            "openai_chat": "/v1/chat/completions",
            "models": "/v1/models",
            "tools": "/v1/tools",
            "execute_tool": "/v1/tools/execute",
            "health": "/health",
            "mcp_info": "/mcp/info"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        connection = get_connection()
        return {"status": "healthy", "odoo_connected": connection.uid is not None}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [{
            "id": "odoo-mcp",
            "object": "model",
            "created": int(datetime.now().timestamp()),
            "owned_by": "mcp-server-odoo"
        }]
    }


@app.get("/v1/tools")
async def list_tools():
    """List available Odoo tools."""
    return {"tools": ODOO_TOOLS}


@app.get("/mcp/info")
async def mcp_info():
    """MCP server information."""
    return {
        "name": "mcp-server-odoo",
        "version": "1.0.0",
        "capabilities": {"tools": True, "resources": True},
        "tools": [t["function"]["name"] for t in ODOO_TOOLS],
        "transport": {
            "stdio": "python -m mcp_server_odoo",
            "http": "/v1/chat/completions"
        }
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions with function calling."""
    try:
        last_message = request.messages[-1] if request.messages else None
        
        # If last message has tool results, acknowledge
        if last_message and last_message.role == "tool":
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                created=int(datetime.now().timestamp()),
                model=request.model,
                choices=[Choice(
                    index=0,
                    message=Message(role="assistant", content=f"Tool result: {last_message.content}"),
                    finish_reason="stop"
                )],
                usage=Usage()
            )
        
        # If assistant had tool_calls, execute them
        if last_message and last_message.role == "assistant" and last_message.tool_calls:
            results = []
            for tc in last_message.tool_calls:
                args = json.loads(tc.function.arguments)
                result = await execute_tool(tc.function.name, args)
                results.append({"tool_call_id": tc.id, "result": result})
            
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                created=int(datetime.now().timestamp()),
                model=request.model,
                choices=[Choice(
                    index=0,
                    message=Message(role="assistant", content=json.dumps(results, indent=2)),
                    finish_reason="stop"
                )],
                usage=Usage()
            )
        
        # Default: return tools info
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(datetime.now().timestamp()),
            model=request.model,
            choices=[Choice(
                index=0,
                message=Message(
                    role="assistant",
                    content="I'm the Odoo MCP server. Use the available tools to interact with your Odoo ERP: search_records, read_record, create_record, update_record, delete_record, get_model_fields."
                ),
                finish_reason="stop"
            )],
            usage=Usage()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tools/execute")
async def execute_tool_endpoint(request: Request):
    """Direct tool execution endpoint."""
    try:
        body = await request.json()
        tool_name = body.get("name")
        arguments = body.get("arguments", {})
        
        if not tool_name:
            raise HTTPException(status_code=400, detail="Tool name is required")
        
        return await execute_tool(tool_name, arguments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
