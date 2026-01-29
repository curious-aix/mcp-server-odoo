"""OpenAI-compatible API layer for MCP Server Odoo.

This module provides an HTTP API that exposes the MCP tools in a format
compatible with OpenAI's function calling API, enabling ChatGPT integration.
"""

import json
import os
import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import get_config
from .odoo_connection import OdooConnection
from .access_control import AccessController
from .logging_config import get_logger

logger = get_logger(__name__)

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

# Global connection instance
_connection: Optional[OdooConnection] = None
_access_controller: Optional[AccessController] = None
_config = None


def get_connection() -> OdooConnection:
    """Get or create Odoo connection."""
    global _connection, _access_controller, _config
    if _connection is None:
        _config = get_config()
        _connection = OdooConnection(_config)
        _connection.connect()
        _access_controller = AccessController(_config, _connection)
    return _connection


def get_access_controller() -> AccessController:
    """Get access controller instance."""
    global _access_controller
    if _access_controller is None:
        get_connection()
    return _access_controller


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
                        "description": "List of field names to return. Use ['__all__'] for all fields."
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
                    "model": {
                        "type": "string",
                        "description": "The Odoo model name"
                    },
                    "record_id": {
                        "type": "integer",
                        "description": "The ID of the record to read"
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of field names to return"
                    }
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
                    "model": {
                        "type": "string",
                        "description": "The Odoo model name"
                    },
                    "values": {
                        "type": "object",
                        "description": "Dictionary of field values for the new record"
                    }
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
                    "model": {
                        "type": "string",
                        "description": "The Odoo model name"
                    },
                    "record_id": {
                        "type": "integer",
                        "description": "The ID of the record to update"
                    },
                    "values": {
                        "type": "object",
                        "description": "Dictionary of field values to update"
                    }
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
                    "model": {
                        "type": "string",
                        "description": "The Odoo model name"
                    },
                    "record_id": {
                        "type": "integer",
                        "description": "The ID of the record to delete"
                    }
                },
                "required": ["model", "record_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_models",
            "description": "List available Odoo models with their access permissions.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
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
                    "model": {
                        "type": "string",
                        "description": "The Odoo model name"
                    }
                },
                "required": ["model"]
            }
        }
    }
]


async def execute_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute an Odoo tool and return the result."""
    connection = get_connection()
    access_controller = get_access_controller()
    
    try:
        if name == "search_records":
            model = arguments["model"]
            domain = arguments.get("domain")
            fields = arguments.get("fields")
            limit = arguments.get("limit", 10)
            offset = arguments.get("offset", 0)
            order = arguments.get("order")
            
            # Parse domain if it's a string
            if domain and isinstance(domain, str):
                domain = json.loads(domain)
            
            # Search for records
            record_ids = connection.search(model, domain or [], limit=limit, offset=offset, order=order)
            
            # Read records
            if record_ids:
                records = connection.read(model, record_ids, fields)
                return {"success": True, "count": len(records), "records": records}
            return {"success": True, "count": 0, "records": []}
            
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
            
        elif name == "list_models":
            # Get enabled models from access controller
            enabled_models = access_controller.get_enabled_models()
            return {"success": True, "models": enabled_models}
            
        elif name == "get_model_fields":
            model = arguments["model"]
            fields_info = connection.fields_get(model)
            return {"success": True, "model": model, "fields": fields_info}
            
        else:
            return {"success": False, "error": f"Unknown tool: {name}"}
            
    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        return {"success": False, "error": str(e)}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "MCP Server Odoo - OpenAI Compatible API",
        "version": "1.0.0",
        "description": "OpenAI function-calling compatible API for Odoo ERP operations",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
            "tools": "/v1/tools",
            "health": "/health",
            "mcp_info": "/mcp/info"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        connection = get_connection()
        return {"status": "healthy", "odoo_connected": connection.is_authenticated}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "odoo-mcp",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "mcp-server-odoo",
                "permission": [],
                "root": "odoo-mcp",
                "parent": None
            }
        ]
    }


@app.get("/v1/tools")
async def list_tools():
    """List available Odoo tools."""
    return {
        "tools": ODOO_TOOLS
    }


@app.get("/mcp/info")
async def mcp_info():
    """MCP server information for native MCP clients."""
    return {
        "name": "mcp-server-odoo",
        "version": "1.0.0",
        "description": "MCP server for Odoo ERP integration",
        "capabilities": {
            "tools": True,
            "resources": True
        },
        "tools": [t["function"]["name"] for t in ODOO_TOOLS],
        "transport": {
            "stdio": "python -m mcp_server_odoo",
            "http": "/v1/chat/completions"
        }
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint with function calling support.
    
    This endpoint processes messages and can execute Odoo tools when requested.
    """
    try:
        # Check if we need to execute a tool call
        last_message = request.messages[-1] if request.messages else None
        
        # If the last message is a tool result, just acknowledge it
        if last_message and last_message.role == "tool":
            response_message = Message(
                role="assistant",
                content=f"Tool execution completed. Result: {last_message.content}"
            )
            
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                created=int(datetime.now().timestamp()),
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        message=response_message,
                        finish_reason="stop"
                    )
                ],
                usage=Usage()
            )
        
        # Check if we should suggest/execute tool calls
        tools_to_use = request.tools or ODOO_TOOLS
        
        # If the last message is from user, analyze if we need to call a tool
        if last_message and last_message.role == "user":
            content = last_message.content or ""
            
            # Simple intent detection (in production, you'd use the LLM for this)
            tool_call = None
            
            # Check for search-related queries
            if any(word in content.lower() for word in ["search", "find", "list", "show", "get"]):
                # Try to extract model name from common patterns
                for model_hint in ["partner", "contact", "customer", "order", "product", "invoice", "user"]:
                    if model_hint in content.lower():
                        model_map = {
                            "partner": "res.partner",
                            "contact": "res.partner",
                            "customer": "res.partner",
                            "order": "sale.order",
                            "product": "product.product",
                            "invoice": "account.move",
                            "user": "res.users"
                        }
                        model = model_map.get(model_hint, "res.partner")
                        
                        tool_call = ToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            type="function",
                            function=FunctionCall(
                                name="search_records",
                                arguments=json.dumps({"model": model, "limit": 10})
                            )
                        )
                        break
            
            # If we identified a tool to call, return with tool_calls
            if tool_call:
                response_message = Message(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call]
                )
                
                return ChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    created=int(datetime.now().timestamp()),
                    model=request.model,
                    choices=[
                        Choice(
                            index=0,
                            message=response_message,
                            finish_reason="tool_calls"
                        )
                    ],
                    usage=Usage()
                )
        
        # Check if we need to execute tool calls from assistant message
        if last_message and last_message.role == "assistant" and last_message.tool_calls:
            # Execute each tool call
            results = []
            for tool_call in last_message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                result = await execute_tool(tool_call.function.name, args)
                results.append({
                    "tool_call_id": tool_call.id,
                    "result": result
                })
            
            # Return the results
            response_message = Message(
                role="assistant",
                content=json.dumps(results, indent=2)
            )
            
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                created=int(datetime.now().timestamp()),
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        message=response_message,
                        finish_reason="stop"
                    )
                ],
                usage=Usage()
            )
        
        # Default response - return available tools info
        response_message = Message(
            role="assistant",
            content="I'm the Odoo MCP server. I can help you interact with your Odoo ERP system. Available operations: search records, read records, create records, update records, delete records, list models, and get model fields. What would you like to do?"
        )
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(datetime.now().timestamp()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=response_message,
                    finish_reason="stop"
                )
            ],
            usage=Usage()
        )
        
    except Exception as e:
        logger.error(f"Error in chat completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tools/execute")
async def execute_tool_endpoint(request: Request):
    """
    Direct tool execution endpoint.
    
    This is a simpler endpoint for direct tool execution without the chat format.
    """
    try:
        body = await request.json()
        tool_name = body.get("name")
        arguments = body.get("arguments", {})
        
        if not tool_name:
            raise HTTPException(status_code=400, detail="Tool name is required")
        
        result = await execute_tool(tool_name, arguments)
        return result
        
    except Exception as e:
        logger.error(f"Error executing tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import sys
    port = int(os.environ.get("PORT", 8000))
    run_server(port=port)
