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
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Constants
MAX_RESPONSE_SIZE = 50 * 1024  # 50KB max response size
DEFAULT_FIELDS = ["id", "name", "display_name"]

# Initialize FastAPI app
app = FastAPI(
    title="MCP Server Odoo - OpenAI Compatible API",
    description="OpenAI function-calling compatible API for Odoo ERP operations",
    version="1.1.0"
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
        """Get models proxy, connecting if needed."""
        if not self._models:
            self.connect()
        return self._models
    
    def execute(self, model: str, method: str, *args, **kwargs):
        """Execute an Odoo method."""
        if not self.uid:
            self.connect()
        return self.models.execute_kw(
            self.db, self.uid, self.api_key,
            model, method, args, kwargs
        )


# Global connection instance
odoo = OdooConnection()


def truncate_response(data: Any, max_size: int = MAX_RESPONSE_SIZE) -> Dict[str, Any]:
    """Truncate response if it exceeds max size to prevent ResponseTooLargeError."""
    json_str = json.dumps(data)
    if len(json_str) <= max_size:
        return {"data": data, "truncated": False}
    
    # If it's a list, truncate items
    if isinstance(data, list):
        truncated_data = []
        current_size = 2  # For []
        for item in data:
            item_str = json.dumps(item)
            if current_size + len(item_str) + 1 > max_size - 100:  # Leave room for metadata
                break
            truncated_data.append(item)
            current_size += len(item_str) + 1
        
        return {
            "data": truncated_data,
            "truncated": True,
            "original_count": len(data),
            "returned_count": len(truncated_data),
            "message": f"Response truncated: {len(data)} records reduced to {len(truncated_data)} to stay under {max_size // 1024}KB limit"
        }
    
    # For dict or other types, return a truncated message
    return {
        "data": None,
        "truncated": True,
        "message": f"Response too large ({len(json_str)} bytes). Use 'fields' parameter to limit returned data."
    }


# =============================================================================
# Pydantic Models for REST Endpoints
# =============================================================================

class SearchRequest(BaseModel):
    model: str
    domain: Optional[List] = None
    limit: Optional[int] = 100
    offset: Optional[int] = 0
    order: Optional[str] = None
    fields: Optional[List[str]] = None

class ReadRequest(BaseModel):
    model: str
    id: int
    fields: Optional[List[str]] = None

class CreateRequest(BaseModel):
    model: str
    values: Dict[str, Any]

class UpdateRequest(BaseModel):
    model: str
    id: int
    values: Dict[str, Any]

class DeleteRequest(BaseModel):
    model: str
    id: int

class FieldsRequest(BaseModel):
    model: str


# =============================================================================
# REST Endpoints for Direct Operations
# =============================================================================

@app.post("/v1/records/search")
async def search_records(request: SearchRequest):
    """Search for records in an Odoo model.
    
    Args:
        model: Odoo model name (e.g., 'res.partner', 'sale.order')
        domain: Odoo domain filter (e.g., [['is_company', '=', True]])
        limit: Maximum records to return (default: 100)
        offset: Number of records to skip (default: 0)
        order: Sort order (e.g., 'name asc, id desc')
        fields: List of fields to return (default: ['id', 'name', 'display_name'])
    
    Returns:
        List of matching records with specified fields
    """
    try:
        domain = request.domain or []
        fields = request.fields or DEFAULT_FIELDS
        
        # Search for record IDs
        ids = odoo.execute(
            request.model,
            'search',
            domain,
            offset=request.offset,
            limit=request.limit,
            order=request.order or False
        )
        
        if not ids:
            return {"data": [], "truncated": False, "count": 0}
        
        # Read the records with specified fields
        records = odoo.execute(request.model, 'read', ids, fields)
        
        result = truncate_response(records)
        result["count"] = len(records) if not result["truncated"] else result.get("returned_count", 0)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/records/read")
async def read_record(request: ReadRequest):
    """Read a single record by ID.
    
    Args:
        model: Odoo model name
        id: Record ID to read
        fields: List of fields to return (default: ['id', 'name', 'display_name'])
    
    Returns:
        Record data with specified fields
    """
    try:
        fields = request.fields or DEFAULT_FIELDS
        records = odoo.execute(request.model, 'read', [request.id], fields)
        
        if not records:
            raise HTTPException(status_code=404, detail=f"Record {request.id} not found in {request.model}")
        
        result = truncate_response(records[0])
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/records/create")
async def create_record(request: CreateRequest):
    """Create a new record.
    
    Args:
        model: Odoo model name
        values: Dictionary of field values for the new record
    
    Returns:
        ID of the created record
    """
    try:
        record_id = odoo.execute(request.model, 'create', request.values)
        return {"id": record_id, "success": True}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/records/update")
async def update_record(request: UpdateRequest):
    """Update an existing record.
    
    Args:
        model: Odoo model name
        id: Record ID to update
        values: Dictionary of field values to update
    
    Returns:
        Success status
    """
    try:
        result = odoo.execute(request.model, 'write', [request.id], request.values)
        return {"success": result, "id": request.id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/records/delete")
async def delete_record(request: DeleteRequest):
    """Delete a record.
    
    Args:
        model: Odoo model name
        id: Record ID to delete
    
    Returns:
        Success status
    """
    try:
        result = odoo.execute(request.model, 'unlink', [request.id])
        return {"success": result, "id": request.id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/models/fields")
async def get_model_fields(request: FieldsRequest):
    """Get field definitions for a model.
    
    Args:
        model: Odoo model name
    
    Returns:
        Dictionary of field names to field definitions
    """
    try:
        fields = odoo.execute(request.model, 'fields_get', [], {'attributes': ['string', 'type', 'required', 'readonly', 'selection', 'relation']})
        result = truncate_response(fields)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Original OpenAI-Compatible Tool Execution Endpoint (Backward Compatibility)
# =============================================================================

# Pydantic models for requests/responses (keep existing models)
class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

class ToolExecuteRequest(BaseModel):
    tool_calls: List[ToolCall]

class ToolResult(BaseModel):
    tool_call_id: str
    result: Any
    error: Optional[str] = None

class ToolExecuteResponse(BaseModel):
    results: List[ToolResult]


# Tool definitions
TOOLS = {
    "search_records": {
        "description": "Search for records in an Odoo model using domain filters",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Odoo model name (e.g., 'res.partner', 'sale.order')"},
                "domain": {"type": "array", "description": "Odoo domain filter", "default": []},
                "limit": {"type": "integer", "description": "Maximum records to return", "default": 100},
                "offset": {"type": "integer", "description": "Number of records to skip", "default": 0},
                "order": {"type": "string", "description": "Sort order (e.g., 'name asc')"},
                "fields": {"type": "array", "items": {"type": "string"}, "description": "Fields to return (default: id, name, display_name)"}
            },
            "required": ["model"]
        }
    },
    "read_record": {
        "description": "Read a specific record by ID",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Odoo model name"},
                "id": {"type": "integer", "description": "Record ID"},
                "fields": {"type": "array", "items": {"type": "string"}, "description": "Fields to return (default: id, name, display_name)"}
            },
            "required": ["model", "id"]
        }
    },
    "create_record": {
        "description": "Create a new record in an Odoo model",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Odoo model name"},
                "values": {"type": "object", "description": "Field values for the new record"}
            },
            "required": ["model", "values"]
        }
    },
    "update_record": {
        "description": "Update an existing record",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Odoo model name"},
                "id": {"type": "integer", "description": "Record ID to update"},
                "values": {"type": "object", "description": "Field values to update"}
            },
            "required": ["model", "id", "values"]
        }
    },
    "delete_record": {
        "description": "Delete a record",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Odoo model name"},
                "id": {"type": "integer", "description": "Record ID to delete"}
            },
            "required": ["model", "id"]
        }
    },
    "get_model_fields": {
        "description": "Get field definitions for a model",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Odoo model name"}
            },
            "required": ["model"]
        }
    }
}


def execute_tool(name: str, arguments: Dict[str, Any]) -> Any:
    """Execute an Odoo tool and return the result."""
    
    if name == "search_records":
        model = arguments["model"]
        domain = arguments.get("domain", [])
        limit = arguments.get("limit", 100)
        offset = arguments.get("offset", 0)
        order = arguments.get("order")
        fields = arguments.get("fields", DEFAULT_FIELDS)
        
        ids = odoo.execute(model, 'search', domain, offset=offset, limit=limit, order=order or False)
        if ids:
            records = odoo.execute(model, 'read', ids, fields)
            return truncate_response(records)
        return {"data": [], "truncated": False}
    
    elif name == "read_record":
        model = arguments["model"]
        record_id = arguments["id"]
        fields = arguments.get("fields", DEFAULT_FIELDS)
        records = odoo.execute(model, 'read', [record_id], fields)
        if records:
            return truncate_response(records[0])
        raise ValueError(f"Record {record_id} not found")
    
    elif name == "create_record":
        model = arguments["model"]
        values = arguments["values"]
        record_id = odoo.execute(model, 'create', values)
        return {"id": record_id, "success": True}
    
    elif name == "update_record":
        model = arguments["model"]
        record_id = arguments["id"]
        values = arguments["values"]
        result = odoo.execute(model, 'write', [record_id], values)
        return {"success": result}
    
    elif name == "delete_record":
        model = arguments["model"]
        record_id = arguments["id"]
        result = odoo.execute(model, 'unlink', [record_id])
        return {"success": result}
    
    elif name == "get_model_fields":
        model = arguments["model"]
        fields = odoo.execute(model, 'fields_get', [], {'attributes': ['string', 'type', 'required', 'readonly', 'selection', 'relation']})
        return truncate_response(fields)
    
    else:
        raise ValueError(f"Unknown tool: {name}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test Odoo connection
        odoo.connect()
        return {
            "status": "healthy",
            "odoo_connected": True,
            "odoo_uid": odoo.uid,
            "odoo_url": odoo.url,
            "odoo_db": odoo.db
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "odoo_connected": False,
                "error": str(e)
            }
        )


@app.get("/v1/tools")
async def list_tools():
    """List available tools in OpenAI function format."""
    tools = []
    for name, spec in TOOLS.items():
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": spec["parameters"]
            }
        })
    return {"tools": tools}


@app.post("/v1/tools/execute")
async def execute_tools(request: ToolExecuteRequest):
    """Execute tool calls (backward compatible endpoint).
    
    This endpoint is maintained for backward compatibility.
    Consider using the direct REST endpoints instead:
    - POST /v1/records/search
    - POST /v1/records/read
    - POST /v1/records/create
    - POST /v1/records/update
    - POST /v1/records/delete
    - POST /v1/models/fields
    """
    results = []
    
    for i, tool_call in enumerate(request.tool_calls):
        tool_result = ToolResult(
            tool_call_id=f"call_{i}",
            result=None
        )
        
        try:
            if tool_call.name not in TOOLS:
                tool_result.error = f"Unknown tool: {tool_call.name}"
            else:
                tool_result.result = execute_tool(tool_call.name, tool_call.arguments)
        except Exception as e:
            tool_result.error = str(e)
        
        results.append(tool_result)
    
    return ToolExecuteResponse(results=results)


# =============================================================================
# MCP Protocol Endpoints (unchanged)
# =============================================================================

@app.get("/mcp/tools")
async def mcp_list_tools():
    """List tools in MCP format."""
    tools = []
    for name, spec in TOOLS.items():
        tools.append({
            "name": name,
            "description": spec["description"],
            "inputSchema": spec["parameters"]
        })
    return {"tools": tools}


@app.post("/mcp/tools/{tool_name}")
async def mcp_execute_tool(tool_name: str, request: Request):
    """Execute a single tool via MCP protocol."""
    try:
        body = await request.json()
        arguments = body.get("arguments", {})
        
        if tool_name not in TOOLS:
            raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")
        
        result = execute_tool(tool_name, arguments)
        return {"result": result}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
