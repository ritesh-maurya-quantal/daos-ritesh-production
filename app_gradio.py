"""
app_gradio.py
Gradio web interface for a LangGraph agent that can query and modify a Notion database

Features
- Web chat interface with conversation history
- Notion Query Database tool with pagination + property normalization
- Create, update, and list pages in databases
- System prompt with your workspace schema so the model knows IDs & properties
- Conditional-edge routing fixed ("end": END) to avoid __end__ KeyError
- Optional streaming prints node updates (toggle STREAM_UPDATES)

Environment:
  NOTION_TOKEN=secret_...
  OPENAI_API_KEY=sk-...

Run:
  python app_gradio.py
"""
from timeit import default_timer as timer
import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

import gradio as gr

# ── Notion client (official Python SDK) ────────────────────────────────────────
from notion_client import Client as NotionClient

# ── LangGraph / LangChain imports ─────────────────────────────────────────────
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
notion = NotionClient(auth=NOTION_TOKEN)

# Choose any ChatOpenAI model that supports tool calling

# model = "gpt-3.5-turbo" 
model = "gpt-4o"

llm = ChatOpenAI(model=model, api_key=OPENAI_API_KEY, temperature=0)
# llm = ChatAnthropic(model=model, api_key=ANTHROPIC_API_KEY, temperature=0)
# Toggle this if you want to see per-node streaming updates
STREAM_UPDATES = True
 
# ──────────────────────────────────────────────────────────────────────────────
# Your Notion Workspace Schema (used in the system prompt)
# ──────────────────────────────────────────────────────────────────────────────
DB = {
    "Projects":  "23c65a04-5ba4-808e-b0d2-f2afe3038a5d",
    "Tasks":     "23c65a04-5ba4-80dc-8eb9-f40fa7d18a19",
    "Clients":   "23c65a04-5ba4-801f-8c8f-c71773634e04",
    "Goals":     "23c65a04-5ba4-8013-833b-e6f8621299b5",
    "Milestones":"23c65a04-5ba4-80aa-9b3a-d66540233727",
}

SYSTEM_CONTEXT = f"""
You are a Notion data agent. You can fetch, create, update, and manage rows in Notion databases using the following tools:

1. `fetch_notion_db_pages` - Query and retrieve data from databases
2. `create_notion_page` - Create new pages in databases  
3. `update_notion_page` - Update existing pages
4. `list_notion_pages` - List pages from a database with basic info
5. `get_database_properties` - Get schema/properties of a database
6. `list_available_databases` - List all available databases configured in this workspace

Database IDs:
- Projects:   {DB["Projects"]}
- Tasks:      {DB["Tasks"]}
- Clients:    {DB["Clients"]}
- Goals:      {DB["Goals"]}
- Milestones: {DB["Milestones"]}

General guidance:
- If the user says a database name ("Projects", "Tasks", etc.), pass the correct `database_id` from the mapping above.
- Use `list_available_databases` to get the latest database info if they change or are inaccessible.
- Use Notion's query filter/sorts schema when needed. For example, to filter
  Status equals "In progress", build:
  {{
    "property": "Status",
    "status": {{"equals": "In progress"}}
  }}
- For creating/updating pages, format properties according to Notion's schema:
  - title: {{"title": [{{"text": {{"content": "value"}}}}]}} 
  - rich_text: {{"rich_text": [{{"text": {{"content": "value"}}}}]}} 
  - select: {{"select": {{"name": "value"}}}} 
  - status: {{"status": {{"name": "value"}}}} 
  - number: {{"number": 123}} 
  - date: {{"date": {{"start": "2024-01-01"}}}} 
  - people: {{"people": [{{"id": "user_id"}}]}}
  - relation: {{"relation": [{{"id": "page_id"}}]}} 
  - url: {{"url": "https://example.com"}}
  - email: {{"email": "user@example.com"}}
  - phone_number: {{"phone_number": "+1234567890"}}
- NEVER try to set rollup or formula properties - they are computed automatically
- When filtering by status, use exact status names (case-sensitive)
- For relations, you need the actual page IDs to link items
- If the user says "show fields A, B, C", fetch full rows and then display only those fields
- Limit large outputs; if user doesn't specify, return first 10 rows
- When creating or updating, always confirm the action was successful and show the page ID

Return concise, tabular summaries when appropriate. Handle rollup and formula fields gracefully by displaying their computed values.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Utilities to normalize Notion property values → Python primitives
# ──────────────────────────────────────────────────────────────────────────────
def _rich_text_to_str(rich_text: List[Dict[str, Any]]) -> str:
    return "".join([span.get("plain_text", "") for span in (rich_text or [])])

def _prop_value_to_python(prop: Dict[str, Any]) -> Any:
    """Convert a single Notion property object to a simple Python value."""
    t = prop.get("type")
    if t == "title":
        return _rich_text_to_str(prop.get("title", []))
    if t == "rich_text":
        return _rich_text_to_str(prop.get("rich_text", []))
    if t == "select":
        val = prop.get("select")
        return val["name"] if val else None
    if t == "status":
        val = prop.get("status")
        return val["name"] if val else None
    if t == "multi_select":
        vals = prop.get("multi_select", [])
        return [v.get("name") for v in vals]
    if t == "number":
        return prop.get("number")
    if t == "date":
        d = prop.get("date")
        return d.get("start") if d else None
    if t == "people":
        ppl = prop.get("people", [])
        # Names are best-effort; may be None depending on permissions
        return [p.get("name") or p.get("id") for p in ppl]
    if t == "url":
        return prop.get("url")
    if t == "email":
        return prop.get("email")
    if t == "phone_number":
        return prop.get("phone_number")
    if t == "relation":
        rel = prop.get("relation", [])
        return [r.get("id") for r in rel]
    if t == "rollup":
        # Handle rollup properties - return the computed value
        rollup = prop.get("rollup", {})
        rollup_type = rollup.get("type")
        if rollup_type == "number":
            return rollup.get("number")
        elif rollup_type == "array":
            array_vals = rollup.get("array", [])
            return [_prop_value_to_python({"type": item.get("type"), item.get("type"): item.get(item.get("type"))}) for item in array_vals if item.get("type")]
        else:
            return rollup.get(rollup_type) if rollup_type else None
    if t == "formula":
        # Handle formula properties - return the computed value
        formula = prop.get("formula", {})
        formula_type = formula.get("type")
        if formula_type in ["string", "number", "boolean", "date"]:
            return formula.get(formula_type)
        else:
            return str(formula) if formula else None
    # Fallback to raw storage of the type value if unknown
    return prop.get(t, None)

def _page_to_row(page: Dict[str, Any]) -> Dict[str, Any]:
    props = page.get("properties", {})
    parsed = {name: _prop_value_to_python(value) for name, value in props.items()}
    parsed["_page_id"] = page.get("id")
    parsed["_created_time"] = page.get("created_time")
    parsed["_last_edited_time"] = page.get("last_edited_time")
    return parsed

# ──────────────────────────────────────────────────────────────────────────────
# CRUD Operations (integrated from notion_updater.py)
# ──────────────────────────────────────────────────────────────────────────────
def create_notion_page(database_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new page in the specified database."""
    try:
        response = notion.pages.create(
            parent={"database_id": database_id},
            properties=properties
        )
        return {
            "success": True,
            "page_id": response["id"],
            "url": response.get("url", ""),
            "message": "Page created successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error creating page: {e}"
        }

def update_notion_page(page_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing page with new properties."""
    try:
        response = notion.pages.update(
            page_id=page_id,
            properties=properties
        )
        return {
            "success": True,
            "page_id": page_id,
            "url": response.get("url", ""),
            "message": "Page updated successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error updating page: {e}"
        }

def list_notion_pages(database_id: str, limit: int = 10) -> Dict[str, Any]:
    """List pages from a database with basic information."""
    try:
        response = notion.databases.query(
            database_id=database_id,
            page_size=min(limit, 100) #change max page size when needed
        )
        pages = response["results"]
        
        page_list = []
        for page in pages:
            page_info = {
                "page_id": page["id"],
                "created_time": page.get("created_time"),
                "last_edited_time": page.get("last_edited_time"),
                "url": page.get("url", "")
            }
            
            # Try to get title/name
            properties = page.get("properties", {})
            for prop_name, prop_value in properties.items():
                if prop_value.get("type") == "title":
                    title_list = prop_value.get("title", [])
                    if title_list:
                        page_info["title"] = title_list[0].get("text", {}).get("content", "Untitled")
                        break
            else:
                page_info["title"] = "Untitled"
            
            page_list.append(page_info)
        
        return {
            "success": True,
            "count": len(page_list),
            "pages": page_list,
            "has_more": response.get("has_more", False)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error listing pages: {e}"
        }

def get_database_properties(database_id: str) -> Dict[str, Any]:
    """Get the schema/properties of a database."""
    try:
        response = notion.databases.retrieve(database_id=database_id)
        properties = response["properties"]
        
        # Simplify property information for the AI
        simplified_props = {}
        for prop_name, prop_info in properties.items():
            prop_type = prop_info["type"]
            prop_data = {"type": prop_type}
            
            # Add options for select/status properties
            if prop_type == "select":
                options = prop_info.get("select", {}).get("options", [])
                prop_data["options"] = [opt["name"] for opt in options]
            elif prop_type == "status":
                options = prop_info.get("status", {}).get("options", [])
                prop_data["options"] = [opt["name"] for opt in options]
            elif prop_type == "rollup":
                prop_data["note"] = "Computed field - cannot be set directly"
            elif prop_type == "formula":
                prop_data["note"] = "Computed field - cannot be set directly"
            elif prop_type == "relation":
                relation_db = prop_info.get("relation", {}).get("database_id")
                if relation_db:
                    # Try to identify which database this relates to
                    for db_name, db_id in DB.items():
                        if db_id == relation_db:
                            prop_data["relates_to"] = db_name
                            break
            
            simplified_props[prop_name] = prop_data
        
        return {
            "success": True,
            "database_id": database_id,
            "title": response.get("title", [{}])[0].get("text", {}).get("content", "Unknown"),
            "properties": simplified_props
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error getting database properties: {e}"
        }

def list_available_databases() -> Dict[str, Any]:
    """List all available databases that are configured in this workspace."""
    try:
        databases = []
        for name, db_id in DB.items():
            try:
                # Try to get database info to verify it's accessible
                response = notion.databases.retrieve(database_id=db_id)
                title = response.get("title", [{}])[0].get("text", {}).get("content", name)
                databases.append({
                    "name": name,
                    "database_id": db_id,
                    "title": title,
                    "created_time": response.get("created_time"),
                    "last_edited_time": response.get("last_edited_time")
                })
            except Exception as e:
                # Database might not be accessible, but still list it
                databases.append({
                    "name": name,
                    "database_id": db_id,
                    "title": name,
                    "error": f"Not accessible: {str(e)}"
                })
        
        return {
            "success": True,
            "count": len(databases),
            "databases": databases
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error listing databases: {e}"
        }

# ──────────────────────────────────────────────────────────────────────────────
# Tool: fetch Notion database pages (with pagination)
# ──────────────────────────────────────────────────────────────────────────────
def fetch_notion_db_pages(
    database_id: str,
    filter: Optional[Dict[str, Any]] = None,
    sorts: Optional[List[Dict[str, Any]]] = None,
    page_size: int = 100,
    max_pages: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Query a Notion database and return normalized rows.

    Args:
        database_id: The Notion database ID (e.g., "23c65a04-...").
        filter: Optional Notion filter JSON (see Notion docs).
        sorts: Optional Notion sorts JSON (list of { "property": ..., "direction": ... }).
        page_size: Page size for each Notion API request (max 100).
        max_pages: Optional limit on the number of pages of results to fetch.

    Returns:
        dict with:
            - count: number of normalized rows
            - rows: list of dicts (normalized properties + _page_id, timestamps)
            - has_more: whether more results exist beyond next_cursor
            - next_cursor: cursor to continue from
    """
    all_pages: List[Dict[str, Any]] = []
    cursor = None
    pages_fetched = 0

    while True:
        payload: Dict[str, Any] = {"database_id": database_id, "page_size": page_size}
        if filter:
            payload["filter"] = filter
        if sorts:
            payload["sorts"] = sorts
        if cursor:
            payload["start_cursor"] = cursor

        resp = notion.databases.query(**payload)
        results = resp.get("results", [])
        all_pages.extend(results)

        cursor = resp.get("next_cursor")
        has_more = resp.get("has_more", False)
        pages_fetched += 1

        if not has_more:
            break
        if max_pages is not None and pages_fetched >= max_pages:
            break

    rows = [_page_to_row(p) for p in all_pages]
    return {
        "count": len(rows),
        "rows": rows,
        "has_more": cursor is not None,
        "next_cursor": cursor,
    }

# ──────────────────────────────────────────────────────────────────────────────
# LangGraph: AI agent with all tools
# ──────────────────────────────────────────────────────────────────────────────
# Bind all tools to the model
all_tools = [
    fetch_notion_db_pages,
    create_notion_page,
    update_notion_page,
    list_notion_pages,
    get_database_properties,
    list_available_databases
]

model_with_tools = llm.bind_tools(all_tools)

def call_model(state: MessagesState):
    """Model node: let the LLM decide whether to call tools."""
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: MessagesState):
    """
    Router: if last AI message requested a tool, go to tools; else end.
    Returning string labels and mapping "end": END avoids the __end__ KeyError.
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "end"

# Build the graph
graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(all_tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")
app = graph.compile()

# ──────────────────────────────────────────────────────────────────────────────
# Gradio Chat Interface
# ──────────────────────────────────────────────────────────────────────────────
class NotionChatBot:
    def __init__(self):
        self.messages: List[Any] = [SystemMessage(content=SYSTEM_CONTEXT)]
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.messages = [SystemMessage(content=SYSTEM_CONTEXT)]
        return "Conversation reset."
    
    def chat(self, message: str, history: List[List[str]]) -> str:
        """Process a chat message and return the response"""
        if not message.strip():
            return "Please enter a message."
        
        # Handle special commands
        if message.lower() in ("/reset", "/clear"):
            self.reset_conversation()
            return "Conversation reset. You can start a new conversation."
        
        # Add user message to conversation
        self.messages.append(HumanMessage(content=message))
        
        # Timing starts here
        start_time = timer()
        ttfu_time = None
        
        try:
            if STREAM_UPDATES:
                # Show node-by-node updates and capture time-to-first-update
                final_state: Optional[Dict[str, Any]] = None
                updates_log = []
                
                for update in app.stream({"messages": self.messages}, stream_mode="updates"):
                    if ttfu_time is None:
                        ttfu_time = timer() - start_time
                    for node, state_piece in update.items():
                        if node == "__end__":
                            continue
                        updates_log.append(f"[update] node={node}")
                        final_state = state_piece
                
                if final_state is None:
                    final_state = app.invoke({"messages": self.messages})
                out = final_state
            else:
                out = app.invoke({"messages": self.messages})
        
        except Exception as e:
            elapsed = timer() - start_time
            error_msg = f"Encountered an error after {elapsed:.2f}s: {str(e)}"
            # Don't advance message history on failure
            self.messages.pop()  # Remove the failed user message
            return error_msg
        
        elapsed = timer() - start_time
        
        # Update conversation state
        self.messages = out["messages"]
        
        # Get the final AI response
        final_ai_msgs = [m for m in self.messages if isinstance(m, AIMessage)]
        if final_ai_msgs:
            response = final_ai_msgs[-1].content
        else:
            response = "(no AI message produced)"
        
        # Add timing information
        timing_info = f"\n\n---\n⏱️ Response time: {elapsed:.2f}s"
        if STREAM_UPDATES and ttfu_time is not None:
            timing_info += f" | Time to first update: {ttfu_time:.2f}s"
        
        # Log the interaction
        # try:
        #     with open(f"test logs/{model} test log.txt", "a", encoding="utf-8") as f:
        #         f.write(f"User query: {message}\nModel: {model}\nResponse length: {len(response)}\nResponse time: {elapsed:.2f}\n\n\n")
        # except Exception:
        #     pass  # Ignore logging errors
        
        return response + timing_info

# Initialize the chatbot
chatbot = NotionChatBot()

def create_gradio_interface():
    """Create and configure the Gradio interface"""
    
    with gr.Blocks(title="Notion Agent Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🗃️ Notion Agent Chat
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                chatinterface = gr.ChatInterface(
                    fn=chatbot.chat,
                    examples=[
                        "List all projects",
                        "Show me tasks that are in progress",
                        "Create a new task called 'Review documentation'",
                        "What databases are available?",
                        "Show me the schema for the Tasks database"
                    ],
                    textbox=gr.Textbox(
                        placeholder="Ask me about your Notion data...",
                        container=False
                    )
                )
            
            # with gr.Column(scale=1, min_width=300):
            #     gr.Markdown("### ⚙️ Configuration")
                
            #     with gr.Group():
            #         gr.Markdown(f"**Model:** {model}")
            #         gr.Markdown(f"**Streaming:** {'Enabled' if STREAM_UPDATES else 'Disabled'}")
                
            #     gr.Markdown("### 📊 Database IDs")
            #     db_info = "\n".join([f"**{name}:** `{db_id}`" for name, db_id in DB.items()])
            #     gr.Markdown(db_info)
                
            #     gr.Markdown("### 💡 Tips")
            #     gr.Markdown("""
            #     - Be specific about which database you want to query
            #     - Use exact status names when filtering (case-sensitive)
            #     - The agent can create, read, update, and list pages
            #     - Rollup and formula fields are read-only
            #     """)
    
    return demo

def main():
    """Main function to launch the Gradio interface"""
    # Check environment variables
    if not NOTION_TOKEN:
        print("❌ NOTION_TOKEN environment variable is required")
        sys.exit(1)
    
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY environment variable is required")
        sys.exit(1)
    
    print(f"🚀 Starting Notion Agent Chat with model: {model}")
    print(f"📡 Streaming updates: {'Enabled' if STREAM_UPDATES else 'Disabled'}")
    
    # Create and launch the interface
    demo = create_gradio_interface()
    
    # Launch with appropriate settings
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True if you want a public link
        debug=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
    
# if __name__ == "__main__":
#     # Remove the old repl() call and replace with main()
#     main()
