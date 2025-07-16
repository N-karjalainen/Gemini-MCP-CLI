import asyncio
import sys
import os
from dotenv import load_dotenv
from contextlib import AsyncExitStack
from typing import List, Dict, Any, Optional, Tuple
import logging

from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging for the app
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables (for GEMINI_API_KEY)
load_dotenv()


class MCPChatApp:
    def __init__(self, gemini_model_name: str = "gemini-2.0-flash-001"):
        self.gemini_model_name = gemini_model_name
        self.gemini_sync_client: Optional[genai.Client] = None
        self.gemini_client: Optional[genai.client.AsyncClient] = None
        self.exit_stack = AsyncExitStack()
        self.mcp_sessions: List[ClientSession] = []
        self.mcp_tools: List[Any] = []
        self.tool_to_session: Dict[str, ClientSession] = {}
        self.chat_history: List[genai_types.Content] = []
        self.connected_server_paths: set[str] = set()
        # --- Caching ---
        self.cached_gemini_declarations: Optional[List[genai_types.FunctionDeclaration]] = None
        self.gemini_tools_dirty: bool = True
        # ---------------

    async def initialize_gemini(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables.")
            raise ValueError("GEMINI_API_KEY is required.")
        try:
            self.gemini_sync_client = genai.Client(api_key=api_key)
            self.gemini_client = self.gemini_sync_client.aio
            logger.info("Gemini async client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise

    async def connect_to_mcp_server(self, server_script_path: str):
        if not os.path.exists(server_script_path):
            logger.error(f"MCP server script not found: {server_script_path}")
            raise FileNotFoundError(
                f"Server script not found: {server_script_path}")

        if server_script_path in self.connected_server_paths:
            logger.warning(
                f"Server script '{server_script_path}' is already connected. Skipping.")
            raise ValueError(
                f"Server '{server_script_path}' is already connected.")

        try:
            logger.info(
                f"Connecting to MCP server: {server_script_path} using command: '{command}' with args: {args}")
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()
            self.mcp_sessions.append(session)
            self.connected_server_paths.add(server_script_path)
            logger.info(f"Connected to MCP server: {server_script_path}")

            response = await session.list_tools()
            server_tools = response.tools
            logger.info(
                f"Server {server_script_path} provides tools: {[tool.name for tool in server_tools]}")
            added_tools = []
            for tool in server_tools:
                if tool.name in self.tool_to_session:
                    logger.warning(
                        f"Tool name conflict: '{tool.name}' already exists. Skipping tool from {server_script_path}.")
                else:
                    self.mcp_tools.append(tool)
                    self.tool_to_session[tool.name] = session
                    added_tools.append(tool.name)
                    self.gemini_tools_dirty = True  # Mark as dirty if new tools are added
            return added_tools
        except Exception as e:
            logger.error(
                f"Failed to connect to or initialize MCP server {server_script_path}: {e}", exc_info=True)
            raise


    # ... (rest of your MCPChatApp class and main function) ...

    async def chat_loop(self):
        print("\nMCP Gemini Chat App")
        print("Enter your message, 'add_server <path>' to add a server, or 'quit' to exit.")
        # New default prompt placeholder logic here
        while True:
            # Check for a default prompt from an environment variable or set a hardcoded one
            # if you want non-interactive runs
            default_prompt = os.getenv("DEFAULT_MCP_PROMPT")
            if default_prompt:
                query = default_prompt
                print(f"\nGemini (Auto-prompt): {query}")
                # Clear the env var so it only runs once per execution, unless explicitly reset
                del os.environ["DEFAULT_MCP_PROMPT"]
            else:
                query = input("\nYou: ").strip()

            if query.lower() == 'quit':
                break
            if not query:
                continue

            if query.lower().startswith("add_server "):
                parts = query.split(" ", 1)
                if len(parts) == 2 and parts[1]:
                    server_path = parts[1].strip()
                    await self.add_server_runtime(server_path)
                else:
                    print("\nGemini: Usage: add_server <path_to_server_script>")
            else:
                response = await self.process_query(query)
                print(f"\nGemini: {response}")

            # If a default prompt was used, exit after processing it
            if default_prompt:
                print("\nAuto-prompt processed. Exiting.")
                break

            # ... rest of chat_loop error handling