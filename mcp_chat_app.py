# mcp_chat_app.py
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
logging.basicConfig(level=logging.DEBUG, # Changed to DEBUG for more verbose output
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

        command: str
        args: List[str]
        env_vars = os.environ.copy()

        # Define the absolute path to the ServiceNow MCP server's Python executable
        # !!! IMPORTANT: YOU MUST UPDATE THIS PATH to your specific environment !!!
        # This path should point to the 'python.exe' inside the .venv\Scripts\
        # folder of your cloned 'servicenow-mcp-echelon' project.
        SERVICENOW_MCP_PYTHON_EXE = r"C:\Users\donvi\Documents\GitHub\servicenow-mcp-echelon\.venv\Scripts\python.exe"

        if "servicenow-mcp-echelon" in server_script_path.lower():
            logger.info(f"Detected ServiceNow MCP server at {server_script_path}. Using specific launch command.")
            command = SERVICENOW_MCP_PYTHON_EXE
            args = ["-m", "servicenow_mcp.cli"]
        elif server_script_path.endswith('.py'):
            logger.info(f"Detected generic Python server at {server_script_path}. Using current Python executable.")
            command = sys.executable
            args = [server_script_path]
        elif server_script_path.endswith('.js'):
            logger.info(f"Detected Node.js server at {server_script_path}. Using 'node' command.")
            command = "node"
            args = [server_script_path]
        else:
            logger.warning(
                f"Unsupported server script type: {server_script_path}. Skipping.")
            raise ValueError(
                f"Unsupported server script type: {server_script_path}")

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env_vars
        )

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
                    self.gemini_tools_dirty = True
            return added_tools
        except Exception as e:
            logger.error(
                f"Failed to connect to or initialize MCP server {server_script_path}: {e}", exc_info=True)
            raise

    async def add_server_runtime(self, server_script_path: str):
        try:
            added_tools = await self.connect_to_mcp_server(server_script_path)
            if added_tools:
                print(
                    f"\nGemini: Successfully added server '{server_script_path}' with tools: {added_tools}")
                all_tool_names = [tool.name for tool in self.mcp_tools]
                logger.info(f"Total available tools now: {all_tool_names}")
            else:
                print(
                    f"\nGemini: Connected to server '{server_script_path}', but no new, non-conflicting tools were added.")
        except FileNotFoundError as e:
            print(f"\nGemini: Error adding server: {e}")
        except ValueError as e:
            print(f"\nGemini: Error adding server: {e}")
        except Exception as e:
            print(
                f"\nGemini: Failed to add server '{server_script_path}': {e}")
            logger.error(
                f"Error adding server {server_script_path} at runtime: {e}", exc_info=True)

    def get_gemini_tool_declarations(self) -> List[genai_types.FunctionDeclaration]:
        if not self.gemini_tools_dirty and self.cached_gemini_declarations is not None:
            logger.debug("Using cached Gemini tool declarations.")
            return self.cached_gemini_declarations

        logger.info("Generating Gemini tool declarations.")
        declarations = []
        type_mapping = {
            'string': 'STRING',
            'number': 'NUMBER',
            'integer': 'INTEGER',
            'boolean': 'BOOLEAN',
            'array': 'ARRAY',
            'object': 'OBJECT',
            # Add other types if needed
        }

        # Helper function to convert a single MCP schema property to Gemini Schema
        def _convert_mcp_prop_to_gemini_schema(prop_schema_dict: Dict[str, Any]) -> Optional[genai_types.Schema]:
            mcp_type = prop_schema_dict.get('type', '').lower()
            gemini_type_str = type_mapping.get(mcp_type)

            if not gemini_type_str:
                logger.warning(
                    f"Unmappable MCP type '{mcp_type}'. Skipping property.")
                return None

            gemini_schema_kwargs = {
                "type": gemini_type_str,
                "description": prop_schema_dict.get('description')
            }

            if mcp_type == 'array':
                # --- CRITICAL CHANGE FOR ARRAY TYPE ---
                # Recursively convert the 'items' schema for arrays
                mcp_items_schema = prop_schema_dict.get('items')
                if mcp_items_schema and isinstance(mcp_items_schema, dict):
                    gemini_items_schema = _convert_mcp_prop_to_gemini_schema(mcp_items_schema)
                    if gemini_items_schema:
                        gemini_schema_kwargs["items"] = gemini_items_schema
                    else:
                        logger.warning(f"Could not convert 'items' schema for array type '{mcp_type}'. Skipping property.")
                        return None
                else:
                    logger.warning(f"Array type '{mcp_type}' is missing 'items' schema or it's not a dict. Skipping property.")
                    return None
            elif mcp_type == 'object':
                # --- HANDLE NESTED OBJECTS ---
                gemini_properties_for_object = {}
                mcp_properties_for_object = prop_schema_dict.get('properties', {})
                if isinstance(mcp_properties_for_object, dict):
                    for nested_prop_name, nested_prop_schema_dict in mcp_properties_for_object.items():
                        if isinstance(nested_prop_schema_dict, dict):
                            converted_nested_schema = _convert_mcp_prop_to_gemini_schema(nested_prop_schema_dict)
                            if converted_nested_schema:
                                gemini_properties_for_object[nested_prop_name] = converted_nested_schema
                        else:
                            logger.warning(f"Nested property '{nested_prop_name}' has non-dict schema. Skipping.")

                    gemini_schema_kwargs["properties"] = gemini_properties_for_object
                    gemini_schema_kwargs["required"] = prop_schema_dict.get('required') # Pass through required for objects

                else:
                    logger.warning(f"Object type '{mcp_type}' has malformed 'properties'. Skipping property.")
                    return None


            return genai_types.Schema(**gemini_schema_kwargs)

        for mcp_tool in self.mcp_tools:
            try:
                if hasattr(mcp_tool.inputSchema, 'model_dump'):
                    mcp_schema_dict = mcp_tool.inputSchema.model_dump(
                        exclude_none=True)
                elif isinstance(mcp_tool.inputSchema, dict):
                    mcp_schema_dict = mcp_tool.inputSchema
                else:
                    logger.warning(
                        f"MCP tool '{mcp_tool.name}' has unexpected inputSchema type: {type(mcp_tool.inputSchema)}. Skipping.")
                    continue

                logger.debug(
                    f"Processing MCP tool '{mcp_tool.name}' for Gemini. Schema: {mcp_schema_dict}")

                # Ensure top-level is an object
                if mcp_schema_dict.get('type', '').lower() != 'object':
                    logger.warning(
                        f"MCP tool '{mcp_tool.name}' has non-OBJECT inputSchema ('{mcp_schema_dict.get('type')}'). Skipping for Gemini.")
                    continue

                gemini_properties = {}
                required_props = mcp_schema_dict.get('required', [])
                valid_properties_found = False

                for prop_name, prop_schema_dict in mcp_schema_dict.get('properties', {}).items():
                    if not isinstance(prop_schema_dict, dict):
                        logger.warning(
                            f"Property '{prop_name}' in tool '{mcp_tool.name}' has non-dict schema. Skipping property.")
                        continue

                    # Use the helper function for conversion
                    gemini_prop_schema = _convert_mcp_prop_to_gemini_schema(prop_schema_dict)
                    if gemini_prop_schema:
                        gemini_properties[prop_name] = gemini_prop_schema
                        valid_properties_found = True
                        logger.debug(
                            f"Successfully mapped property '{prop_name}' for tool '{mcp_tool.name}'")
                    else:
                        logger.warning(
                            f"Skipping property '{prop_name}' in tool '{mcp_tool.name}' due to conversion failure.")


                if valid_properties_found or not mcp_schema_dict.get('properties'):
                    gemini_params_schema = genai_types.Schema(
                        type='OBJECT',
                        properties=gemini_properties if gemini_properties else None,
                        required=required_props if required_props and gemini_properties else None
                    )

                    declaration = genai_types.FunctionDeclaration(
                        name=mcp_tool.name,
                        description=mcp_tool.description,
                        parameters=gemini_params_schema,
                    )
                    declarations.append(declaration)
                    logger.info(
                        f"Successfully created Gemini FunctionDeclaration for MCP tool: '{mcp_tool.name}'")
                else:
                    logger.warning(
                        f"Skipping tool '{mcp_tool.name}' for Gemini: No valid properties could be mapped from its OBJECT schema.")

            except Exception as e:
                logger.error(
                    f"Failed to convert MCP tool '{mcp_tool.name}' to Gemini declaration: {e}. Skipping this tool.", exc_info=True)
                continue

        self.cached_gemini_declarations = declarations
        self.gemini_tools_dirty = False
        logger.info(f"Cached {len(declarations)} Gemini tool declarations.")
        return declarations

    async def execute_mcp_tool(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        if tool_name not in self.tool_to_session:
            logger.error(f"Attempted to call unknown MCP tool: {tool_name}")
            return f"Error: Tool '{tool_name}' not found."
        session = self.tool_to_session[tool_name]
        try:
            logger.info(f"Executing MCP tool '{tool_name}' with args: {args}")
            response = await session.call_tool(tool_name, args)
            logger.info(f"MCP tool '{tool_name}' executed successfully.")
            return response.content
        except Exception as e:
            logger.error(f"Error executing MCP tool '{tool_name}': {e}")
            return f"Error executing tool '{tool_name}': {e}"

    async def process_query(self, query: str) -> str:
        if not self.gemini_client:
            return "Error: Gemini client not initialized."

        self.chat_history.append(genai_types.Content(
            role="user", parts=[genai_types.Part(text=query)]))

        gemini_function_declarations = self.get_gemini_tool_declarations()
        gemini_tools = [genai_types.Tool(
            function_declarations=[decl]) for decl in gemini_function_declarations]
        config = genai_types.GenerateContentConfig(
            tools=gemini_tools) if gemini_tools else None

        try:
            response = await self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=self.chat_history,
                config=config,
            )

            if not response.candidates or not response.candidates[0].content:
                feedback = response.prompt_feedback if hasattr(
                    response, 'prompt_feedback') else None
                if feedback and feedback.block_reason:
                    logger.warning(
                        f"Gemini response blocked: {feedback.block_reason}")
                    return f"Response blocked due to: {feedback.block_reason}. {feedback.block_reason_message or ''}"
                return "Error: No response content from Gemini."

            model_content = response.candidates[0].content
            self.chat_history.append(model_content)

            function_calls_to_execute = [
                part.function_call for part in model_content.parts if part.function_call
            ]

            if function_calls_to_execute:
                tool_response_parts = []
                for function_call in function_calls_to_execute:
                    tool_name = function_call.name
                    tool_args = dict(function_call.args)
                    logger.info(
                        f"Gemini requested tool call: {tool_name} with args: {tool_args}")
                    tool_result = await self.execute_mcp_tool(tool_name, tool_args)
                    tool_response_parts.append(genai_types.Part.from_function_response(
                        name=tool_name,
                        response={
                            "result": tool_result if tool_result is not None else "Error executing tool."},
                    ))

                if tool_response_parts:
                    self.chat_history.append(genai_types.Content(
                        role="tool", parts=tool_response_parts))
                    response = await self.gemini_client.models.generate_content(
                        model=self.gemini_model_name,
                        contents=self.chat_history,
                        config=config,
                    )
                    if response.candidates and response.candidates[0].content:
                        final_model_content = response.candidates[0].content
                        self.chat_history.append(final_model_content)
                        return final_model_content.parts[0].text if final_model_content.parts else "Received empty response after tool call."
                    else:
                        return "Error: No response from Gemini after tool execution."
                else:
                    return "Error: Tool calls were requested but no responses could be generated."

            elif model_content.parts and model_content.parts[0].text:
                return model_content.parts[0].text
            else:
                return "Received content without text or function call."

        except genai_errors.APIError as e:
            logger.error(f"Gemini API error: {e}")
            return f"Gemini API Error: {e.message}"
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"An unexpected error occurred: {e}"

    async def chat_loop(self):
        print("\nMCP Gemini Chat App")
        print("Enter your message, 'add_server <path>' to add a server, or 'quit' to exit.")
        while True:
            # Removed the default_prompt logic here
            query = input("\nYou: ").strip() # Always wait for user input

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

            # Removed the exit logic for default_prompt here
            
            # ... rest of chat_loop error handling

    async def cleanup(self):
        logger.info("Cleaning up resources...")
        await self.exit_stack.aclose()
        logger.info("Cleanup complete.")


async def main():
    print("--- APP START: Entering main function ---") # ADDED FOR DEBUGGING
    if len(sys.argv) < 2:
        print(
            "Usage: python mcp_chat_app.py [path_to_mcp_server_script1] [path_to_mcp_server_script2] ...")
        print("You can also add servers at runtime using 'add_server <path>'.")
        server_scripts = []
    else:
        server_scripts = sys.argv[1:]

    app = MCPChatApp()

    try:
        print("--- APP DEBUG: Initializing Gemini client ---") # ADDED FOR DEBUGGING
        await app.initialize_gemini()
        print("--- APP DEBUG: Gemini client initialized ---") # ADDED FOR DEBUGGING

        if server_scripts: # ADDED FOR DEBUGGING
            print(f"--- APP DEBUG: Attempting to connect to initial servers: {server_scripts} ---") # ADDED FOR DEBUGGING
        for script in server_scripts:
            try:
                await app.connect_to_mcp_server(script)
            except Exception as e:
                print(
                    f"Warning: Failed to connect to initial server {script}: {e}")
        print("--- APP DEBUG: Finished initial server connections loop ---") # ADDED FOR DEBUGGING

        if not app.mcp_sessions:
            logger.warning(
                "No MCP servers connected initially. You can add them at runtime using 'add_server <path>'.")
        else:
            logger.info(
                f"Initially connected to {len(app.mcp_sessions)} MCP server(s).")
            gemini_tools = app.get_gemini_tool_declarations()
            if gemini_tools:
                logger.info(
                    f"Initially prepared and cached {len(gemini_tools)} tools for Gemini: {[tool.name for tool in gemini_tools]}")
            else:
                logger.warning(
                    "No MCP tools could be initially prepared for Gemini.")

        print("--- APP DEBUG: Entering chat loop ---") # ADDED FOR DEBUGGING
        await app.chat_loop()
        print("--- APP DEBUG: Exited chat loop ---") # ADDED FOR DEBUGGING

    except ValueError as e:
        print(f"Initialization Error: {e}")
    except Exception as e:
        logger.critical(
            f"Critical error during app execution: {e}", exc_info=True)
        print(f"An unexpected critical error occurred: {e}")
    finally:
        print("--- APP DEBUG: Entering cleanup ---") # ADDED FOR DEBUGGING
        await app.cleanup()
        print("--- APP END: Cleanup complete, exiting ---") # ADDED FOR DEBUGGING

if __name__ == "__main__":
    asyncio.run(main())