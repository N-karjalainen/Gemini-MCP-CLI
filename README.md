```markdown
# MCP Gemini Chat CLI 💬

This is a simple command-line interface (CLI) chat application powered by Google's Gemini model (specifically `gemini-2.0-flash-001` by default). It acts as an MCP (Model Context Protocol) host, allowing it to connect to one or more MCP servers and leverage the tools they provide directly within the chat interface.

## ✨ Features

- **Gemini Powered:** Uses the Gemini API for conversational responses.
- **MCP Host:** Acts as a host for MCP servers.
- **MCP Client Integration:** Includes an MCP client to connect to and interact with MCP servers.
- **Dynamic Tool Discovery:** Automatically discovers and makes tools available to Gemini from any connected MCP server.
- **Multi-Server Support:** Can connect to multiple MCP servers simultaneously at startup.
- **🚀 Runtime Server Addition:** Connect to new MCP servers while the chat app is running using the `add_server` command!
- **Simple CLI:** Easy-to-use command-line interface for interaction.

## 🌊 Flow of Calls

Here's how the application handles user input:

**1. Basic Chat (No Tool Usage):**
```

User Input -> MCP Gemini Chat App (Gemini Client) -> Gemini API -> MCP Gemini Chat App -> Display to User

```

**2. Chat with Tool Usage (e.g., Calculator or Weather):**

```

User Input -> MCP Gemini Chat App (Gemini Client) -> Gemini API (Decides to use a tool)
|
v
MCP Gemini Chat App (MCP Client) -> MCP Server (e.g., Calculator) -> Executes Tool
|
v
MCP Gemini Chat App (MCP Client) <- Tool Result from MCP Server
|
v
MCP Gemini Chat App (Gemini Client) -> Gemini API (with tool result) -> Gemini API (Synthesizes final response)
|
v
MCP Gemini Chat App -> Display to User

````

## 🛠️ Prerequisites

*   Python 3.10+
*   `uv` (or `pip`) for package management
*   A Google Gemini API Key

## ⚙️ Setup

1.  **Clone the repository (or create the project directory):**
    ```bash
    git clone <your-repo-url> # If you have one
    cd mcp-gemini-chat
    # Or if you just have the files:
    # mkdir mcp-gemini-chat
    # cd mcp-gemini-chat
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    uv add mcp google-genai python-dotenv httpx
    ```

4.  **Set up your Gemini API Key:**
    Create a file named `.env` in the project root and add your API key:
    ```dotenv
    # .env
    GEMINI_API_KEY=YOUR_GEMINI_API_KEY
    ```

## ▶️ Running the Application

You run the `mcp_chat_app.py` script. You can optionally provide paths to MCP server scripts you want to connect to at startup as command-line arguments. Servers can also be added while the application is running.

**Example with one server at startup (Weather):**

```bash
python mcp_chat_app.py mcp_server_weather.py
````

**Example with two servers at startup (Weather and Calculator):**

```bash
python mcp_chat_app.py mcp_server_weather.py mcp_server_calc.py
```

**Example starting with no servers (you can add them later):**

```bash
python mcp_chat_app.py
```

The application will:

1.  Initialize the Gemini client.
2.  Attempt to connect to each MCP server script provided at startup (if any).
3.  List the tools discovered from all connected servers.
4.  Start the interactive chat prompt.

## 💡 Example Usage

```
python mcp_chat_app.py mcp_server_calc.py
# ... (Initialization logs) ...
# INFO - Successfully prepared 4 tools for Gemini: ['add', 'subtract', 'multiply', 'divide']

MCP Gemini Chat App
Enter your message, 'add_server <path>' to add a server, or 'quit' to exit.

You: hi
Gemini: Hello! How can I help you today?

You: what is 5 plus 12?
# ... (Logs showing 'add' tool being called) ...
Gemini: 5 plus 12 is 17.0.

You: add_server mcp_server_weather.py
# ... (Logs showing connection attempt) ...
Gemini: Successfully added server 'mcp_server_weather.py' with tools: ['get_alerts', 'get_forecast']

You: any weather alerts in CA?
# ... (Logs showing 'get_alerts' tool being called) ...
Gemini: # ... (Weather alert information or "No active alerts...") ...

You: quit
# ... (Cleanup logs) ...
```

## ➕ Adding More MCP Servers

You can add more tools by connecting to more MCP servers:

1.  **At Startup:** Provide the path to your server script as a command-line argument when starting `mcp_chat_app.py`:

    ```bash
    python mcp_chat_app.py mcp_server_weather.py mcp_server_calc.py my_custom_server.py
    ```

2.  **At Runtime:** Use the `add_server` command within the chat:
    ```
    You: add_server path/to/my_other_server.py
    ```

The chat app will attempt to connect, discover the tools from the new server, and make them available to Gemini for use during the conversation. If a tool with the same name already exists, the new one will be skipped to avoid conflicts.

