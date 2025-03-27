# MCP Gemini Chat CLI 💬

A simple command-line interface (CLI) chat application powered by Google's Gemini model that acts as an MCP (Model Context Protocol) host.

## ✨ Features

* **🤖 Gemini Powered:** Uses the Gemini API (`gemini-2.0-flash-001` by default) for conversational responses.
* **🔌 MCP Host & Client:** Connects to and leverages tools from multiple MCP servers.
* **🔎 Dynamic Tool Discovery:** Automatically discovers available tools from connected servers.
* **⚡ Runtime Server Management:** Add new MCP servers dynamically while the chat app is running using the `add_server` command.

## 🛠️ Prerequisites

* 🐍 Python 3.10+
* 📦 `uv` (or `pip`) for package management.
* 🔑 Google Gemini API Key.

## 🚀 Quick Start

1.  **Clone & Enter Directory:**
    ```bash
    git clone <your-repo-url> # Replace <your-repo-url> with the actual repo URL
    cd mcp-gemini-chat
    ```

2.  **Create Virtual Environment & Activate:**
    ```bash
    uv venv
    # On Linux/macOS
    source .venv/bin/activate
    # On Windows
    # .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    (Installs `mcp`, `google-genai`, `python-dotenv`, `httpx`)
    ```bash
    uv pip install -r requirements.txt
    ```

4.  **Configure API Key:** ⚠️ **Replace `YOUR_GEMINI_API_KEY` with your actual key!**
    ```bash
    echo "GEMINI_API_KEY=YOUR_GEMINI_API_KEY" > .env
    ```

5.  **Run the App:** (Optionally specify MCP server scripts to connect on startup)
    ```bash
    # Example assuming the app script is in a 'src' directory
    python src/mcp_chat_app.py [path/to/server1.py] [path/to/server2.py]

    # Example without servers at startup
    # python src/mcp_chat_app.py
    ```

## 🌊 Flow of Calls

### Basic Chat Flow (No Tools)
```
👤 User Input → 🤖 MCP Gemini Chat App → ✨ Gemini API → 🖥️ Display to User
```

### Tool Usage Flow (e.g., Calculator, Weather)
1.  👤 User Input → 🤖 MCP Gemini Chat App → ✨ Gemini API
2.  ✨ Gemini decides to use a tool → 🔌 MCP Client → ⚙️ MCP Server (e.g., Calculator Server)
3.  ⚙️ Server executes the tool → 📄 Returns result → 🔌 MCP Client
4.  📄 Result → ✨ Gemini API → 🤔 Synthesizes final response
5.  🗣️ Final response → 🖥️ Display to User

## ▶️ Usage Guide

### Running the Application

Start the application script, optionally providing paths to MCP server scripts you want to connect to immediately.

```bash
# Start without any initial servers
python src/mcp_chat_app.py

# Start and connect to specific servers
python src/mcp_chat_app.py mcp_server_weather.py mcp_server_calc.py
```

### Adding MCP Servers

You can add more tools by connecting to more MCP servers:

**1. At Startup:**
Provide the paths to your server scripts as command-line arguments when starting `mcp_chat_app.py`:
```bash
python src/mcp_chat_app.py path/to/mcp_server_weather.py path/to/mcp_server_calc.py
```

**2. At Runtime:**
Use the `add_server` command within the chat interface:
```text
You: add_server path/to/my_other_server.py
```
The application will attempt to connect and integrate the tools from the new server.

### Example Session

```text
$ python src/mcp_chat_app.py mcp_server_calc.py
# ... Initialization logs ...
MCP Gemini Chat App
Enter your message, 'add_server <path>' to add a server, or 'quit' to exit.

You: hi
Gemini: Hello! How can I help you today?

You: what is 5 plus 12?
# ... Tool call logs ...
Gemini: 5 plus 12 is 17.0.

You: add_server mcp_server_weather.py
# ... Connection and tool discovery logs ...
Gemini: Successfully added server 'mcp_server_weather.py' with tools: ['get_alerts', 'get_forecast']

You: any weather alerts in London?
# ... Tool call logs ...
Gemini: # ... (Response about weather alerts in London) ...

You: quit
# ... Cleanup logs ...
$
