"""
Klarity — main launcher
Run: python main.py
"""
from backend import app
import uvicorn

if __name__ == "__main__":
    print("🚀 Starting Klarity Backend...")
    print("📧 Gmail: Composio MCP with OAuth")
    print("🤖 LLM: ASI1 (asi1.ai)")
    print("🌐 http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
