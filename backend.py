"""
Klarity FastAPI Backend
- Uses Composio (ComposioToolSet) for Gmail OAuth + email fetching (last 5-6 mails)
- Uses ASI1 (asi1.ai) cloud model for text simplification
"""

import json
import os
from pathlib import Path

# Load .env file FIRST before any os.getenv calls
from dotenv import load_dotenv
load_dotenv()

import httpx
import textstat
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from gtts import gTTS
import io
import shutil
import re

# Document parsing (optional dependencies)
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

# -- Composio --
from composio import ComposioToolSet, Action, App

# ── Config (loaded from .env) ──
COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY", "")
COMPOSIO_ENTITY_ID = os.getenv("COMPOSIO_ENTITY_ID", "default")

ASI1_API_KEY = os.getenv("ASI1_API_KEY", "")
ASI1_BASE_URL = "https://api.asi1.ai/v1"
ASI1_MODEL = "asi1"

# ── FastAPI app ──
app = FastAPI(title="Klarity Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Composio toolset (lazy init) ──
_toolset = None


def get_toolset() -> ComposioToolSet:
    global _toolset
    if _toolset is None:
        _toolset = ComposioToolSet(
            api_key=COMPOSIO_API_KEY,
            entity_id=COMPOSIO_ENTITY_ID,
        )
    return _toolset


# ══════════════════════════════════════════
#  Pydantic models
# ══════════════════════════════════════════

class MailFetchRequest(BaseModel):
    email: str | None = None
    imap_server: str | None = None
    password: str | None = None

class SimplifyRequest(BaseModel):
    text: str | None = None
    emails: list[dict] | None = None
    mode: str = "standard"

class AnalyzeRequest(BaseModel):
    text: str

class TTSRequest(BaseModel):
    text: str


# ══════════════════════════════════════════
#  Readability helpers
# ══════════════════════════════════════════

def compute_metrics(text: str) -> dict:
    flesch = round(textstat.flesch_reading_ease(text), 1)
    grade = round(textstat.flesch_kincaid_grade(text), 1)
    fog = round(textstat.gunning_fog(text), 1)
    difficult = textstat.difficult_words(text)
    sentences = text.count('.') + text.count('!') + text.count('?') or 1
    words = len(text.split())
    avg_sent_len = round(words / sentences, 1)
    syllables = textstat.syllable_count(text)
    avg_syl = round(syllables / max(words, 1), 2)
    coleman = round(textstat.coleman_liau_index(text), 1)

    if flesch >= 90:
        label, color = "Very Easy", "#2D8B57"
    elif flesch >= 70:
        label, color = "Easy", "#4AAB6E"
    elif flesch >= 60:
        label, color = "Standard", "#D4A017"
    elif flesch >= 50:
        label, color = "Fairly Difficult", "#E07B39"
    elif flesch >= 30:
        label, color = "Difficult", "#C0392B"
    else:
        label, color = "Very Difficult", "#8B0000"

    return {
        "flesch_reading_ease": flesch,
        "flesch_kincaid_grade": grade,#us school grade level 
        "gunning_fog": fog,
        "difficult_words": difficult,
        "avg_sentence_length": avg_sent_len,
        "avg_syllables_per_word": avg_syl,
        "coleman_liau_index": coleman,
        "difficulty_label": label,
        "difficulty_color": color,
    }


# ══════════════════════════════════════════
#  ASI1 helper (OpenAI-compatible API)
# ══════════════════════════════════════════

async def asi1_generate(prompt: str, system: str = "") -> str:
    """Call ASI1 chat completions API."""
    if not ASI1_API_KEY or ASI1_API_KEY == "your-asi1-api-key-here":
        raise HTTPException(500, "ASI1_API_KEY not set. Get one at https://asi1.ai")

    headers = {
        "Authorization": f"Bearer {ASI1_API_KEY}",
        "Content-Type": "application/json",
    }

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": ASI1_MODEL,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 2048,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            r = await client.post(
                f"{ASI1_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except httpx.ConnectError:
            raise HTTPException(502, "Cannot connect to ASI1 API.")
        except httpx.HTTPStatusError as e:
            raise HTTPException(502, f"ASI1 API error: {e.response.text}")
        except (KeyError, IndexError) as e:
            raise HTTPException(502, f"Unexpected ASI1 response format: {str(e)}")


# ══════════════════════════════════════════
#  Endpoints
# ══════════════════════════════════════════

@app.get("/health")
async def health():
    """Check ASI1 connection status."""
    connected = bool(ASI1_API_KEY) and ASI1_API_KEY != "your-asi1-api-key-here"
    return {
        "ollama": "connected" if connected else "disconnected",  # frontend compat
        "models": [ASI1_MODEL] if connected else [],
        "active_model": ASI1_MODEL,
        "provider": "ASI1 (asi1.ai)",
    }


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """Return readability metrics for the given text."""
    if len(req.text.strip()) < 10:
        raise HTTPException(400, "Text too short to analyze.")
    return compute_metrics(req.text)


@app.post("/simplify")
async def simplify(req: SimplifyRequest):
    """Simplify text or structured emails using ASI1 model."""
    mode_instructions = {
        "standard": "Simplify the text for a general audience. Use clear, straightforward language. Aim for an 8th-grade reading level.",
        "simple": "Simplify the text for someone with limited reading ability. Use very short sentences and common words. Aim for a 5th-grade reading level.",
        "eli5": "Explain this text like I'm 5 years old. Use the simplest words possible, short sentences, and relatable examples.",
    }
    instruction = mode_instructions.get(req.mode, mode_instructions["standard"])

    # Prepare the input for the LLM
    source_map = {}
    if req.emails:
        # Structured email input
        combined_text = "I am summarizing a list of emails. Please provide a combined summary and key points.\n\n"
        for i, m in enumerate(req.emails):
            sender = m.get("sender", "Unknown")
            subject = m.get("subject", "No Subject")
            snippet = m.get("snippet", m.get("text", ""))
            combined_text += f"---\n[Email {i+1}]\nFrom: {sender}\nSubject: {subject}\nContent: {snippet}\n"
            source_map[i+1] = f"{sender} - {subject}"
        
        system_prompt = f"""You are a text simplification assistant. {instruction}
        
        Rules for sources:
        - For each bullet point, you MUST indicate which email it came from using the format [1], [2], etc.
        - Example: "- [1] John sent the budget report."
        
        You MUST also provide a "Word Map" in Mermaid graph format (graph LR).
        The Word Map should show 5-8 key concepts from the emails and how they relate.
        
        Respond with valid JSON:
        {{
          "summary": "...",
          "bullets": ["...", "..."],
          "reading_level": "...",
          "map_mermaid": "graph LR; A[Concept1] --> B[Concept2];"
        }}"""
        text_to_process = combined_text
    else:
        # Plain text input
        text_to_process = req.text.strip() if req.text else ""
        if not text_to_process:
            raise HTTPException(400, "No text or emails provided.")
        
        system_prompt = f"""You are a text simplification assistant. {instruction}
        
        You MUST also provide a "Word Map" in Mermaid graph format (graph LR).
        The Word Map should show 5-8 key concepts from the text and how they relate.
        
        Respond with valid JSON:
        {{
          "summary": "...",
          "bullets": ["...", "..."],
          "reading_level": "...",
          "map_mermaid": "graph LR; A[Concept1] --> B[Concept2];"
        }}"""

    raw = await asi1_generate(text_to_process, system=system_prompt)

    # Parse the JSON response
    try:
        start = raw.find('{')
        end = raw.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found")
        data = json.loads(raw[start:end])
    except:
        data = {"summary": raw[:200].strip(), "bullets": [], "reading_level": "Unknown", "map_mermaid": ""}

    # Post-process bullets to extract source tags
    import re
    final_bullets = []
    source_found_any = False
    bullets_raw = data.get("bullets", [])
    if isinstance(bullets_raw, list):
        for b in bullets_raw:
            if not isinstance(b, str): continue
            source_label = None
            # Look for [1], [2] etc.
            match = re.search(r'\[(\d+)\]', b)
            if match:
                idx_str = match.group(1)
                idx = int(idx_str)
                if idx in source_map:
                    source_label = source_map[idx]
                    source_found_any = True
                    # Clean the bullet text
                    b = re.sub(r'\[\d+\]', '', b).strip()
            
            final_bullets.append({
                "text": b,
                "source": source_label
            })
    
    # Readability computation
    bullet_texts = [str(b.get("text", "")) for b in final_bullets]
    summary_text = str(data.get("summary", ""))
    full_text_for_metrics = summary_text + " " + " ".join(bullet_texts)
    
    # Safely get metrics for original text
    orig_text_sample = str(text_to_process)[:5000]
    original_metrics = compute_metrics(orig_text_sample)
    
    simplified_metrics = compute_metrics(full_text_for_metrics) if len(full_text_for_metrics.strip()) > 20 else original_metrics

    return {
        "summary": data.get("summary", ""),
        "bullets": final_bullets,
        "reading_level": data.get("reading_level", "Unknown"),
        "map_mermaid": data.get("map_mermaid", ""),
        "original_metrics": original_metrics,
        "simplified_metrics": simplified_metrics,
        "has_sources": source_found_any
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Extract text from uploaded PDF, DOCX, or TXT."""
    print(f"📥 Received file: {file.filename} (type: {file.content_type})")
    ext = file.filename.split('.')[-1].lower()
    content = ""

    try:
        if ext == "pdf":
            print("📄 Attempting PDF extraction...")
            if not pypdf: 
                print("❌ pypdf not installed!")
                raise HTTPException(500, "pypdf not installed on server. Please run pip install pypdf")
            reader = pypdf.PdfReader(file.file)
            for page in reader.pages:
                text = page.extract_text()
                if text: content += text + "\n"
            print(f"✅ Extracted {len(content)} chars from PDF")
        elif ext == "docx":
            print("📝 Attempting DOCX extraction...")
            if not docx: 
                print("❌ python-docx not installed!")
                raise HTTPException(500, "python-docx not installed on server. Please run pip install python-docx")
            doc = docx.Document(file.file)
            for para in doc.paragraphs:
                content += para.text + "\n"
            print(f"✅ Extracted {len(content)} chars from DOCX")
        else:
            print("📜 Attempting plain text extraction...")
            data = await file.read()
            content = data.decode("utf-8", errors="ignore")
            print(f"✅ Extracted {len(content)} chars from text file")
    except Exception as e:
        print(f"💥 Extraction error: {str(e)}")
        raise HTTPException(500, f"Extraction error: {str(e)}")

    if not content.strip():
        raise HTTPException(400, "Could not extract any text from this file.")
    
    return {"text": content, "filename": file.filename}


@app.post("/tts")
async def tts(req: TTSRequest):
    """Text-to-speech using gTTS."""
    if not req.text.strip():
        raise HTTPException(400, "No text provided.")
    try:
        tts_obj = gTTS(text=req.text, lang="en", slow=False)
        buf = io.BytesIO()
        tts_obj.write_to_fp(buf)
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(500, f"TTS error: {str(e)}")


# ══════════════════════════════════════════
#  Gmail via Composio (ComposioToolSet API)
# ══════════════════════════════════════════

@app.get("/auth/gmail")
async def auth_gmail():
    """Start Composio Gmail OAuth flow — returns a redirect URL."""
    try:
        toolset = get_toolset()
        # Initiate OAuth connection for Gmail
        connection_request = toolset.initiate_connection(
            app=App.GMAIL,
            entity_id=COMPOSIO_ENTITY_ID,
            redirect_url="http://localhost:8000/auth/gmail/callback",
        )
        return {"url": connection_request.redirectUrl}
    except Exception as e:
        raise HTTPException(500, f"Failed to start Gmail auth: {str(e)}")


@app.get("/auth/gmail/callback")
async def auth_gmail_callback():
    """OAuth callback page — user sees this after authorizing."""
    return HTMLResponse("""
    <html><body style="font-family:sans-serif;text-align:center;padding:60px;">
    <h2>✅ Gmail Connected!</h2>
    <p>You can close this tab and go back to Klarity.</p>
    </body></html>
    """)


@app.get("/auth/gmail/status")
async def auth_gmail_status():
    """Check if Gmail is connected via Composio."""
    try:
        toolset = get_toolset()
        entity = toolset.get_entity(id=COMPOSIO_ENTITY_ID)
        connections = toolset.get_connected_accounts(entity_id=COMPOSIO_ENTITY_ID)
        # Check if any Gmail connection exists
        gmail_connected = any(
            getattr(conn, 'appName', '').lower() == 'gmail' or
            getattr(conn, 'app_name', '').lower() == 'gmail'
            for conn in connections
        )
        return {"connected": gmail_connected}
    except Exception as e:
        print(f"Gmail status check error: {e}")
        return {"connected": False}


@app.post("/mail/fetch")
async def mail_fetch(req: MailFetchRequest = None, email: str = None):
    """Fetch the last 5-6 emails using Composio Gmail."""
    try:
        toolset = get_toolset()

        # Execute the Gmail fetch emails action
        result = toolset.execute_action(
            action=Action.GMAIL_FETCH_EMAILS,
            params={
                "max_results": 6,
                "label_ids": ["INBOX"],
            },
            entity_id=COMPOSIO_ENTITY_ID,
        )

        # Format the emails into a structured list of dicts
        structured_emails = []
        def extract_structured(data):
            if isinstance(data, dict):
                if "data" in data: return extract_structured(data["data"])
                if "response_data" in data: return extract_structured(data["response_data"])
                for key in ["messages", "emails", "results", "threadMessages"]:
                    if key in data and isinstance(data[key], list):
                        for item in data[key][:6]: extract_structured(item)
                        return
                
                if any(k in data for k in ["subject", "Subject", "snippet", "messageText"]):
                    subj = data.get("subject", data.get("Subject", data.get("headers", {}).get("Subject", "(No Subject)")))
                    body = data.get("snippet", data.get("body", data.get("messageText", data.get("Body", ""))))
                    # Truncate body for LLM context
                    if isinstance(body, str) and len(body) > 600: body = body[:600] + "..."
                    send = data.get("from", data.get("sender", data.get("From", 
                           data.get("headers", {}).get("From", "Unknown"))))
                    structured_emails.append({
                        "sender": str(send),
                        "subject": str(subj),
                        "snippet": str(body)
                    })
            elif isinstance(data, list):
                for item in data[:6]: extract_structured(item)

        extract_structured(result)
        
        # Also build the combined string for the UI textarea
        combined = ""
        for i, m in enumerate(structured_emails):
            combined += f"From: {m['sender']}\nSubject: {m['subject']}\n{m['snippet']}\n\n---\n\n"

        return {
            "text": combined.strip(),
            "count": len(structured_emails),
            "emails": structured_emails
        }

    except Exception as e:
        error_msg = str(e)
        if "auth" in error_msg.lower() or "connect" in error_msg.lower() or "not found" in error_msg.lower():
            raise HTTPException(401, f"Gmail not connected. Please connect via Settings → Connect Gmail first. Error: {error_msg}")
        raise HTTPException(500, f"Failed to fetch emails: {error_msg}")


# ══════════════════════════════════════════
#  Main
# ══════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("🚀 Klarity Backend starting...")
    print(f"📧 Gmail: Composio (API key: {COMPOSIO_API_KEY[:8]}...)")
    print(f"🤖 LLM: ASI1 (asi1.ai) → model: {ASI1_MODEL}")
    print("🌐 Server: http://localhost:8000")
    if not ASI1_API_KEY or ASI1_API_KEY == "your-asi1-api-key-here":
        print("⚠️  Set ASI1_API_KEY! Get a key at https://asi1.ai")
    uvicorn.run(app, host="0.0.0.0", port=8000)
