# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Any
import httpx, os, time

app = FastAPI(title="Neural Brain API v5", version="5.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MINIMAX_KEY = os.environ.get("MINIMAX_API_KEY", "")
OLLAMA_ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")
MINIMAX_BASE = "https://api.minimax.io/anthropic"

MINIMAX_MODELS = ["MiniMax-M2.5","MiniMax-M2.5-Lightning","MiniMax-VL-01","MiniMax-M2.1","MiniMax-M2.1-lightning"]
OLLAMA_MODELS = ["qwen3:8b","qwen3:4b","llama3.3:latest","llama3.2:3b","deepseek-r1:8b","qwen2.5-coder:7b","mistral-small:latest","phi4:latest","gemma3:12b","llava:latest"]

@app.get("/health")
async def health():
    return {"status":"ok","service":"neural-brain-api","version":"5.0.0","port":8200}

@app.get("/api/v1/models")
async def list_models():
    models = []
    for m in MINIMAX_MODELS:
        models.append({"id":f"minimax/{m}","object":"model","owned_by":"minimax","context_length":200000})
    for m in OLLAMA_MODELS:
        models.append({"id":m,"object":"model","owned_by":"ollama","context_length":128000})
    return {"object":"list","data":models}

class Message(BaseModel):
    role: str
    content: Any

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    system: Optional[str] = None

@app.post("/api/v1/chat/completions")
async def chat(req: ChatRequest):
    model = req.model
    if model.startswith("minimax/") or model in MINIMAX_MODELS:
        return await _minimax(model.replace("minimax/",""), req)
    return await _ollama(model, req)

async def _minimax(model: str, req: ChatRequest):
    if not MINIMAX_KEY:
        raise HTTPException(503, "MINIMAX_API_KEY not set")
    msgs = [{"role":m.role,"content":m.content} for m in req.messages]
    sys_msgs = [m for m in msgs if m["role"]=="system"]
    user_msgs = [m for m in msgs if m["role"]!="system"]
    system_text = sys_msgs[0]["content"] if sys_msgs else (req.system or "")
    if user_msgs and user_msgs[0]["role"] != "user":
        user_msgs.insert(0, {"role":"user","content":"."})
    payload = {"model":model,"system":system_text,"messages":user_msgs,"max_tokens":req.max_tokens or 4096,"temperature":req.temperature or 0.7}
    headers = {"x-api-key":MINIMAX_KEY,"Content-Type":"application/json","anthropic-version":"2023-06-01"}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{MINIMAX_BASE}/v1/messages", json=payload, headers=headers)
        if r.status_code != 200:
            raise HTTPException(r.status_code, r.text)
        data = r.json()
    blocks = data.get("content",[])
    text = next((b["text"] for b in blocks if b.get("type")=="text"), "")
    return {"id":data.get("id","mm-"+str(int(time.time()))),"object":"chat.completion","model":f"minimax/{model}","choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":data.get("stop_reason","stop")}],"usage":data.get("usage",{})}

async def _ollama(model: str, req: ChatRequest):
    msgs = [{"role":m.role,"content":m.content} for m in req.messages]
    payload = {"model":model,"messages":msgs,"stream":False,"options":{"temperature":req.temperature or 0.7,"num_predict":req.max_tokens or 4096}}
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(f"{OLLAMA_ENDPOINT}/api/chat", json=payload)
            if r.status_code != 200:
                raise HTTPException(r.status_code, f"Ollama error: {r.text}")
            data = r.json()
        except httpx.ConnectError:
            raise HTTPException(503, "Ollama not running on localhost:11434")
    text = data.get("message",{}).get("content","")
    return {"id":"ol-"+str(int(time.time())),"object":"chat.completion","model":model,"choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}

@app.post("/api/v1/tts")
async def tts(request: Request):
    data = await request.json()
    text = data.get("text","")
    voice = data.get("voice","Friendly_Person")
    if not MINIMAX_KEY:
        raise HTTPException(503, "MINIMAX_API_KEY not set")
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post("https://api.minimax.io/v1/t2a_v2",
            json={"model":"speech-02-hd","text":text,"voice_setting":{"voice_id":voice,"speed":1.0,"vol":1.0,"pitch":0},"audio_setting":{"sample_rate":32000,"bitrate":128000,"format":"mp3"}},
            headers={"Authorization":f"Bearer {MINIMAX_KEY}","Content-Type":"application/json"})
        if r.status_code != 200:
            raise HTTPException(r.status_code, "TTS failed")
        result = r.json()
    audio_hex = result.get("data",{}).get("audio","")
    if not audio_hex:
        raise HTTPException(500, "No audio in response")
    import base64
    return JSONResponse({"audio_base64":base64.b64encode(bytes.fromhex(audio_hex)).decode()})

@app.get("/ollama/tags")
async def ollama_tags():
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.get(f"{OLLAMA_ENDPOINT}/api/tags")
            return r.json()
        except:
            return {"models":[]}

if __name__ == "__main__":
    import uvicorn
    print("Neural Brain API v5 starting on port 8200...")
    uvicorn.run(app, host="0.0.0.0", port=8200)