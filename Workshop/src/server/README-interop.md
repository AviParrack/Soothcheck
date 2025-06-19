Technical Specification: Frontend-Backend Streaming API
Architecture Overview
Frontend (GitHub Pages): Custom HTML/CSS/JS interface
Backend (Vast.ai): FastAPI server with your ML inference pipeline
Communication: Server-Sent Events (SSE) for streaming + REST for control

API Specification
1. REST Endpoints
POST /chat/stream
Purpose: Initiate streaming inference
Backend Implementation: FastAPI endpoint that returns SSE stream

python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourusername.github.io"],  # Your GitHub Pages domain
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    async def generate():
        # Format prompt using your existing pipeline
        formatted_prompt = load_and_format_prompt_from_utils(
            user_input=request.message,
            tokenizer=tokenizer,
            model_type=model_type,
            prompt_name=request.prompt_name,
        )
        
        # Use your existing streaming function
        async for token in generate_text_yield_tokens(
            prompt=formatted_prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        ):
            # SSE format: data field + double newline
            yield f"data: {json.dumps({'token': token, 'type': 'token'})}\n\n"
        
        # Send completion signal
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"},
    )
GET /health
Purpose: Health check
Response: {"status": "ok", "model_loaded": true}

2. Request/Response Schemas
python
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    prompt_name: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
3. Frontend Implementation
typescript
// Replace your mock API with this
export const callWorkshopAPI = async (
  message: string,
  onToken?: (token: string) => void
): Promise<string> => {
  const response = await fetch(`${API_BASE_URL}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message,
      max_tokens: 4096,
      temperature: 0.7,
      top_p: 0.9,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let fullResponse = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          
          if (data.type === 'token') {
            fullResponse += data.token;
            onToken?.(data.token); // Stream to UI
          } else if (data.type === 'complete') {
            return fullResponse;
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  return fullResponse;
};
CORS Configuration
Backend (FastAPI)
python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourusername.github.io",  # Production
        "http://localhost:3000",           # Development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
Frontend
No special CORS handling needed - browser handles preflight automatically.

Error Handling
Backend Error Responses
python
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": "server_error"}
    )
Frontend Error Handling
typescript
// Add timeout and error handling to your existing pattern
const timeoutPromise = new Promise<never>((_, reject) =>
  setTimeout(() => reject(new Error("timeout")), 30000) // Longer for inference
);

try {
  return await Promise.race([
    callWorkshopAPI(message, onToken),
    timeoutPromise
  ]);
} catch (error) {
  if (error.message === "timeout") {
    throw new Error("Request timed out");
  }
  throw error;
}
Deployment Considerations
Backend (Vast.ai)
bash
# Install dependencies
pip install fastapi uvicorn python-multipart

# Run server
uvicorn main:app --host 0.0.0.0 --port 8000
Environment Variables
python
import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "https://yourusername.github.io").split(",")
Frontend Environment
javascript
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://your-vast-instance.com'
  : 'http://localhost:8000';
Responsibility Boundaries
Backend Responsibilities
Model loading and inference via your existing preload_model_and_tokenizer
Prompt formatting via load_and_format_prompt_from_utils
Token streaming via generate_text_yield_tokens
CORS headers and security
Error handling and logging
Resource management (GPU memory)
Frontend Responsibilities
UI state management
SSE parsing and token accumulation
User input validation
Loading states and progress indicators
Error display and retry logic
Message history management
Integration with Your Existing Code
Minimal changes needed:

Wrap your generate_text_yield_tokens in FastAPI endpoint
Add SSE formatting to token yields
Replace frontend mock API with SSE client
Configure CORS for your GitHub Pages domain
Your existing inference pipeline (preload_model_and_tokenizer, prompt formatting, etc.) can be used as-is.




