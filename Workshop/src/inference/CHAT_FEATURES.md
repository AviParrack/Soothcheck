# Chat History Features

This document describes the conversation history functionality added to the inference system.

## Overview

The system now supports maintaining conversation history across multiple interactions:

- **CLI**: Optional chat mode with `--chat` flag
- **Server**: Always-on session-based conversation history
- **Backward Compatible**: All existing functionality continues to work unchanged

## CLI Usage

### Regular Interactive Mode (No History)
```bash
# Standard interactive mode - each message is independent
python -m scripts.inference gemma-3-27b-it
```

### Chat Mode with History
```bash
# Enable conversation history
python -m scripts.inference gemma-3-27b-it --chat

# With additional options
python -m scripts.inference gemma-3-27b-it --chat --prompt_name friendly_assistant
```

#### Chat Mode Commands
- Type your message normally to continue the conversation
- Type `clear` to reset the conversation history
- Type `exit` or `quit` to quit

#### Example Chat Session
```
=== Language Model Chat Mode (with history) ===
Model type: chat
Model is preloaded. Type 'clear' to reset conversation, 'exit' or 'quit' to quit.

Conversation history: 0 messages
Your prompt: Hello! My name is Alice.

Generating response:
Hello Alice! Nice to meet you. How can I help you today?

Conversation history: 2 messages
Your prompt: What's my name?

Generating response:
Your name is Alice, as you just told me!

Conversation history: 4 messages
Your prompt: clear
Conversation history cleared.

Conversation history: 0 messages
Your prompt: What's my name?

Generating response:
I don't have any information about your name. Could you please tell me?
```

## Server API

### Chat Endpoint with Session Support

**Endpoint**: `POST /chat/stream`

**Request Body**:
```json
{
  "message": "Hello, how are you?",
  "session_id": "user123-session456",
  "max_tokens": 4096,
  "temperature": 0.7,
  "top_p": 0.9,
  "prompt_name": "friendly_assistant"
}
```

**Key Fields**:
- `session_id` (optional): Unique identifier for the conversation session
- If `session_id` is omitted, uses "default" session
- All other fields work as before

**Response**: Server-Sent Events stream (same format as before)

### Reset Session Endpoint

**Endpoint**: `POST /chat/reset`

**Request Body**:
```json
{
  "session_id": "user123-session456"
}
```

**Response**:
```json
{
  "message": "Session 'user123-session456' history cleared"
}
```

### Example API Usage

```python
import aiohttp
import json

# Start a conversation
async def chat_example():
    session_id = "my-chat-session"
    
    # First message
    payload1 = {
        "message": "Hi, I'm working on a Python project",
        "session_id": session_id
    }
    # ... send request and get response ...
    
    # Follow-up message (will include context from first message)
    payload2 = {
        "message": "Can you help me debug it?",
        "session_id": session_id
    }
    # ... send request and get response ...
    
    # Reset conversation
    reset_payload = {"session_id": session_id}
    # POST to /chat/reset
    
    # New conversation (history cleared)
    payload3 = {
        "message": "What were we talking about?",
        "session_id": session_id
    }
    # ... send request and get response ...
```

## Testing

Use the provided test client to verify functionality:

```bash
# Start the server
python -m src.server.main gemma-3-27b-it

# In another terminal, run the test client
python -m src.server.test_chat_client
```

The test client demonstrates:
- Multiple conversation turns with history
- Session management
- History reset functionality
- Backward compatibility

## Implementation Details

### ChatManager Class
- Located in `src/inference/chat_manager.py`
- Handles conversation history as a list of `{"role": "user/assistant", "content": "..."}` messages
- Integrates with existing `TemplateManager` for proper chat formatting
- Supports both "chat" and "text_generation" model types

### Server Session Management
- Sessions stored in memory (`server_state.chat_sessions`)
- Sessions created automatically on first access
- History limited to 50 turns by default (configurable)
- No automatic cleanup (sessions persist until server restart)

### CLI Integration
- New `interactive_chat_mode()` function in `src/inference/inference.py`
- Uses non-streaming generation for simplicity in chat mode
- Maintains conversation state throughout the session

## Backward Compatibility

✅ **All existing functionality continues to work exactly as before**

- Existing CLI usage without `--chat` works unchanged
- Server requests without `session_id` work unchanged
- All existing functions and APIs maintain their original behavior
- No breaking changes to any existing code

## Future Enhancements

Potential improvements (not currently implemented):
- Persistent session storage (Redis, database, etc.)
- Session TTL and automatic cleanup
- Conversation export/import
- Token count management for very long conversations
- Streaming support in CLI chat mode 