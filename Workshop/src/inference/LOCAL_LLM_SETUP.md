# Local LLM Inference Setup

This system enables using local Llama 3.3 70B Instruct for data augmentation instead of Gemini API calls. It's designed to maintain compatibility with existing augmentation code while providing better cost control and privacy.

## Architecture Overview

The system consists of three main components:

1. **Queue Manager Server** (`src/inference/queue_manager.py`) - FastAPI server that loads and serves a local model
2. **Local LLM Adapter** (`src/inference/local_llm_adapter.py`) - HTTP client that mimics litellm.completion interface  
3. **Updated Augmentation Pipeline** - Supports both Gemini and local models via `--model` parameter
4. **Enhanced TemplateManager** - Centralized template management with function calling support

### Key Improvements

- **Consistent Template Handling**: The queue manager now uses `TemplateManager.format_for_model()` for all formatting, ensuring Llama 3.3 template fixes are applied consistently for both regular chat and function calling
- **Centralized Function Calling**: Tool definitions are processed through the same template system used for training and inference
- **Robust Llama Support**: Custom Llama templates avoid default system messages and handle termination tokens properly
- **Accurate Function Call Parsing**: Properly handles Llama 3.2/3.3's pythonic function call format: `[func(param=value)]`

### Function Call Format

Llama 3.2/3.3 outputs function calls in pythonic format:

```python
# Single function call
[get_weather(location="New York")]

# Multiple function calls  
[get_weather(location="Seattle"), search_web(query="Python tutorial", max_results=5)]

# Complex parameters with quotes and commas
[search_web(query="Python, machine learning, and AI")]
```

The parser correctly handles:
- Single and multiple function calls
- Quoted and unquoted string parameters
- Multiple parameters per function
- Complex strings containing commas and special characters
- Validation against available tool definitions

## Quick Start

### 1. Start the Local Inference Server

First, start the queue manager with your desired model:

```bash
# Start Llama 3.3 70B Instruct server on port 8001
python -m src.inference.queue_manager llama-3.3-70b-instruct --port 8001

# The server will:
# - Load the model (takes ~2 minutes for 70B)
# - Start accepting requests on http://127.0.0.1:8001
# - Provide health checks at /health
```

### 2. Run Augmentations with Local Model

Once the server is running, use the `--model local` parameter:

```bash
# Run QA augmentation with local Llama instead of Gemini
python -m scripts.augment_dataset general adam --augmentation qa --model local

# Run all augmentations with local model
python -m scripts.augment_dataset general adam --model local
```

### 3. Health Check

Verify the server is running:

```bash
curl http://127.0.0.1:8001/health
```

## Features

### Function Calling Support

The system supports Llama 3.3's native function calling via `apply_chat_template`:

- Automatically converts litellm tool definitions to Llama 3.3 format
- Parses both JSON and function-style responses
- Maintains compatibility with existing augmentation tool schemas

### Robust Error Handling

- Automatic retries with exponential backoff
- Comprehensive logging for debugging
- Graceful degradation on failures

### Performance Benefits

- **No Model Reloading**: Server keeps model loaded between requests
- **Concurrent Requests**: Multiple augmentation processes can share the same server
- **Cost Effective**: No API costs after initial setup

## Configuration

### Server Configuration

```bash
python -m src.inference.queue_manager llama-3.3-70b-instruct \
  --host 127.0.0.1 \
  --port 8001 \
  --device cuda
```

### Client Configuration

```python
from src.inference.local_llm_adapter import configure_local_llm

# Configure connection settings
configure_local_llm(
    host="127.0.0.1",
    port=8001,
    timeout=300,  # 5 minutes
    max_retries=3
)
```

## Usage Examples

### Basic QA Augmentation

```bash
# Start server
python -m src.inference.queue_manager llama-3.3-70b-instruct &

# Wait for model to load, then run augmentation
python -m scripts.augment_dataset general adam --augmentation qa --model local
```

### Multiple Augmentations

```bash
# Run all configured augmentations with local model
python -m scripts.augment_dataset general adam --model local
```

### Mixed Usage

```bash
# Some augmentations with Gemini (default)
python -m scripts.augment_dataset general adam --augmentation takes

# Others with local model
python -m scripts.augment_dataset general adam --augmentation qa --model local
```

## Troubleshooting

### Server Not Starting

1. Check GPU memory: Llama 3.3 70B requires ~40GB VRAM
2. Verify model specification in configs
3. Check logs for CUDA/memory errors

### Connection Errors

1. Verify server is running: `curl http://127.0.0.1:8001/health`
2. Check firewall settings
3. Ensure port 8001 is not in use

### Function Calling Issues

1. Check server logs for parsing errors
2. Verify tool definitions match expected format
3. Test with simple prompts first

### Performance Issues

1. Monitor GPU utilization
2. Check for memory leaks in long-running servers
3. Consider reducing max_new_tokens for faster responses

## Advanced Configuration

### Custom Models

```bash
# Use different model
python -m src.inference.queue_manager your-custom-model --port 8001
```

### Multiple Servers

```bash
# Start multiple servers on different ports for load balancing
python -m src.inference.queue_manager llama-3.3-70b-instruct --port 8001 &
python -m src.inference.queue_manager llama-3.3-70b-instruct --port 8002 &
```

### Production Deployment

For production use, consider:

- Using a process manager (systemd, supervisor)
- Setting up load balancing
- Monitoring GPU memory and temperature
- Implementing request queuing for high load

## Comparison: Gemini vs Local

| Feature | Gemini API | Local Llama 3.3 |
|---------|------------|------------------|
| Cost | Per request | Hardware only |
| Privacy | Data sent to Google | Fully local |
| Speed | Network dependent | GPU dependent |
| Function Calling | Native support | Via chat templates |
| Setup | API key only | Model + server setup |
| Reliability | Google's infrastructure | Your infrastructure |

## Future Improvements

Planned enhancements:

1. **Batching Support**: Process multiple requests simultaneously
2. **Auto-scaling**: Start/stop servers based on demand  
3. **Model Caching**: Faster switching between different models
4. **Load Balancing**: Distribute requests across multiple GPUs
5. **Monitoring**: Metrics and alerting for production use

This system provides a solid foundation for local LLM inference while maintaining the flexibility to use both local and API-based models as needed. 