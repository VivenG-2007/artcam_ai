# AI Filter Platform

A production-oriented Python app that turns natural-language prompts into safe
OpenCV filter code, validates it, smoke-tests it, and stores both filters and
generation audit data in MongoDB.

## Architecture

```text
User Prompt
   ->
Groq (LLaMA 3 70B) fast draft
   ->
Validator (AST + safety rules)
   ->
Compiler smoke test
   ->
If fail: OpenRouter DeepSeek Coder repair
   ->
Validator + compiler smoke test
   ->
If fail: Groq strict retry
   ->
Final success or structured error
```

## Storage

MongoDB stores:

- `filters`: saved reusable filter code
- `generation_sessions`: full generation runs with provider/model attempts,
  latency, validation failures, and final outcome

## Environment Variables

```bash
GROQ_API_KEY=...
OPENROUTER_API_KEY=...
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=artcam
SHARE_WEBHOOK_URL=http://localhost:5678/webhook/share-filter
GROQ_MODEL=llama3-70b-8192
GROQ_STRICT_MODEL=llama3-70b-8192
OPENROUTER_DEEPSEEK_MODEL=deepseek/deepseek-coder
AI_PROVIDER_TIMEOUT_SEC=30
```

The UI and API also accept Groq and OpenRouter keys per request. If request
keys are omitted, environment variables are used.

## Install

```bash
pip install -r requirements.txt
```

## Run Gradio UI

```bash
python app.py
```

Open `http://localhost:7860`.

## Run FastAPI

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs`.

## API Notes

- `POST /generate`: runs the full staged generation pipeline and saves the
  session to MongoDB
- `POST /apply`: validates and smoke-tests code before applying it
- `GET /filters`: lists saved MongoDB filters
- `POST /filters`: saves validated filter code
- `DELETE /filters/{name}`: deletes a saved filter
- `POST /filters/{name}/share`: shares a saved filter through the webhook
- `GET /health`: checks API and MongoDB connectivity

## Security Model

- AST validator blocks unsafe imports, dangerous nodes, dunder access, and
  blocked builtins
- Compiler uses restricted globals with no Python builtins
- Smoke test executes generated code on a tiny frame before it is accepted
- Docker sandbox support remains available for isolated image-mode execution

## Main Files

- `ai_generator.py`: staged provider pipeline and retry orchestration
- `validator.py`: AST and rule-based safety validation
- `compiler.py`: restricted compilation and smoke testing
- `database.py`: MongoDB persistence for filters and generation sessions
- `app.py`: Gradio UI
- `api.py`: FastAPI backend
- `realtime.py`: webcam preview engine
- `docker_runner.py`: optional Docker sandbox execution
- `share_service.py`: webhook integration
"# artcam_ai" 
