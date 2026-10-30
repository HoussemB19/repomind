# RepoMind

RepoMind is a code-question answering service for public GitHub repositories. It clones a repository, extracts Python functions and classes, stores their embeddings in ChromaDB, and uses an OpenRouter-compatible LLM to answer questions with file-and-line citations.

## Features

- Clone and index public GitHub repositories.
- AST-aware Python chunking by function, async function, and class.
- Semantic retrieval with Sentence Transformers and ChromaDB.
- LLM answers through OpenRouter using the OpenAI-compatible API.
- File-and-line citations such as `src/app.py:42-48`.
- Citation range validation and normalization.
- REST API and MCP tools.
- Docker Compose deployment with a separate ChromaDB service.

## Requirements

- Docker Desktop with Docker Compose
- An OpenRouter API key
- A public GitHub repository to index

## Configuration

Create `.env` from `.env.example` and set your API key:

```env
OPENROUTER_API_KEY=your_key_here
```

Never commit `.env` or expose the API key. The repository ignores `.env` by default.

## Run with Docker

From the project directory:

```cmd
docker compose up --build -d
```

Check the API:

```cmd
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok","phase":"3 - API"}
```

Open the interactive FastAPI documentation at:

```text
http://localhost:8000/docs
```

There is no route for `/` in the current API, so opening `http://localhost:8000/` returns `404 Not Found`. This is expected. Use `/health`, `/docs`, `/index`, or `/query` instead.

## FastAPI REST API

The FastAPI service runs in the `api` container and listens on `http://localhost:8000`. Use it from `curl`, Postman, or another HTTP client.

Start the API and Chroma services before testing the endpoints:

```cmd
docker compose up --build -d api chroma
```

### Index a repository

```cmd
curl -X POST http://localhost:8000/index -H "Content-Type: application/json" -d "{\"repo_url\":\"https://github.com/pallets/flask\"}"
```

The response reports the number of scanned Python files and indexed chunks.

For a large repository, indexing can take several minutes. The API logs one message per indexing batch, for example `Indexed 256 chunks` followed by the remaining batch. This is normal.

### Ask a question

```cmd
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"repo_url\":\"https://github.com/pallets/flask\",\"question\":\"How is a Flask application created and configured?\"}"
```

Index the repository again after it changes. Re-indexing clears the previous chunks for that repository before storing the new ones.

## MCP server

The MCP HTTP service is available through the `mcp-http` Compose profile:

```cmd
docker compose --profile mcp-http up --build -d
```

It exposes two tools:

- `index_repo(repo_url)` — clones and indexes a public repository.
- `ask_repo(repo_url, question)` — answers questions about a repository that has already been indexed.

When using an MCP client, call `index_repo` before `ask_repo`. Preserve the returned answer and its citations when presenting it to a user.

### Connect Claude Desktop on Windows

This is the Docker configuration used by this project. It starts the `repomind-api` image in MCP stdio mode, connects it to the Docker network `repomind_net`, loads the local `.env` file, and uses the Chroma service through the hostname `chroma`.

First build the image and start Chroma:

```cmd
docker compose up --build -d api chroma
```

Open the Claude Desktop configuration file:

```text
%APPDATA%\Claude\claude_desktop_config.json
```

Add this entry inside the top-level `mcpServers` object. Change the Windows path if RepoMind is stored elsewhere:

```json
{
  "mcpServers": {
    "repomind-docker": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "--network",
        "repomind_net",
        "--env-file",
        "C:\\path\\to\\repomind\\.env",
        "-e",
        "CHROMA_HOST=chroma",
        "repomind-api",
        "python",
        "app/mcp_server.py"
      ]
    }
  }
}
```

Replace `C:\\path\\to\\repomind\\.env` with the real path on your computer. Do not copy the contents of `.env` into the README, and never commit API keys.

Make sure Docker Desktop is running, then restart Claude Desktop. Claude should show the RepoMind tools `index_repo` and `ask_repo`. Call `index_repo` first with a public GitHub URL, then call `ask_repo` with the same URL and question.

The FastAPI and MCP interfaces are separate:

- FastAPI uses port `8000` and exposes `/health`, `/index`, and `/query`.
- This Claude Desktop configuration uses MCP over Docker stdio and does not need port `8001`.
- The optional Compose HTTP profile exposes Streamable HTTP MCP on port `8001` for clients that support HTTP MCP.

The `repomind-api` image must exist before Claude Desktop launches it. If the source code changes, rebuild it with `docker compose build api` and restart the API/Chroma services.

The Compose HTTP profile is also available for MCP clients that support local Streamable HTTP:

```cmd
docker compose --profile mcp-http up --build -d
```

The Streamable HTTP endpoint is:

```text
http://localhost:8001/mcp
```

Claude Desktop's remote-connector flow is intended for publicly reachable remote servers; for a server running only on your Windows machine, use the local `stdio` configuration above. See Anthropic's [remote MCP guidance](https://support.anthropic.com/en/articles/11503834-building-custom-integrations-via-remote-mcp-servers) for the distinction.

## Local development

Create or activate a Python virtual environment, install dependencies, and run the API:

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Without `CHROMA_HOST`, local development uses the `chroma_data` directory. Docker Compose uses its own named Chroma volume and does not share the local directory automatically.

## Tests

Run the test suite with:

```cmd
pytest -q
```

The tests cover Python chunking, metadata, syntax-error handling, relative paths, and citation normalization/validation.

## Troubleshooting

Check that the containers are running:

```cmd
docker compose ps
```

View the FastAPI logs:

```cmd
docker compose logs --tail=100 api
```

Expected log entries include `Application startup complete`, `GET /health ... 200 OK`, and `POST /index ... 200 OK` or `POST /query ... 200 OK`.

If the source code or Dockerfile changed, rebuild the API image:

```cmd
docker compose up --build -d api chroma
```

## Project layout

```text
app/
  chunking.py       Python AST chunking
  generation.py     Retrieval, prompt construction, LLM calls, citations
  indexing.py       Embeddings and ChromaDB persistence
  ingestion.py      Git cloning and file discovery
  main.py           FastAPI REST service
  mcp_server.py     MCP tools
  search.py         Command-line search helper
tests/              Automated tests
.github/workflows/  GitHub Actions CI
Dockerfile          API/MCP image
docker-compose.yml  API, ChromaDB, and MCP services
```

## Current limitations

- Only public GitHub repositories are supported.
- Only Python files are indexed currently.
- Python code is chunked around functions, async functions, and classes.
- Repositories with no parseable Python chunks cannot answer code questions.
- The Docker deployment uses a named ChromaDB volume; local persistent data is separate.
- The OpenRouter request has a 90-second timeout.

## License

No license file is currently included. Add a license before distributing RepoMind publicly.
