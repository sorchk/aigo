# AGENTS.md - Go AI Model Hub

## Project Overview

Go service providing OpenAI-compatible `/v1/rerank`, `/v1/embeddings`, `/v1/chat/completions`, and `/v1/translations` endpoints. Unified model configuration with `type` (rerank/embedding/llm) and `runtime` (onnx/llama). Rerank uses ONNX Runtime with goSentencePiece. LLM and embedding (via llama.cpp) proxy to llama-server over Unix sockets. Multi-model support with lazy loading, alias-based routing, optional Bearer token auth.

## Build & Run Commands

```bash
go mod tidy                                          # Install dependencies
go build -o server ./cmd/server                      # Build
go run ./cmd/server                       # Run with defaults
HOST=0.0.0.0 PORT=8000 ./server                      # Custom host/port
air                                                  # Hot-reload development (requires .air.toml)
docker build -t go-rerank-api . && docker run -p 8000:8000 go-rerank-api
```

No test framework, linter, or CI/CD is configured. Format with `go fmt`.

## Directory Structure

```
cmd/server/main.go          # Entry point, Gin router, graceful shutdown
internal/config/config.go   # Unified ModelConfig, YAML config loading, env var helpers
internal/handler/           # Gin handlers (rerank, embedding, proxy, health)
internal/middleware/auth.go  # Bearer token auth middleware
internal/model/             # hub.go (ModelHub), registry.go (generic), reranker.go, llm.go
internal/schema/            # Request/response DTOs and custom error types
config.yaml                 # Unified model list with type + runtime per model
```

## Architecture Patterns

- **ModelHub**: Central model manager in `internal/model/hub.go` — holds registries for each type (rerank, embedding, llm), routes by model name/alias
- **Generic Registry**: `MultiModelRegistry[T any]` in `internal/model/registry.go` — alias-based routing, lazy/background loading, `sync.RWMutex` thread safety
- **Handler Factory**: Handlers are exported functions returning `gin.HandlerFunc` closures, receiving `*model.ModelHub`
- **Reverse Proxy (LLM)**: All non-custom `/v1/*` paths proxy to default llama-server via Unix socket
- **Error Dispatch**: `handleModelErrors()` in `handler/errors.go` classifies `ModelNotReadyError` → 503, `ValueError` → 404, else → 500

## API Endpoints

All routes under `/v1` with auth middleware: `POST /rerank`, `POST /embeddings`, `GET /health`, `GET /models`. All other `/v1/*` routes transparently proxy to default LLM. Non-`/v1` paths (Web UI) proxy without auth.

## Code Style

### Imports

Three groups separated by blank lines: stdlib → third-party → internal. Use alias `ort` for `onnxruntime_go`.

```go
import (
    "context"
    "fmt"

    "github.com/gin-gonic/gin"
    ort "github.com/yalue/onnxruntime_go"

    "github.com/sction-ai/go/internal/config"
    "github.com/sction-ai/go/internal/model"
)
```

### Naming Conventions

| Element | Style | Examples |
|---|---|---|
| Packages | lowercase single word | `config`, `handler`, `model`, `schema` |
| Structs | PascalCase | `ModelHub`, `RerankerService`, `ModelConfig` |
| Exported functions | PascalCase verb-noun | `NewModelHub`, `CreateEmbeddings`, `NewRerankerService` |
| Private functions | camelCase | `handleModelErrors`, `initOnnxRuntime`, `encodeBatch` |
| Variables | camelCase | `modelName`, `batchSize`, `socketPath` |
| Receivers | Short first letter | `(s *RerankerService)`, `(h *ModelHub)`, `(r *MultiModelRegistry[T])` |
| Custom errors | PascalCase + `Error` suffix | `ModelNotReadyError`, `ValueError` |

### Error Handling

Wrap with `fmt.Errorf("<verb> <noun>: %w", err)`. Custom error types for domain errors:
- `schema.ModelNotReadyError` — model still loading (HTTP 503 + Retry-After)
- `model.ValueError` — model name/alias not found (HTTP 404)

ONNX tensors: manually call `Destroy()` on error paths; no `defer` for tensors (early release).

### Type Patterns

- **Optional JSON fields**: pointer types with `json:",omitempty"` (`*string`, `*int`, `*bool`)
- **Generics**: `MultiModelRegistry[T any]` is the only generic type
- **Constructor pattern**: `NewXxx(...) (*Xxx, error)` — returns pointer and error
- **Inference return triple**: `(result, totalTokens int, error)` for rerank methods

### Structured Logging

Standard `log` package only: `log.Printf("Starting server on %s:%d", host, port)`.

## Configuration

YAML-based (`config.yaml`) with env var overrides: `CONFIG_PATH` (default `config.yaml`), `API_TOKEN`, `HOST` (`0.0.0.0`), `PORT` (`8000`). Helper functions `GetEnvString()`/`GetEnvInt()` in `internal/config/config.go`.

Top-level: `onnxruntime_lib`, `llama_server_path`. Models is a flat list where each entry has `type` (rerank/embedding/llm), `runtime` (onnx/llama), plus domain-specific fields.

## Key Dependencies

`gin-gonic/gin` (HTTP), `goccy/go-yaml` (YAML), `yalue/onnxruntime_go` v1.12.1 (ONNX inference, API v19), `tggo/goSentencePiece` (tokenizer), `google/uuid` (response IDs).

## Development Notes

- Tests: stdlib-only `*_test.go` files exist for config parsing and llama-server log controls (`internal/config/config_log_test.go`, `internal/model/llm_support_test.go`); no third-party test deps in `go.mod`. Run `go test ./...` inside a CGO-enabled environment (`internal/model` pulls in `onnxruntime_go`).
- No linting tools configured — run `go fmt` before commits
- `gin.ReleaseMode` set explicitly; `gin.New()` + `gin.Recovery()` (not `gin.Default()`)
- ONNX Runtime: requires v1.12.1 binding (API v19) to match ORT 1.19.2 shared library
- goSentencePiece: pure Go, loads tokenizer.json (Unigram model), thread-safe after creation
