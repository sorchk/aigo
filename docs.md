# AIGO - Go AI Model Hub 文档

---

## 一、功能清单 / 能力清单

### 1. 核心能力

| 能力 | 说明 | 接口路径 |
|------|------|----------|
| **文本重排序 (Rerank)** | 基于 ONNX Runtime 的 Cross-Encoder 重排序，支持批量推理 | `POST /v1/rerank` |
| **文本向量化 (Embedding)** | 支持 ONNX Runtime 原生推理和 llama.cpp 代理两种运行时 | `POST /v1/embeddings` |
| **对话补全 (Chat Completions)** | 支持纯文本对话、多模态（图文）、流式/非流式输出 | `POST /v1/chat/completions` |
| **多模态 (Vision)** | 支持 Vision Encoder 图像编码 + LLM 联合推理（图像理解） | 同上，通过 `image_url` 传入 |
| **透明代理** | 未被自定义处理的 `/v1/*` 路径自动代理到默认 llama-server | 所有 `/v1/*` |

### 2. 运行时支持

| 运行时 | 类型 | 说明 |
|--------|------|------|
| `onnx` | rerank / embedding / llm | 基于 ONNX Runtime + goSentencePiece 原生推理 |
| `llama` | embedding / llm | 启动 llama-server 子进程，通过 Unix Socket 代理 |

### 3. 系统特性

| 特性 | 说明 |
|------|------|
| **多模型管理** | 统一 ModelHub 管理 rerank / embedding / llm 三类模型 |
| **别名路由** | 模型支持多个别名，请求时可用任意别名调用 |
| **懒加载** | 模型按需加载，支持 `auto_load` 启动时后台加载 |
| **失败重试** | 指数退避重试策略（30s 基础，最大 5 分钟） |
| **Bearer Token 鉴权** | 可选的 API Token 认证中间件 |
| **SSE 流式输出** | Chat 接口支持 Server-Sent Events 流式响应 |
| **GPU 加速** | ONNX Runtime 支持 CUDA 执行提供者；llama.cpp 支持 GPU 层数配置 |
| **健康检查** | 系统状态 + 所有模型加载状态一览 |
| **模型列表** | OpenAI 兼容的 `/v1/models` 接口 |
| **systemd 服务** | 内置 install/uninstall 命令，一键安装为系统服务 |
| **Docker 支持** | 提供 Dockerfile，支持容器化部署 |
| **优雅关闭** | 监听 SIGINT/SIGTERM，10s 超时优雅关闭 |

### 4. ONNX LLM 推理引擎细节

- 支持 **RWKV-6 混合注意力**架构（linear attention + full attention 层）
- 自动检测并加载 `embed_tokens`、`decoder_model`、`vision_encoder` ONNX 模型文件
- KV Cache 管理：linear attention 层使用 conv + recurrent 状态，full attention 层使用 key/value 状态
- 采样策略：Temperature、Top-K、Top-P

---

## 二、部署配置说明

### 1. 环境要求

| 依赖 | 说明 |
|------|------|
| Go 1.23+ | 编译语言 |
| ONNX Runtime 共享库 | ONNX 模型推理需要（如 `libonnxruntime.so`） |
| llama-server | llama 运行时模型需要（在 PATH 中或通过配置指定） |
| CUDA（可选） | GPU 加速需要 CUDA Toolkit |

### 2. 编译构建

```bash
# 安装依赖
go mod tidy

# 编译
go build -o server ./cmd/server

# 交叉编译（使用 build.sh）
bash build.sh
# 输出到 dist/aigo-linux-arm64 和 dist/aigo-linux-amd64
```

### 3. Docker 部署

```bash
docker build -t aigo .
docker run -p 8000:8000 \
  -v /path/to/models:/datadisk/models/hf \
  -v /path/to/config.yaml:/app/config.yaml \
  aigo
```

### 4. systemd 服务部署

```bash
# 编译
go build -o server ./cmd/server

# 安装为系统服务（需要 sudo）
sudo ./server install
sudo ./server install --host 0.0.0.0 --port 8000 --config /etc/aigo/config.yaml

# 服务管理
sudo systemctl status aigo
sudo systemctl restart aigo
sudo systemctl stop aigo
sudo journalctl -u aigo -f

# 卸载服务
sudo ./server uninstall
```

### 5. 配置文件说明 (config.yaml)

```yaml
# API 认证令牌（留空则不启用认证，也可通过环境变量 API_TOKEN 设置）
api_token: "your-secret-token"

# ONNX Runtime 共享库路径
onnxruntime_lib: "/path/to/libonnxruntime.so"

# llama-server 可执行文件路径（留空则使用 PATH 中的 llama-server）
llama_server_path: ""

# 日志配置（可选，均有默认值）
logging:
  enabled: true               # 是否记录请求日志
  path: "logs/requests.log"   # 请求日志文件
  max_size_mb: 100            # 单文件大小上限
  max_backups: 7              # 保留的历史文件数
  max_age_days: 30            # 历史文件保留天数
  compress: true              # 历史文件 gzip 压缩
  sanitize: true              # 脱敏请求内容
  llama_server_log: "file"    # llama-server 子进程日志去向：file / stdout / discard
  llama_server_log_dir: "logs"  # file 模式下的日志目录

models:
  # === Rerank 模型示例 ===
  - name: bge-reranker-v2-m3
    type: rerank              # 模型类型：rerank / embedding / llm
    runtime: onnx             # 运行时：onnx / llama
    model_path: "/path/to/bge-reranker-v2-m3"  # 模型目录（含 tokenizer.json）
    onnx_subdir: "onnx"       # ONNX 文件子目录
    onnx_output_name: "logits" # ONNX 输出节点名（默认 logits）
    auto_load: true           # 启动时自动加载
    enabled: true             # 是否启用（默认 true）
    aliases: ["rerank", "bge-rerank"]  # 模型别名

  # === Embedding 模型（ONNX 运行时）===
  - name: bge-m3
    type: embedding
    runtime: onnx
    model_path: "/path/to/bge-m3/onnx"
    onnx_subdir: ""
    onnx_output_name: "token_embeddings"  # 输出节点名
    onnx_hidden_size: 1024                # 隐藏层维度
    auto_load: true
    enabled: true

  # === Embedding 模型（llama 运行时）===
  - name: bge-m3-llama
    type: embedding
    runtime: llama
    gguf_file: "/path/to/bge-m3-FP16.gguf"
    n_gpu_layers: 99          # GPU 层数
    auto_load: true
    enabled: true

  # === LLM 模型（llama 运行时）===
  - name: qwen3.5-4b
    type: llm
    runtime: llama
    gguf_file: "/path/to/Qwen3.5-4B-Q4_K_M.gguf"
    mmproj_file: "/path/to/mmproj-F16.gguf"  # 多模态投影文件（可选）
    n_gpu_layers: 50
    n_ctx: 131072             # 上下文窗口大小
    n_parallel: 4             # 并发 slot 数（-np），0 = 服务端自动值
    kv_unified: true          # 统一 KV 池，所有 slot 共享，单请求可用满 n_ctx
    log_verbosity: 3          # llama-server 日志级别（0~5，4=TRACE 会刷爆日志盘）
    cache_idle_slots: false   # 空闲 slot 是否写入 prompt 缓存
    extra_args:               # 额外透传给 llama-server 的参数
      - "--flash-attn"
      - "--cache-type-k"
      - "q8_0"
    auto_load: true
    enabled: true

  # === LLM 模型（ONNX 运行时）===
  - name: qwen3.5-4b-onnx
    type: llm
    runtime: onnx
    model_path: "/path/to/Qwen3.5-4B-ONNX"
    onnx_subdir: "onnx_q4"
    auto_load: true
    enabled: true

  # === ONNX GPU 加速 ===
  - name: model-gpu
    type: llm
    runtime: onnx
    model_path: "/path/to/model"
    device: "cuda"            # 使用 CUDA 加速
    auto_load: true
    enabled: true
```

### 6. 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CONFIG_PATH` | `config.yaml` | 配置文件路径 |
| `API_TOKEN` | - | API 认证令牌（覆盖配置文件） |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8000` | 监听端口 |

### 7. 配置字段参考

#### ModelConfig 通用字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | string | 模型名称（唯一标识） |
| `type` | string | 模型类型：`rerank` / `embedding` / `llm` |
| `runtime` | string | 运行时：`onnx` / `llama` |
| `aliases` | []string | 模型别名列表 |
| `auto_load` | bool | 是否在启动时自动加载 |
| `enabled` | *bool | 是否启用（默认 true，可省略） |
| `device` | string | 推理设备：`cpu`（默认）/ `cuda` |

#### ONNX 模型专用字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `model_path` | string | 模型根目录（含 tokenizer.json） |
| `onnx_subdir` | string | ONNX 文件子目录 |
| `onnx_output_name` | string | 输出节点名（rerank 默认 logits，embedding 需指定） |
| `onnx_hidden_size` | int | 隐藏层维度（embedding 需要） |
| `tokenizer_path` | string | tokenizer.json 路径（默认自动拼接 model_path） |

#### llama 运行时专用字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `gguf_file` | string | GGUF 模型文件路径 |
| `mmproj_file` | string | 多模态投影文件路径（可选） |
| `n_gpu_layers` | int | GPU 加载层数 |
| `n_ctx` | int | 上下文窗口大小 |
| `n_batch` | int | 逻辑批上限（`-b`） |
| `n_ubatch` | int | 物理批上限（`-ub`），自动裁剪到 ≤ `n_batch` |
| `n_parallel` | int | 并发 slot 数（`-np`）；0 = 不设置，沿用服务端自动值 |
| `kv_unified` | *bool | 统一 KV 池：`true` → `--kv-unified`，`false` → `--no-kv-unified`，不填 = 不传该开关，由服务端默认值决定 |
| `log_verbosity` | *int | llama-server 日志级别（`-lv`）：0=generic 1=error 2=warning 3=info（默认）4=trace 5=debug；4 及以上会把整个 prompt 缓存逐条打印出来，日志量与业务流量无关 |
| `cache_idle_slots` | *bool | 空闲 slot 是否写入 prompt 缓存：`true` → `--cache-idle-slots`，`false` → `--no-cache-idle-slots`；不填 = 不传该开关 |
| `cache_ram` | int | prompt 缓存上限 MiB（`--cache-ram`）；0 = 不传该开关 |
| `extra_args` | []string | 原样透传给 llama-server 的额外参数，在内置参数之后追加 |
| `socket_path` | string | Unix Socket 路径（默认自动生成） |

> `--cache-idle-slots` / `--cache-ram` 依赖较新的 llama-server。启动前会先执行 `llama-server --help` 探测参数是否被当前二进制支持，不支持时跳过该参数并打印告警，不会导致服务启动失败。

#### LoggingConfig 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `enabled` | bool | 是否记录请求日志（默认 true） |
| `path` | string | 请求日志文件路径（默认 `logs/requests.log`） |
| `max_size_mb` | int | 单文件大小上限（默认 100 MiB） |
| `max_backups` | int | 保留的历史文件数（默认 7） |
| `max_age_days` | int | 历史文件保留天数（默认 30） |
| `compress` | bool | 历史文件是否 gzip 压缩（默认 true） |
| `sanitize` | bool | 是否脱敏请求内容（默认 true） |
| `llama_server_log` | string | llama-server 子进程日志去向：`file`（默认，写独立轮转文件）/ `stdout`（继承父进程，systemd 下进 journald）/ `discard` |
| `llama_server_log_dir` | string | `file` 模式下的日志目录（默认 `logs`），文件名形如 `llama-server-<模型名>.log` |

#### llama-server 日志量控制

llama-server 的 TRACE 级别（`log_verbosity: 4`）会在每次 prompt 缓存更新时把**整份缓存**逐条打印：

```
- cache state: 12789 prompts, ...
- prompt 0x55f1...: 2048 tokens, checkpoints: 0, 0.000 MiB
```

该输出与业务请求量无关，且缓存条目只增不减（条目占用记为 `0.000 MiB`，`--cache-ram` 的淘汰上限永远触发不了），日志量会持续加速增长。

aigo 的默认行为：

1. 显式下发 `-lv 3`（INFO），并清理子进程环境中的 `LLAMA_ARG_LOG_VERBOSITY`，避免外部环境把级别提到 TRACE；
2. 子进程 stdout/stderr 不再继承父进程，改写入独立轮转文件（`logging.llama_server_log: "file"`），避免灌满 journald / 系统日志盘；
3. `cache_idle_slots: false` 关闭空闲 slot 的 prompt 缓存写入，同时释放常驻缓存占用的内存。

#### 统一 KV 池（`kv_unified`）

| 维度 | `true`（统一 KV 池） | `false`（每 slot 独立） |
|------|------|------|
| KV 显存总量 | 约等于 `n_ctx` 级别 | 约等于 `n_ctx` 级别（总量相同） |
| 单请求最大上下文 | 可达整个 `n_ctx` | `n_ctx / n_parallel` |
| 并发请求 | 所有 slot 共享池子，超总池会挤占 | 各 slot 互不影响 |
| 典型场景 | 长度参差不齐的聊天 / Agent 服务 | 长度可预测的批处理（embedding / rerank） |

未显式配置 `kv_unified` 且 `n_parallel > 0` 时，仍沿用旧行为传 `--kv-unified`。

---

## 三、使用说明

### 1. 启动服务

```bash
# 直接运行（默认加载当前目录 config.yaml）
./server

# 指定配置和端口
CONFIG_PATH=/etc/aigo/config.yaml PORT=9000 ./server

# 查看帮助
./server help
```

### 2. 健康检查

```bash
curl http://localhost:8000/v1/health
```

响应示例：
```json
{
  "status": "ok",
  "system": {
    "cpu_cores": 8,
    "os": "linux",
    "arch": "arm64"
  },
  "models": {
    "rerank": [],
    "embedding": [
      {"name": "bge-m3", "status": "ready"}
    ],
    "llm": [
      {"name": "qwen3.5-4b", "status": "ready"}
    ]
  }
}
```

`status` 可能的值：`ok`（全部就绪）、`degraded`（有模型加载中或失败）

### 3. 查看模型列表

```bash
curl http://localhost:8000/v1/models
```

### 4. 模型状态说明

| 状态 | 说明 |
|------|------|
| `ready` | 模型已加载可用 |
| `loading` | 模型正在加载中 |
| `failed` | 模型加载失败 |
| `not_loaded` | 模型未加载（按需加载） |

### 5. 认证

如果配置了 `api_token`，所有 `/v1/*` 请求需携带 Header：

```
Authorization: Bearer your-secret-token
```

---

## 四、接口调用说明

### 通用说明

- 基础 URL：`http://localhost:8000`
- Content-Type：`application/json`
- 认证：`Authorization: Bearer <token>`（如果配置了 api_token）

---

### 1. POST /v1/rerank — 文本重排序

根据查询对文档列表进行相关性排序。

**请求体：**

```json
{
  "model": "bge-reranker-v2-m3",
  "query": "什么是机器学习？",
  "documents": [
    "机器学习是人工智能的一个分支",
    "今天天气很好",
    "深度学习是机器学习的子集",
    "Python 是一种编程语言"
  ],
  "top_n": 3,
  "return_documents": true
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | string | 否 | 模型名称或别名（留空用默认） |
| `query` | string | 是 | 查询文本 |
| `documents` | []string | 是 | 待排序文档列表 |
| `top_n` | int | 否 | 返回前 N 个结果 |
| `top_k` | int | 否 | 同 top_n（兼容） |
| `return_documents` | bool | 否 | 是否返回文档原文（默认 true） |

**响应示例：**

```json
{
  "id": "rerank-a1b2c3d4",
  "model": "bge-reranker-v2-m3",
  "results": [
    {
      "index": 0,
      "document": {"text": "机器学习是人工智能的一个分支"},
      "relevance_score": 0.98
    },
    {
      "index": 2,
      "document": {"text": "深度学习是机器学习的子集"},
      "relevance_score": 0.92
    },
    {
      "index": 3,
      "document": {"text": "Python 是一种编程语言"},
      "relevance_score": 0.45
    }
  ],
  "usage": {
    "prompt_tokens": 85,
    "completion_tokens": 0,
    "total_tokens": 85
  },
  "meta": {
    "api_version": {"version": "2"},
    "billed_units": {"search_units": 1},
    "tokens": {"input_tokens": 85, "output_tokens": 0}
  }
}
```

**curl 示例：**

```bash
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "model": "bge-reranker-v2-m3",
    "query": "什么是机器学习",
    "documents": ["机器学习是AI的分支", "今天天气好"],
    "top_n": 2
  }'
```

---

### 2. POST /v1/embeddings — 文本向量化

将文本转换为向量表示。

**请求体：**

```json
{
  "model": "bge-m3",
  "input": "这是一段需要向量化的文本"
}
```

或批量输入：

```json
{
  "model": "bge-m3",
  "input": ["第一段文本", "第二段文本", "第三段文本"]
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | string | 否 | 模型名称或别名 |
| `input` | string 或 []string | 是 | 单条文本或文本数组 |
| `encoding_format` | string | 否 | 编码格式 |
| `dimensions` | int | 否 | 向量维度 |

**响应示例：**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0123, -0.0456, 0.0789, ...],
      "index": 0
    }
  ],
  "model": "bge-m3",
  "usage": {
    "prompt_tokens": 12,
    "total_tokens": 12
  }
}
```

**curl 示例：**

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "model": "bge-m3",
    "input": ["你好世界", "Hello World"]
  }'
```

---

### 3. POST /v1/chat/completions — 对话补全

#### 3.1 纯文本对话

**请求体：**

```json
{
  "model": "qwen3.5-4b",
  "messages": [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "请介绍一下 Go 语言的优势"}
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | string | 是 | 模型名称或别名 |
| `messages` | []ChatMessage | 是 | 对话消息列表 |
| `max_tokens` | int | 否 | 最大生成 token 数（默认 512） |
| `temperature` | float | 否 | 采样温度（默认 0.6） |
| `top_p` | float | 否 | Top-P 采样（默认 0） |
| `top_k` | int | 否 | Top-K 采样（默认 0） |
| `stream` | bool | 否 | 是否流式输出（默认 false） |
| `stop` | []string | 否 | 停止词列表 |
| `frequency_penalty` | float | 否 | 频率惩罚 |
| `presence_penalty` | float | 否 | 存在惩罚 |

**ChatMessage 格式：**

```json
{"role": "system" | "user" | "assistant", "content": "文本内容"}
```

**非流式响应示例：**

```json
{
  "id": "chatcmpl-a1b2c3d4",
  "object": "chat.completion",
  "created": 1712200000,
  "model": "qwen3.5-4b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Go 语言有以下优势：\n1. 高性能的并发模型..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 120,
    "total_tokens": 145
  }
}
```

#### 3.2 多模态对话（图文）

**请求体：**

```json
{
  "model": "qwen3.5-4b",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "描述这张图片"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
      ]
    }
  ],
  "max_tokens": 256
}
```

`image_url.url` 支持以下格式：
- Base64 Data URI：`data:image/jpeg;base64,<base64_data>`
- 图片文件路径：`file:///path/to/image.jpg`

> 注意：多模态需要模型配置了 `mmproj_file`（llama 运行时）或包含 `vision_encoder` ONNX 文件（onnx 运行时）。

#### 3.3 流式输出

设置 `"stream": true`，响应为 Server-Sent Events (SSE) 格式：

```
data: {"id":"chatcmpl-a1b2c3d4","object":"chat.completion.chunk","created":1712200000,"model":"qwen3.5-4b","choices":[{"index":0,"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl-a1b2c3d4","object":"chat.completion.chunk","created":1712200000,"model":"qwen3.5-4b","choices":[{"index":0,"delta":{"content":"Go"}}]}

data: {"id":"chatcmpl-a1b2c3d4","object":"chat.completion.chunk","created":1712200000,"model":"qwen3.5-4b","choices":[{"index":0,"delta":{"content":"语言"}}]}

...

data: {"id":"chatcmpl-a1b2c3d4","object":"chat.completion.chunk","created":1712200000,"model":"qwen3.5-4b","choices":[{"index":0,"finish_reason":"stop"}]}

data: {"id":"chatcmpl-a1b2c3d4","object":"chat.completion.chunk","created":1712200000,"model":"qwen3.5-4b","choices":[],"usage":{"prompt_tokens":25,"completion_tokens":120,"total_tokens":145}}

data: [DONE]
```

**curl 流式示例：**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "model": "qwen3.5-4b",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
```

---

### 4. GET /v1/health — 健康检查

```bash
curl http://localhost:8000/v1/health
```

---

### 5. GET /v1/models — 模型列表

```bash
curl http://localhost:8000/v1/models
```

**响应示例：**

```json
{
  "object": "list",
  "data": [
    {
      "id": "bge-reranker-v2-m3",
      "object": "model",
      "created": 1712200000,
      "owned_by": "custom",
      "type": "rerank",
      "loaded": true
    },
    {
      "id": "bge-m3",
      "object": "model",
      "created": 1712200000,
      "owned_by": "custom",
      "type": "embedding",
      "loaded": true
    },
    {
      "id": "qwen3.5-4b",
      "object": "model",
      "created": 1712200000,
      "owned_by": "custom",
      "type": "llm",
      "loaded": true
    }
  ]
}
```

---

### 6. 透明代理

所有未被自定义处理的 `/v1/*` 路径会被代理到默认 llama-server，例如：

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5-4b", "prompt": "Hello"}'
```

非 `/v1` 路径（如 Web UI）也会被代理，但无需认证。

---

### 7. 错误响应

| HTTP 状态码 | 错误类型 | 说明 |
|------------|---------|------|
| 400 | `invalid_request_error` | 请求参数错误 |
| 401 | - | API Token 无效或缺失 |
| 404 | `model_not_found` | 模型名称/别名不存在 |
| 500 | `internal_error` | 服务器内部错误 |
| 503 | `model_not_ready` | 模型正在加载中，附带 `Retry-After: 30` Header |

**503 响应示例：**

```json
{
  "error": "model_not_ready",
  "message": "Model 'qwen3.5-4b' is loading, please retry later"
}
```

---

### 8. Python 调用示例

使用 OpenAI SDK 调用：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-token"  # 如果配置了 api_token
)

# Chat Completions
response = client.chat.completions.create(
    model="qwen3.5-4b",
    messages=[
        {"role": "user", "content": "你好，介绍一下自己"}
    ],
    max_tokens=256
)
print(response.choices[0].message.content)

# Embeddings
response = client.embeddings.create(
    model="bge-m3",
    input="这是一段测试文本"
)
print(response.data[0].embedding[:5])

# 流式 Chat
stream = client.chat.completions.create(
    model="qwen3.5-4b",
    messages=[{"role": "user", "content": "写一首诗"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

Rerank 调用（使用 requests）：

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/rerank",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer your-token"
    },
    json={
        "model": "bge-reranker-v2-m3",
        "query": "什么是深度学习",
        "documents": [
            "深度学习是机器学习的分支",
            "今天天气晴朗",
            "神经网络是深度学习的基础"
        ],
        "top_n": 2
    }
)
print(response.json())
```
