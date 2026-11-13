package config

import (
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/goccy/go-yaml"

	"github.com/sction-ai/go/conf"
)

// DefaultLogVerbosity 是 llama-server 的默认日志级别：3 = INFO。
// 上游默认同样是 INFO，但进程环境里的 LLAMA_ARG_LOG_VERBOSITY 可以把它抬到 TRACE，
// 所以 aigo 会显式下发该级别并清理子进程环境变量。
const DefaultLogVerbosity = 3

type ModelConfig struct {
	Name     string   `yaml:"name"`
	Type     string   `yaml:"type"`
	Runtime  string   `yaml:"runtime"`
	Aliases  []string `yaml:"aliases"`
	AutoLoad bool     `yaml:"auto_load"`
	Enabled  *bool    `yaml:"enabled"`

	OnnxPath       string   `yaml:"onnx_path"`
	OnnxSubdir     string   `yaml:"onnx_subdir"`
	OnnxOutputName string   `yaml:"onnx_output_name"`
	OnnxHiddenSize int      `yaml:"onnx_hidden_size"`
	OnnxFiles      []string `yaml:"-"`
	TokenizerPath  string   `yaml:"tokenizer_path"`
	ModelPath      string   `yaml:"model_path"`
	Device         string   `yaml:"device"`
	DeviceID       int      `yaml:"device_id"`

	GGUFFile   string `yaml:"gguf_file"`
	MMProjFile string `yaml:"mmproj_file"`
	NGPULayers int    `yaml:"n_gpu_layers"`
	NCtx       int    `yaml:"n_ctx"`
	NBatch     int    `yaml:"n_batch"`
	NUbatch    int    `yaml:"n_ubatch"`
	NParallel  int    `yaml:"n_parallel"`
	// KVUnified 控制 llama-server 的统一 KV 池。
	//   nil  : 不传该开关，由服务端默认值决定（n_parallel > 0 时为兼容旧行为仍传 --kv-unified）
	//   true : 传 --kv-unified，所有 slot 共享同一个 KV 池
	//   false: 传 --no-kv-unified，每个 slot 各自持有 n_ctx / n_parallel 的上下文配额
	KVUnified *bool `yaml:"kv_unified"`
	// ExtraArgs 原样透传给 llama-server 的额外参数，例如 ["--flash-attn", "--cache-type-k", "q8_0"]。
	// 仅对 runtime: llama 的模型生效，在内置参数之后追加。
	ExtraArgs  []string `yaml:"extra_args"`
	SocketPath string   `yaml:"socket_path"`

	// LogVerbosity 控制 llama-server 的日志级别（-lv N）：
	//   0=generic 1=error 2=warning 3=info 4=trace 5=debug
	// 不填 = 默认 3(INFO)。TRACE(4)/DEBUG(5) 会把 prompt 缓存等内部状态逐条打印，
	// 长驻服务里会迅速撑爆日志盘，因此默认强制 INFO。
	LogVerbosity *int `yaml:"log_verbosity"`
	// CacheIdleSlots 控制空闲 slot 是否写入 prompt 缓存（--cache-idle-slots / --no-cache-idle-slots）。
	// nil = 不传该开关，沿用 llama-server 默认（开启）。
	// 统一 KV 池（--kv-unified）下服务端会自行清理空闲 slot，关掉它可以避免
	// 缓存条目无限增长（条目只增不减，--cache-ram 的淘汰上限永远触发不了）。
	CacheIdleSlots *bool `yaml:"cache_idle_slots"`
	// CacheRAM 限制 prompt 缓存占用的内存上限（MiB，--cache-ram N）；0 = 不传该开关。
	CacheRAM int `yaml:"cache_ram"`

	MaxOutputTokens    int                    `yaml:"max_output_tokens"`
	ChatTemplateKwargs map[string]interface{} `yaml:"chat_template_kwargs"`
	Jinja              bool                   `yaml:"jinja"`
	ReasoningFormat    string                 `yaml:"reasoning_format"`
	ChatTemplate       string                 `yaml:"chat_template"`
	ChatTemplateFile   string                 `yaml:"chat_template_file"`
	Webui              *bool                  `yaml:"webui"`

	PresencePenalty string `yaml:"presence_penalty"`
	Temperature     string `yaml:"temperature"`
	TopK            string `yaml:"top_k"`
	TopP            string `yaml:"top_p"`
	MinP            string `yaml:"min_p"`
}

func IsCUDADevice(device string) bool {
	return device == "cuda" || strings.HasPrefix(device, "cuda:")
}

func IsROCmDevice(device string) bool {
	return device == "rocm" || device == "hip" || strings.HasPrefix(device, "rocm:") || strings.HasPrefix(device, "hip:")
}

func IsVulkanDevice(device string) bool {
	return device == "vulkan" || strings.HasPrefix(device, "vulkan:")
}

func IsCoreMLDevice(device string) bool {
	return device == "coreml" || device == "mlx"
}

func IsTensorRTDevice(device string) bool {
	return device == "tensorrt" || strings.HasPrefix(device, "tensorrt:")
}

func IsDMLDevice(device string) bool {
	return device == "dml" || strings.HasPrefix(device, "dml:")
}

func IsOpenVINODevice(device string) bool {
	return device == "openvino" || strings.HasPrefix(device, "openvino:")
}

func IsGPUDevice(device string) bool {
	return device != "" && device != "cpu"
}

func ParseDeviceID(device string, defaultID int) int {
	parts := strings.SplitN(device, ":", 2)
	if len(parts) == 2 {
		if id, err := strconv.Atoi(parts[1]); err == nil {
			return id
		}
	}
	return defaultID
}

func DeviceCategory(device string) string {
	switch {
	case IsCUDADevice(device):
		return "cuda"
	case IsROCmDevice(device):
		return "rocm"
	case IsVulkanDevice(device):
		return "vulkan"
	case IsCoreMLDevice(device):
		return "coreml"
	case IsTensorRTDevice(device):
		return "tensorrt"
	case IsDMLDevice(device):
		return "dml"
	case IsOpenVINODevice(device):
		return "openvino"
	default:
		return "cpu"
	}
}

type LoggingConfig struct {
	Enabled    bool   `yaml:"enabled"`
	Path       string `yaml:"path"`
	MaxSizeMB  int    `yaml:"max_size_mb"`
	MaxBackups int    `yaml:"max_backups"`
	MaxAgeDays int    `yaml:"max_age_days"`
	Compress   bool   `yaml:"compress"`
	Sanitize   bool   `yaml:"sanitize"`

	// LlamaServerLog 控制 llama-server 子进程的日志去向：
	//   file    = 写入独立轮转文件（默认，<llama_server_log_dir>/llama-server-<模型名>.log）
	//   stdout  = 继承父进程 stdout/stderr（systemd 下会进 journald）
	//   discard = 直接丢弃
	// 默认走独立文件：子进程的 TRACE 级别日志与业务流量无关，
	// 继承 stdout 会把它们灌进系统日志并撑爆磁盘。
	LlamaServerLog string `yaml:"llama_server_log"`
	// LlamaServerLogDir llama-server 子进程日志目录（默认 logs）。
	LlamaServerLogDir string `yaml:"llama_server_log_dir"`
}

type Config struct {
	APIToken        string         `yaml:"api_token"`
	OnnxRuntimeLib  string         `yaml:"onnxruntime_lib"`
	LlamaServerPath string         `yaml:"llama_server_path"`
	Models          []ModelConfig  `yaml:"models"`
	Logging         *LoggingConfig `yaml:"logging"`
}

func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	if env := os.Getenv("API_TOKEN"); env != "" {
		cfg.APIToken = env
	}
	if env := os.Getenv("ONNXRUNTIME_LIB"); env != "" {
		cfg.OnnxRuntimeLib = env
	}
	if env := os.Getenv("LLAMA_SERVER_PATH"); env != "" {
		cfg.LlamaServerPath = env
	}

	cfg.applyDefaults()
	return &cfg, nil
}

func (c *Config) applyDefaults() {
	if c.Logging == nil {
		c.Logging = &LoggingConfig{
			Enabled:    false,
			Path:       "logs/requests.log",
			MaxSizeMB:  100,
			MaxBackups: 7,
			MaxAgeDays: 30,
			Compress:   true,
			Sanitize:   true,
		}
	}
	if c.Logging.Path == "" {
		c.Logging.Path = "logs/requests.log"
	}
	if c.Logging.MaxSizeMB == 0 {
		c.Logging.MaxSizeMB = 100
	}
	if c.Logging.MaxBackups == 0 {
		c.Logging.MaxBackups = 7
	}
	if c.Logging.MaxAgeDays == 0 {
		c.Logging.MaxAgeDays = 30
	}
	if c.Logging.LlamaServerLog == "" {
		c.Logging.LlamaServerLog = "file"
	}
	if c.Logging.LlamaServerLogDir == "" {
		c.Logging.LlamaServerLogDir = "logs"
	}

	for i := range c.Models {
		m := &c.Models[i]

		// llama-server 日志级别默认锁定 INFO(3)：TRACE 会把 prompt 缓存逐条打印出来，
		// 与业务流量无关但足以在短时间内产生数百 MB 日志。
		if m.LogVerbosity == nil {
			v := DefaultLogVerbosity
			m.LogVerbosity = &v
		} else if *m.LogVerbosity < 0 || *m.LogVerbosity > 5 {
			log.Printf("model %s: log_verbosity %d out of range [0,5], clamped", m.Name, *m.LogVerbosity)
			v := *m.LogVerbosity
			if v < 0 {
				v = 0
			} else {
				v = 5
			}
			m.LogVerbosity = &v
		}

		if m.Device == "" {
			m.Device = "cpu"
		}
		// llama-server requires n_ubatch <= n_batch
		if m.NBatch > 0 && m.NUbatch > m.NBatch {
			m.NUbatch = m.NBatch
		}
		// n_parallel: 0 = 不设置，沿用 llama-server 自动值（4 个 slot + 统一 KV 池）
		if m.NParallel < 0 {
			m.NParallel = 0
		}
		// n_ctx 小于 n_ubatch 时，超长输入会先撞上下文上限而不是物理批上限
		if m.NCtx > 0 && m.NUbatch > 0 && m.NCtx < m.NUbatch {
			log.Printf("model %s: n_ctx (%d) < n_ubatch (%d), long inputs will hit the context limit first", m.Name, m.NCtx, m.NUbatch)
		}
		if m.GGUFFile == "" && strings.HasSuffix(m.ModelPath, ".gguf") {
			m.GGUFFile = m.ModelPath
		}
		if m.TokenizerPath == "" && m.ModelPath != "" {
			m.TokenizerPath = m.ModelPath + "/tokenizer.json"
		}
		if m.OnnxPath == "" && m.ModelPath != "" {
			subdir := m.OnnxSubdir
			onnxDir := filepath.Join(m.ModelPath, subdir)
			files := discoverOnnxFiles(onnxDir)
			if len(files) > 0 {
				m.OnnxFiles = files
				m.OnnxPath = files[0]
			} else {
				m.OnnxPath = filepath.Join(m.ModelPath, subdir, "model.onnx")
			}
		}
		if m.OnnxFiles == nil && m.OnnxPath != "" {
			m.OnnxFiles = []string{m.OnnxPath}
		}
		if m.SocketPath == "" {
			switch m.Type {
			case "llm":
				m.SocketPath = fmt.Sprintf("/tmp/llama-%s.sock", m.Name)
			case "embedding":
				m.SocketPath = fmt.Sprintf("/tmp/embedding-%s.sock", m.Name)
			case "rerank":
				m.SocketPath = fmt.Sprintf("/tmp/rerank-%s.sock", m.Name)
			}
		}
	}
}

func discoverOnnxFiles(dir string) []string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}

	preferred := []string{"model.onnx", "model_fp16.onnx", "model_q4.onnx"}

	var preferredFiles []string
	var otherFiles []string

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasSuffix(name, ".onnx") {
			continue
		}
		fullPath := filepath.Join(dir, name)

		isPreferred := false
		for _, p := range preferred {
			if name == p {
				preferredFiles = append(preferredFiles, fullPath)
				isPreferred = true
				break
			}
		}
		if !isPreferred {
			otherFiles = append(otherFiles, fullPath)
		}
	}

	sort.Strings(otherFiles)
	var result []string
	result = append(result, preferredFiles...)
	result = append(result, otherFiles...)
	return result
}

func (c *Config) GetModels(modelType string) []ModelConfig {
	var result []ModelConfig
	for _, m := range c.Models {
		if !IsEnabled(m.Enabled) {
			continue
		}
		if m.Type == modelType {
			result = append(result, m)
		}
	}
	return result
}

func (c *Config) GetModel(modelType, name string) *ModelConfig {
	for i := range c.Models {
		m := &c.Models[i]
		if !IsEnabled(m.Enabled) {
			continue
		}
		if m.Type != modelType {
			continue
		}
		if m.Name == name {
			return m
		}
		for _, alias := range m.Aliases {
			if alias == name {
				return m
			}
		}
	}
	return nil
}

func (c *Config) HasType(modelType string) bool {
	for _, m := range c.Models {
		if IsEnabled(m.Enabled) && m.Type == modelType {
			return true
		}
	}
	return false
}

func GetEnvString(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func IsEnabled(p *bool) bool {
	return p == nil || *p
}

func GetEnvInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}

func EnsureConfDir(dir string) error {
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("create conf dir: %w", err)
		}
		log.Printf("Created conf directory: %s", dir)
	}

	err := fs.WalkDir(conf.ConfigFiles, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}

		data, err := conf.ConfigFiles.ReadFile(path)
		if err != nil {
			return fmt.Errorf("read embedded file %s: %w", path, err)
		}

		targetPath := filepath.Join(dir, path)
		targetDir := filepath.Dir(targetPath)
		if err := os.MkdirAll(targetDir, 0755); err != nil {
			return fmt.Errorf("create dir %s: %w", targetDir, err)
		}

		if _, err := os.Stat(targetPath); err == nil {
			return nil
		}

		if err := os.WriteFile(targetPath, data, 0644); err != nil {
			return fmt.Errorf("write %s: %w", targetPath, err)
		}
		log.Printf("Released config file: %s", targetPath)

		return nil
	})
	if err != nil {
		return fmt.Errorf("release embedded configs: %w", err)
	}

	return nil
}
