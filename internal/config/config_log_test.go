package config

import (
	"path/filepath"
	"testing"

	"github.com/goccy/go-yaml"
)

func TestParseLogFields(t *testing.T) {
	raw := `
onnxruntime_lib: "/x/libonnxruntime.so"
llama_server_path: "/x/llama-server"
logging:
  enabled: true
  path: "logs/requests.log"
  max_size_mb: 100
  max_backups: 7
  max_age_days: 30
  compress: true
  sanitize: true
  llama_server_log: "file"
  llama_server_log_dir: "logs"
models:
  - name: qwen3.5
    type: llm
    runtime: llama
    gguf_file: "/x/q.gguf"
    log_verbosity: 3
    cache_idle_slots: false
    enabled: true
`
	var cfg Config
	if err := yaml.Unmarshal([]byte(raw), &cfg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if cfg.Logging.LlamaServerLog != "file" || cfg.Logging.LlamaServerLogDir != "logs" {
		t.Errorf("logging fields: %+v", cfg.Logging)
	}
	m := cfg.Models[0]
	if m.LogVerbosity == nil || *m.LogVerbosity != 3 {
		t.Errorf("log_verbosity: %v", m.LogVerbosity)
	}
	if m.CacheIdleSlots == nil || *m.CacheIdleSlots != false {
		t.Errorf("cache_idle_slots: %v", m.CacheIdleSlots)
	}
	if m.CacheRAM != 0 {
		t.Errorf("cache_ram: %d", m.CacheRAM)
	}
}

func TestLogDefaults(t *testing.T) {
	var cfg Config
	cfg.Models = []ModelConfig{{Name: "m", Type: "llm", Runtime: "llama"}}
	cfg.applyDefaults()
	m := cfg.Models[0]
	if m.LogVerbosity == nil || *m.LogVerbosity != DefaultLogVerbosity {
		t.Errorf("default log verbosity: %v", m.LogVerbosity)
	}
	if m.CacheIdleSlots != nil {
		t.Errorf("cache_idle_slots should stay unset by default: %v", *m.CacheIdleSlots)
	}
	if cfg.Logging.LlamaServerLog != "file" || cfg.Logging.LlamaServerLogDir != "logs" {
		t.Errorf("logging defaults: %+v", cfg.Logging)
	}
}

func TestClampVerbosity(t *testing.T) {
	bad := 9
	cfg := Config{Models: []ModelConfig{{Name: "m", Type: "llm", Runtime: "llama", LogVerbosity: &bad}}}
	cfg.applyDefaults()
	if v := *cfg.Models[0].LogVerbosity; v < 0 || v > 5 {
		t.Errorf("verbosity not clamped: %d", v)
	}
}

// TestRepoConfigLogSettings 校验仓库自带的 conf/config.yaml 里启用的 llama 模型
// 都已经显式关掉 TRACE 日志和空闲 slot 的 prompt 缓存写入。
func TestRepoConfigLogSettings(t *testing.T) {
	path := filepath.Join("..", "..", "conf", "config.yaml")
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	if cfg.Logging.LlamaServerLog != "file" {
		t.Errorf("llama_server_log: %q", cfg.Logging.LlamaServerLog)
	}
	for _, m := range cfg.Models {
		if m.Runtime != "llama" || !IsEnabled(m.Enabled) {
			continue
		}
		if m.LogVerbosity == nil || *m.LogVerbosity > 3 {
			t.Errorf("model %s: log_verbosity must be <= 3, got %v", m.Name, m.LogVerbosity)
		}
		if m.CacheIdleSlots == nil || *m.CacheIdleSlots {
			t.Errorf("model %s: cache_idle_slots should be false", m.Name)
		}
	}
}
