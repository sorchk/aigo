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
	SocketPath string `yaml:"socket_path"`

	MaxOutputTokens    int                    `yaml:"max_output_tokens"`
	ChatTemplateKwargs map[string]interface{} `yaml:"chat_template_kwargs"`
	EnableThinking     *bool                  `yaml:"enable_thinking"`
	Jinja              bool                   `yaml:"jinja"`
	ReasoningParser    string                 `yaml:"reasoning_parser"`
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

	for i := range c.Models {
		m := &c.Models[i]

		if m.Device == "" {
			m.Device = "cpu"
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
