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

	GGUFFile   string `yaml:"gguf_file"`
	MMProjFile string `yaml:"mmproj_file"`
	NGPULayers int    `yaml:"n_gpu_layers"`
	NCtx       int    `yaml:"n_ctx"`
	SocketPath string `yaml:"socket_path"`

	MaxOutputTokens    int                    `yaml:"max_output_tokens"`
	ChatTemplateKwargs map[string]interface{} `yaml:"chat_template_kwargs"`
	EnableThinking     *bool                  `yaml:"enable_thinking"`
}

type Config struct {
	APIToken        string        `yaml:"api_token"`
	OnnxRuntimeLib  string        `yaml:"onnxruntime_lib"`
	LlamaServerPath string        `yaml:"llama_server_path"`
	Models          []ModelConfig `yaml:"models"`
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

		needWrite := true
		if existing, err := os.ReadFile(targetPath); err == nil {
			if string(existing) == string(data) {
				needWrite = false
			}
		}

		if needWrite {
			if err := os.WriteFile(targetPath, data, 0644); err != nil {
				return fmt.Errorf("write %s: %w", targetPath, err)
			}
			log.Printf("Released config file: %s", targetPath)
		}

		return nil
	})
	if err != nil {
		return fmt.Errorf("release embedded configs: %w", err)
	}

	return nil
}
