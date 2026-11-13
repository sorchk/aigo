package model

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"

	"github.com/sction-ai/go/internal/config"
	"github.com/sction-ai/go/internal/schema"
)

type ModelHub struct {
	cfg         *config.Config
	mu          sync.RWMutex
	rerank      *MultiModelRegistry[*OnnxRerankService]
	llamaRerank *MultiModelRegistry[*LlamaServerManager]
	llm         *MultiModelRegistry[*LlamaServerManager]
	onnxLLM     *MultiModelRegistry[*OnnxLLMService]
	embed       *MultiModelRegistry[*LlamaServerManager]
	onnxEmbed   *MultiModelRegistry[*OnnxEmbeddingService]
	models      map[string]*config.ModelConfig
	aliases     map[string]string
}

func NewModelHub(cfg *config.Config) *ModelHub {
	h := &ModelHub{
		cfg:     cfg,
		models:  make(map[string]*config.ModelConfig),
		aliases: make(map[string]string),
	}

	for i := range cfg.Models {
		m := &cfg.Models[i]
		if !config.IsEnabled(m.Enabled) {
			continue
		}
		h.models[m.Name] = m
		h.aliases[m.Name] = m.Name
		for _, a := range m.Aliases {
			h.aliases[a] = m.Name
		}
	}

	h.initRerank()
	h.initLLM()
	h.initOnnxLLM()
	h.initEmbedding()

	return h
}

func buildRegistryConfigs(models []config.ModelConfig) []ModelConfig {
	configs := make([]ModelConfig, 0, len(models))
	for _, m := range models {
		configs = append(configs, ModelConfig{
			Name:     m.Name,
			Aliases:  m.Aliases,
			AutoLoad: m.AutoLoad,
		})
	}
	return configs
}

func (h *ModelHub) initRerank() {
	var onnxModels []config.ModelConfig
	var llamaModels []config.ModelConfig

	for _, m := range h.cfg.GetModels("rerank") {
		switch m.Runtime {
		case "onnx":
			onnxModels = append(onnxModels, m)
		default:
			llamaModels = append(llamaModels, m)
		}
	}

	if len(onnxModels) > 0 {
		cfg := h.cfg
		h.rerank = NewMultiModelRegistry[*OnnxRerankService](buildRegistryConfigs(onnxModels), "rerank", func(name string, mc ModelConfig) (*OnnxRerankService, error) {
			modelCfg := cfg.GetModel("rerank", name)
			if modelCfg == nil {
				return nil, fmt.Errorf("rerank model config not found: %s", name)
			}
			return NewOnnxRerankService(*modelCfg, cfg.OnnxRuntimeLib)
		})
	}

	if len(llamaModels) > 0 {
		cfg := h.cfg
		h.llamaRerank = NewMultiModelRegistry[*LlamaServerManager](buildRegistryConfigs(llamaModels), "llama-rerank", func(name string, mc ModelConfig) (*LlamaServerManager, error) {
			modelCfg := cfg.GetModel("rerank", name)
			if modelCfg == nil {
				return nil, fmt.Errorf("rerank model config not found: %s", name)
			}
			mgr := NewLlamaServerManager(*modelCfg, cfg.LlamaServerPath, cfg.Logging)
			if err := mgr.Start(); err != nil {
				return nil, err
			}
			return mgr, nil
		})
	}
}

func (h *ModelHub) initLLM() {
	var models []config.ModelConfig
	for _, m := range h.cfg.GetModels("llm") {
		if m.Runtime != "onnx" {
			models = append(models, m)
		}
	}
	if len(models) == 0 {
		return
	}

	cfg := h.cfg
	h.llm = NewMultiModelRegistry[*LlamaServerManager](buildRegistryConfigs(models), "llm", func(name string, mc ModelConfig) (*LlamaServerManager, error) {
		modelCfg := cfg.GetModel("llm", name)
		if modelCfg == nil {
			return nil, fmt.Errorf("llm model config not found: %s", name)
		}
		mgr := NewLlamaServerManager(*modelCfg, cfg.LlamaServerPath, cfg.Logging)
		if err := mgr.Start(); err != nil {
			return nil, err
		}
		return mgr, nil
	})
}

func (h *ModelHub) initOnnxLLM() {
	var onnxModels []config.ModelConfig
	for _, m := range h.cfg.GetModels("llm") {
		if m.Runtime == "onnx" {
			onnxModels = append(onnxModels, m)
		}
	}
	if len(onnxModels) == 0 {
		return
	}

	cfg := h.cfg
	h.onnxLLM = NewMultiModelRegistry[*OnnxLLMService](buildRegistryConfigs(onnxModels), "onnx-llm", func(name string, mc ModelConfig) (*OnnxLLMService, error) {
		modelCfg := cfg.GetModel("llm", name)
		if modelCfg == nil {
			return nil, fmt.Errorf("onnx llm model config not found: %s", name)
		}
		return NewOnnxLLMService(*modelCfg, cfg.OnnxRuntimeLib)
	})
}

func (h *ModelHub) initEmbedding() {
	var llamaModels []config.ModelConfig
	var onnxModels []config.ModelConfig

	for _, m := range h.cfg.GetModels("embedding") {
		switch m.Runtime {
		case "onnx":
			onnxModels = append(onnxModels, m)
		default:
			llamaModels = append(llamaModels, m)
		}
	}

	if len(llamaModels) > 0 {
		cfg := h.cfg
		h.embed = NewMultiModelRegistry[*LlamaServerManager](buildRegistryConfigs(llamaModels), "embedding", func(name string, mc ModelConfig) (*LlamaServerManager, error) {
			modelCfg := cfg.GetModel("embedding", name)
			if modelCfg == nil {
				return nil, fmt.Errorf("embedding model config not found: %s", name)
			}
			mgr := NewLlamaServerManager(*modelCfg, cfg.LlamaServerPath, cfg.Logging)
			if err := mgr.Start(); err != nil {
				return nil, err
			}
			return mgr, nil
		})
	}

	if len(onnxModels) > 0 {
		cfg := h.cfg
		h.onnxEmbed = NewMultiModelRegistry[*OnnxEmbeddingService](buildRegistryConfigs(onnxModels), "onnx-embedding", func(name string, mc ModelConfig) (*OnnxEmbeddingService, error) {
			modelCfg := cfg.GetModel("embedding", name)
			if modelCfg == nil {
				return nil, fmt.Errorf("onnx embedding model config not found: %s", name)
			}
			return NewOnnxEmbeddingService(*modelCfg, cfg.OnnxRuntimeLib)
		})
	}
}

func (h *ModelHub) GetReranker(modelName string) (*OnnxRerankService, error) {
	if h.rerank == nil {
		return nil, &ValueError{Msg: "No onnx rerank models configured"}
	}
	return h.rerank.GetModel(modelName)
}

func (h *ModelHub) GetLlamaReranker(modelName string) (*LlamaServerManager, error) {
	if h.llamaRerank == nil {
		return nil, &ValueError{Msg: "No llama rerank models configured"}
	}
	return h.llamaRerank.GetModel(modelName)
}

func (h *ModelHub) GetLLM(modelName string) (*LlamaServerManager, error) {
	if h.llm == nil {
		return nil, &ValueError{Msg: "No llama LLM models configured"}
	}
	return h.llm.GetModel(modelName)
}

func (h *ModelHub) GetOnnxLLM(modelName string) (*OnnxLLMService, error) {
	if h.onnxLLM == nil {
		return nil, &ValueError{Msg: "No ONNX LLM models configured"}
	}
	return h.onnxLLM.GetModel(modelName)
}

func (h *ModelHub) GetAnyLLM(modelName string) (interface{}, error) {
	modelCfg := h.GetModelConfig(modelName)
	if modelCfg != nil && modelCfg.Type == "llm" {
		if modelCfg.Runtime == "onnx" {
			return h.GetOnnxLLM(modelName)
		}
		return h.GetLLM(modelName)
	}
	if h.onnxLLM != nil {
		return h.GetOnnxLLM(modelName)
	}
	return h.GetLLM(modelName)
}

func (h *ModelHub) GetEmbedding(modelName string) (*LlamaServerManager, error) {
	if h.embed == nil {
		return nil, &ValueError{Msg: "No llama embedding models configured"}
	}
	return h.embed.GetModel(modelName)
}

func (h *ModelHub) GetONNXEmbedding(modelName string) (*OnnxEmbeddingService, error) {
	if h.onnxEmbed == nil {
		return nil, &ValueError{Msg: "No ONNX embedding models configured"}
	}
	return h.onnxEmbed.GetModel(modelName)
}

func (h *ModelHub) GetDefaultLLM() (*LlamaServerManager, error) {
	if h.llm == nil {
		return nil, &ValueError{Msg: "No llama LLM models configured"}
	}
	return h.llm.GetModel("")
}

func (h *ModelHub) ResolveName(name string) string {
	if name == "" {
		return ""
	}
	return h.aliases[name]
}

func (h *ModelHub) GetModelConfig(name string) *config.ModelConfig {
	resolved := h.aliases[name]
	if resolved == "" {
		return nil
	}
	return h.models[resolved]
}

func (h *ModelHub) ListModels() []schema.ModelInfo {
	var result []schema.ModelInfo

	if h.rerank != nil {
		result = append(result, h.rerank.ListModels("rerank")...)
	}
	if h.llamaRerank != nil {
		result = append(result, h.llamaRerank.ListModels("rerank")...)
	}
	if h.embed != nil {
		result = append(result, h.embed.ListModels("embedding")...)
	}
	if h.onnxEmbed != nil {
		result = append(result, h.onnxEmbed.ListModels("embedding")...)
	}
	if h.llm != nil {
		result = append(result, h.llm.ListModels("llm")...)
	}
	if h.onnxLLM != nil {
		result = append(result, h.onnxLLM.ListModels("llm")...)
	}

	return result
}

func (h *ModelHub) GetStatuses() map[string][]ModelStatus {
	result := make(map[string][]ModelStatus)
	var rerankStatuses []ModelStatus
	if h.rerank != nil {
		rerankStatuses = append(rerankStatuses, h.rerank.GetModelStatuses()...)
	}
	if h.llamaRerank != nil {
		rerankStatuses = append(rerankStatuses, h.llamaRerank.GetModelStatuses()...)
	}
	if len(rerankStatuses) > 0 {
		result["rerank"] = rerankStatuses
	}
	var embedStatuses []ModelStatus
	if h.embed != nil {
		embedStatuses = append(embedStatuses, h.embed.GetModelStatuses()...)
	}
	if h.onnxEmbed != nil {
		embedStatuses = append(embedStatuses, h.onnxEmbed.GetModelStatuses()...)
	}
	if len(embedStatuses) > 0 {
		result["embedding"] = embedStatuses
	}
	var llmStatuses []ModelStatus
	if h.llm != nil {
		llmStatuses = append(llmStatuses, h.llm.GetModelStatuses()...)
	}
	if h.onnxLLM != nil {
		llmStatuses = append(llmStatuses, h.onnxLLM.GetModelStatuses()...)
	}
	if len(llmStatuses) > 0 {
		result["llm"] = llmStatuses
	}
	return result
}

func (h *ModelHub) Embed(modelName string, body io.Reader) (*http.Response, error) {
	mgr, err := h.GetEmbedding(modelName)
	if err != nil {
		return nil, err
	}
	return mgr.ProxyRequest(http.MethodPost, "/v1/embeddings", body)
}

func (h *ModelHub) HasLLM() bool {
	return h.llm != nil || h.onnxLLM != nil
}

func (h *ModelHub) HasRerank() bool {
	return h.rerank != nil || h.llamaRerank != nil
}

func (h *ModelHub) HasEmbedding() bool {
	return h.embed != nil || h.onnxEmbed != nil
}

func (h *ModelHub) Shutdown() {
	log.Println("Shutting down model hub...")
	if h.rerank != nil {
		h.rerank.mu.RLock()
		for _, svc := range h.rerank.models {
			svc.Destroy()
		}
		h.rerank.mu.RUnlock()
	}
	if h.llamaRerank != nil {
		h.llamaRerank.mu.RLock()
		for _, mgr := range h.llamaRerank.models {
			mgr.Shutdown()
		}
		h.llamaRerank.mu.RUnlock()
	}
	if h.onnxEmbed != nil {
		h.onnxEmbed.mu.RLock()
		for _, svc := range h.onnxEmbed.models {
			svc.Destroy()
		}
		h.onnxEmbed.mu.RUnlock()
	}
	if h.onnxLLM != nil {
		h.onnxLLM.mu.RLock()
		for _, svc := range h.onnxLLM.models {
			svc.Destroy()
		}
		h.onnxLLM.mu.RUnlock()
	}
	if h.embed != nil {
		h.embed.mu.RLock()
		for _, mgr := range h.embed.models {
			mgr.Shutdown()
		}
		h.embed.mu.RUnlock()
	}
	if h.llm != nil {
		h.llm.mu.RLock()
		for _, mgr := range h.llm.models {
			mgr.Shutdown()
		}
		h.llm.mu.RUnlock()
	}
	log.Println("Model hub shutdown complete")
}
