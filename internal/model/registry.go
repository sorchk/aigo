package model

import (
	"log"
	"strings"
	"sync"
	"time"

	"github.com/sction-ai/go/internal/schema"
)

const retryCooldownBase = 30 * time.Second
const maxRetryCooldown = 5 * time.Minute

type ModelConfig struct {
	Name     string
	Aliases  []string
	AutoLoad bool
}

type MultiModelRegistry[T any] struct {
	models       map[string]T
	modelConfigs map[string]ModelConfig
	aliasMap     map[string]string
	loaded       map[string]bool
	loading      map[string]bool
	failed       map[string]bool
	failCount    map[string]int
	lastFailTime map[string]time.Time
	mu           sync.RWMutex
	defaultModel string
	loadFn       func(name string, cfg ModelConfig) (T, error)
	serviceName  string
}

func NewMultiModelRegistry[T any](
	configs []ModelConfig,
	serviceName string,
	loadFn func(name string, cfg ModelConfig) (T, error),
) *MultiModelRegistry[T] {
	r := &MultiModelRegistry[T]{
		models:       make(map[string]T),
		modelConfigs: make(map[string]ModelConfig),
		aliasMap:     make(map[string]string),
		loaded:       make(map[string]bool),
		loading:      make(map[string]bool),
		failed:       make(map[string]bool),
		failCount:    make(map[string]int),
		lastFailTime: make(map[string]time.Time),
		loadFn:       loadFn,
		serviceName:  serviceName,
	}

	for _, cfg := range configs {
		r.modelConfigs[cfg.Name] = cfg
		r.aliasMap[cfg.Name] = cfg.Name
		for _, alias := range cfg.Aliases {
			r.aliasMap[alias] = cfg.Name
		}
	}

	if len(configs) > 0 {
		r.defaultModel = configs[0].Name
	}

	log.Printf("Initialized %s registry with %d models", serviceName, len(configs))

	for _, cfg := range configs {
		if cfg.AutoLoad {
			r.StartBackgroundLoad(cfg.Name)
		}
	}

	return r
}

func (r *MultiModelRegistry[T]) ResolveName(name string) string {
	if name == "" {
		return r.defaultModel
	}
	if resolved, ok := r.aliasMap[name]; ok {
		return resolved
	}
	return ""
}

func (r *MultiModelRegistry[T]) GetModel(name string) (T, error) {
	var zero T
	resolved := r.ResolveName(name)
	if resolved == "" {
		return zero, &ValueError{Msg: r.notFoundMsg(name)}
	}

	r.mu.RLock()
	if svc, ok := r.models[resolved]; ok {
		r.mu.RUnlock()
		return svc, nil
	}
	loading := r.loading[resolved]
	r.mu.RUnlock()

	if loading {
		return zero, &schema.ModelNotReadyError{ModelName: resolved}
	}

	if !r.shouldRetry(resolved) {
		return zero, &schema.ModelNotReadyError{ModelName: resolved}
	}

	r.StartBackgroundLoad(resolved)
	return zero, &schema.ModelNotReadyError{ModelName: resolved}
}

func (r *MultiModelRegistry[T]) StartBackgroundLoad(name string) {
	r.mu.Lock()
	if r.loading[name] {
		r.mu.Unlock()
		return
	}
	if !r.shouldRetryLocked(name) {
		r.mu.Unlock()
		return
	}
	r.loading[name] = true
	cfg := r.modelConfigs[name]
	r.mu.Unlock()

	go func() {
		log.Printf("Background loading %s model: %s", r.serviceName, name)
		svc, err := r.loadFn(name, cfg)
		r.mu.Lock()
		defer r.mu.Unlock()
		r.loading[name] = false
		if err != nil {
			r.failed[name] = true
			r.failCount[name]++
			r.lastFailTime[name] = time.Now()
			log.Printf("Failed to load %s model '%s' (attempt %d): %v", r.serviceName, name, r.failCount[name], err)
			return
		}
		r.models[name] = svc
		r.loaded[name] = true
		r.failed[name] = false
		r.failCount[name] = 0
		log.Printf("%s model '%s' loaded successfully", r.serviceName, name)
	}()
}

func (r *MultiModelRegistry[T]) shouldRetry(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.shouldRetryLocked(name)
}

func (r *MultiModelRegistry[T]) shouldRetryLocked(name string) bool {
	if !r.failed[name] {
		return true
	}
	count := r.failCount[name]
	if count <= 0 {
		count = 1
	}
	cooldown := retryCooldownBase * time.Duration(count)
	if cooldown > maxRetryCooldown {
		cooldown = maxRetryCooldown
	}
	lastFail := r.lastFailTime[name]
	if time.Since(lastFail) < cooldown {
		log.Printf("Skipping retry for %s model '%s', cooldown %v remaining (failed %d times)", r.serviceName, name, cooldown-time.Since(lastFail), r.failCount[name])
		return false
	}
	return true
}

func (r *MultiModelRegistry[T]) ListModels(modelType string) []schema.ModelInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()

	now := time.Now().Unix()
	var result []schema.ModelInfo

	for name, cfg := range r.modelConfigs {
		aliases := cfg.Aliases
		names := aliases
		if len(names) == 0 {
			names = []string{name}
		}
		for _, n := range names {
			result = append(result, schema.ModelInfo{
				ID:      n,
				Object:  "model",
				Created: now,
				OwnedBy: "custom",
				Type:    modelType,
				Loaded:  r.loaded[name],
			})
		}
	}
	return result
}

func (r *MultiModelRegistry[T]) notFoundMsg(name string) string {
	available := make([]string, 0, len(r.aliasMap))
	for k := range r.aliasMap {
		available = append(available, k)
	}
	return "Model '" + name + "' not found. Available: " + strings.Join(available, ", ")
}

func (r *MultiModelRegistry[T]) DefaultModelName() string {
	return r.defaultModel
}

type ModelStatus struct {
	Name    string   `json:"name"`
	Status  string   `json:"status"`
	Aliases []string `json:"aliases,omitempty"`
}

func (r *MultiModelRegistry[T]) GetModelStatuses() []ModelStatus {
	r.mu.RLock()
	defer r.mu.RUnlock()

	statuses := make([]ModelStatus, 0, len(r.modelConfigs))
	for name, cfg := range r.modelConfigs {
		var status string
		switch {
		case r.loaded[name]:
			status = "ready"
		case r.loading[name]:
			status = "loading"
		case r.failed[name]:
			status = "failed"
		default:
			status = "not_loaded"
		}
		statuses = append(statuses, ModelStatus{
			Name:    name,
			Status:  status,
			Aliases: cfg.Aliases,
		})
	}
	return statuses
}

type ValueError struct {
	Msg string
}

func (e *ValueError) Error() string {
	return e.Msg
}
