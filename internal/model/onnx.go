package model

import (
	"fmt"
	"log"
	"sync"

	sp "github.com/tggo/goSentencePiece"
	ort "github.com/yalue/onnxruntime_go"

	"github.com/sction-ai/go/internal/config"
)

var onnxInitOnce sync.Once

func initOnnxRuntime(libPath string) error {
	var initErr error
	onnxInitOnce.Do(func() {
		if libPath != "" {
			ort.SetSharedLibraryPath(libPath)
		}
		initErr = ort.InitializeEnvironment()
		if initErr != nil {
			log.Printf("Failed to initialize ONNX Runtime: %v", initErr)
		} else {
			log.Println("ONNX Runtime initialized successfully")
		}
	})
	return initErr
}

type OnnxSession struct {
	session   *ort.DynamicAdvancedSession
	tokenizer *sp.Tokenizer
	modelCfg  config.ModelConfig
	mu        sync.RWMutex
}

func NewOnnxSession(cfg config.ModelConfig, onnxLibPath string, inputNames, outputNames []string) (*OnnxSession, error) {
	if err := initOnnxRuntime(onnxLibPath); err != nil {
		return nil, fmt.Errorf("initialize onnx runtime: %w", err)
	}

	tokenizerPath := cfg.TokenizerPath
	if tokenizerPath == "" && cfg.ModelPath != "" {
		tokenizerPath = cfg.ModelPath + "/tokenizer.json"
	}
	if tokenizerPath == "" {
		return nil, fmt.Errorf("tokenizer path not configured for model %s", cfg.Name)
	}

	tok, err := sp.NewTokenizer(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer %s: %w", tokenizerPath, err)
	}

	onnxPath := cfg.OnnxPath
	if onnxPath == "" {
		return nil, fmt.Errorf("onnx path not configured for model %s", cfg.Name)
	}

	var sessionOpts *ort.SessionOptions
	if cfg.Device == "cuda" || cfg.Device == "cuda:0" {
		var optsErr error
		sessionOpts, optsErr = ort.NewSessionOptions()
		if optsErr != nil {
			return nil, fmt.Errorf("create session options: %w", optsErr)
		}
		cudaOpts, optsErr := ort.NewCUDAProviderOptions()
		if optsErr != nil {
			sessionOpts.Destroy()
			return nil, fmt.Errorf("create CUDA provider options (is onnxruntime built with CUDA?): %w", optsErr)
		}
		if optsErr = sessionOpts.AppendExecutionProviderCUDA(cudaOpts); optsErr != nil {
			cudaOpts.Destroy()
			sessionOpts.Destroy()
			return nil, fmt.Errorf("append CUDA execution provider: %w", optsErr)
		}
		cudaOpts.Destroy()
	}

	session, err := ort.NewDynamicAdvancedSession(onnxPath, inputNames, outputNames, sessionOpts)
	if err != nil {
		return nil, fmt.Errorf("create onnx session %s: %w", onnxPath, err)
	}

	log.Printf("ONNX session created for model %s from %s", cfg.Name, onnxPath)

	return &OnnxSession{
		session:   session,
		tokenizer: tok,
		modelCfg:  cfg,
	}, nil
}

func (s *OnnxSession) Destroy() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.session != nil {
		s.session.Destroy()
	}
}

func (s *OnnxSession) Tokenizer() *sp.Tokenizer {
	return s.tokenizer
}

func (s *OnnxSession) Config() config.ModelConfig {
	return s.modelCfg
}

func (s *OnnxSession) Run(inputs []ort.ArbitraryTensor, outputs []ort.ArbitraryTensor) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.session.Run(inputs, outputs)
}
