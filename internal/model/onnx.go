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

func appendExecutionProvider(sessionOpts *ort.SessionOptions, cfg config.ModelConfig) error {
	device := cfg.Device
	if device == "" || device == "cpu" {
		return nil
	}

	switch {
	case config.IsCUDADevice(device):
		deviceID := cfg.DeviceID
		if deviceID == 0 {
			deviceID = config.ParseDeviceID(device, 0)
		}
		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err != nil {
			return fmt.Errorf("create CUDA provider options (is onnxruntime built with CUDA?): %w", err)
		}
		cudaOpts.Update(map[string]string{
			"device_id": fmt.Sprintf("%d", deviceID),
		})
		if err = sessionOpts.AppendExecutionProviderCUDA(cudaOpts); err != nil {
			cudaOpts.Destroy()
			return fmt.Errorf("append CUDA execution provider: %w", err)
		}
		cudaOpts.Destroy()
		log.Printf("Using CUDA execution provider (device %d)", deviceID)

	case config.IsTensorRTDevice(device):
		deviceID := cfg.DeviceID
		if deviceID == 0 {
			deviceID = config.ParseDeviceID(device, 0)
		}
		trtOpts, err := ort.NewTensorRTProviderOptions()
		if err != nil {
			return fmt.Errorf("create TensorRT provider options: %w", err)
		}
		trtOpts.Update(map[string]string{
			"device_id": fmt.Sprintf("%d", deviceID),
		})
		if err = sessionOpts.AppendExecutionProviderTensorRT(trtOpts); err != nil {
			trtOpts.Destroy()
			return fmt.Errorf("append TensorRT execution provider: %w", err)
		}
		trtOpts.Destroy()
		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err != nil {
			return fmt.Errorf("create CUDA provider options for TensorRT: %w", err)
		}
		cudaOpts.Update(map[string]string{
			"device_id": fmt.Sprintf("%d", deviceID),
		})
		if err = sessionOpts.AppendExecutionProviderCUDA(cudaOpts); err != nil {
			cudaOpts.Destroy()
			return fmt.Errorf("append CUDA execution provider for TensorRT: %w", err)
		}
		cudaOpts.Destroy()
		log.Printf("Using TensorRT + CUDA execution providers (device %d)", deviceID)

	case config.IsCoreMLDevice(device):
		if err := sessionOpts.AppendExecutionProviderCoreML(0); err != nil {
			return fmt.Errorf("append CoreML execution provider: %w", err)
		}
		log.Printf("Using CoreML execution provider")

	case config.IsDMLDevice(device):
		deviceID := cfg.DeviceID
		if deviceID == 0 {
			deviceID = config.ParseDeviceID(device, 0)
		}
		if err := sessionOpts.AppendExecutionProviderDirectML(deviceID); err != nil {
			return fmt.Errorf("append DirectML execution provider: %w", err)
		}
		log.Printf("Using DirectML execution provider (device %d)", deviceID)

	case config.IsOpenVINODevice(device):
		if err := sessionOpts.AppendExecutionProviderOpenVINO(nil); err != nil {
			return fmt.Errorf("append OpenVINO execution provider: %w", err)
		}
		log.Printf("Using OpenVINO execution provider")

	case config.IsROCmDevice(device):
		log.Printf("Note: ROCm/MIGraphX EP requires onnxruntime built with ROCm support. Using CPU fallback.")
		log.Printf("For ROCm acceleration, use llama-server runtime with HIP backend.")

	case config.IsVulkanDevice(device):
		log.Printf("Note: Vulkan EP requires onnxruntime built with Vulkan support. Using CPU fallback.")
		log.Printf("For Vulkan acceleration, use llama-server runtime with Vulkan backend.")

	default:
		log.Printf("Warning: unknown device '%s', using CPU", device)
	}

	return nil
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
	if config.IsGPUDevice(cfg.Device) {
		var optsErr error
		sessionOpts, optsErr = ort.NewSessionOptions()
		if optsErr != nil {
			return nil, fmt.Errorf("create session options: %w", optsErr)
		}
		if err := appendExecutionProvider(sessionOpts, cfg); err != nil {
			sessionOpts.Destroy()
			return nil, err
		}
	}

	session, err := ort.NewDynamicAdvancedSession(onnxPath, inputNames, outputNames, sessionOpts)
	if err != nil {
		return nil, fmt.Errorf("create onnx session %s: %w", onnxPath, err)
	}

	log.Printf("ONNX session created for model %s from %s (device: %s)", cfg.Name, onnxPath, cfg.Device)

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
