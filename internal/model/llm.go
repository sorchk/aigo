package model

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/sction-ai/go/internal/config"
)

type LlamaServerManager struct {
	cmd             *exec.Cmd
	client          *http.Client
	socketPath      string
	mu              sync.Mutex
	ready           bool
	cancelFn        context.CancelFunc
	modelCfg        config.ModelConfig
	serverBinary    string
	stopped         bool
	done            chan struct{}
	serverModelName string
}

func NewLlamaServerManager(cfg config.ModelConfig, serverBinary string) *LlamaServerManager {
	socketPath := cfg.SocketPath
	if socketPath == "" {
		socketPath = fmt.Sprintf("/tmp/llama-%s.sock", cfg.Name)
	}

	if serverBinary == "" {
		serverBinary = "llama-server"
	}

	return &LlamaServerManager{
		socketPath:   socketPath,
		modelCfg:     cfg,
		serverBinary: serverBinary,
		client: &http.Client{
			Transport: &http.Transport{
				DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
					return net.DialTimeout("unix", socketPath, 5*time.Second)
				},
			},
			Timeout: 300 * time.Second,
		},
	}
}

// buildArgs 组装 llama-server 的命令行参数
func (m *LlamaServerManager) buildArgs() []string {
	args := []string{
		"-m", m.modelCfg.GGUFFile,
		"--host", m.socketPath,
	}

	if m.modelCfg.NGPULayers > 0 {
		args = append(args, "-ngl", strconv.Itoa(m.modelCfg.NGPULayers))
	}

	device := m.modelCfg.Device
	if device == "" {
		device = "cpu"
	}

	switch config.DeviceCategory(device) {
	case "cuda", "tensorrt":
		if m.modelCfg.NGPULayers == 0 {
			args = append(args, "-ngl", "99")
		}
	case "rocm":
		if m.modelCfg.NGPULayers == 0 {
			args = append(args, "-ngl", "99")
		}
		args = append(args, "-sm", "hip")
	case "vulkan":
		if m.modelCfg.NGPULayers == 0 {
			args = append(args, "-ngl", "99")
		}
		args = append(args, "-sm", "vulkan")
	case "coreml":
		if m.modelCfg.NGPULayers == 0 {
			args = append(args, "-ngl", "99")
		}
		args = append(args, "-sm", "metal")
	}

	if m.modelCfg.NCtx > 0 {
		args = append(args, "-c", strconv.Itoa(m.modelCfg.NCtx))
	}
	if m.modelCfg.NBatch > 0 {
		args = append(args, "-b", strconv.Itoa(m.modelCfg.NBatch))
	}
	if m.modelCfg.NUbatch > 0 {
		args = append(args, "-ub", strconv.Itoa(m.modelCfg.NUbatch))
	}
	if m.modelCfg.NParallel > 0 {
		args = append(args, "-np", strconv.Itoa(m.modelCfg.NParallel))
	}
	if m.modelCfg.KVUnified != nil {
		if *m.modelCfg.KVUnified {
			args = append(args, "--kv-unified")
		} else {
			args = append(args, "--no-kv-unified")
		}
	} else if m.modelCfg.NParallel > 0 {
		// 向后兼容：未显式配置 kv_unified 时，保持旧行为（统一 KV 池）。
		// 开启统一 KV 池后，每个 slot 的上下文仍是 n_ctx，而不是 n_ctx / n_parallel。
		args = append(args, "--kv-unified")
	}
	if m.modelCfg.MMProjFile != "" {
		args = append(args, "--mmproj", m.modelCfg.MMProjFile)
	}
	if m.modelCfg.Type == "embedding" {
		args = append(args, "--embeddings")
	}
	if m.modelCfg.Type == "rerank" {
		args = append(args, "--rerank")
	}
	if m.modelCfg.Jinja {
		args = append(args, "--jinja")
	}
	if m.modelCfg.ReasoningFormat != "" {
		args = append(args, "--reasoning-format", m.modelCfg.ReasoningFormat)
	}
	if m.modelCfg.ChatTemplate != "" {
		args = append(args, "--chat-template", m.modelCfg.ChatTemplate)
	}
	if m.modelCfg.ChatTemplateFile != "" {
		args = append(args, "--chat-template-file", m.modelCfg.ChatTemplateFile)
	}
	if len(m.modelCfg.ChatTemplateKwargs) > 0 {
		kwargsJSON, err := json.Marshal(m.modelCfg.ChatTemplateKwargs)
		if err == nil {
			args = append(args, "--chat-template-kwargs", string(kwargsJSON))
		}
	}
	if m.modelCfg.Webui != nil {
		if *m.modelCfg.Webui {
			args = append(args, "--webui")
		} else {
			args = append(args, "--no-webui")
		}
	}

	if m.modelCfg.PresencePenalty != "" {
		args = append(args, "--presence-penalty", m.modelCfg.PresencePenalty)
	}
	if m.modelCfg.Temperature != "" {
		args = append(args, "--temperature", m.modelCfg.Temperature)
	}
	if m.modelCfg.TopK != "" {
		args = append(args, "--top-k", m.modelCfg.TopK)
	}
	if m.modelCfg.TopP != "" {
		args = append(args, "--top-p", m.modelCfg.TopP)
	}
	if m.modelCfg.MinP != "" {
		args = append(args, "--min-p", m.modelCfg.MinP)
	}

	// 用户自定义参数原样透传，放在内置参数之后，便于覆盖默认值
	args = append(args, m.modelCfg.ExtraArgs...)

	return args
}

func (m *LlamaServerManager) Start() error {
	m.mu.Lock()
	if m.cancelFn != nil {
		m.cancelFn()
	}
	m.mu.Unlock()

	os.Remove(m.socketPath)

	ctx, cancel := context.WithCancel(context.Background())
	m.mu.Lock()
	m.cancelFn = cancel
	m.ready = false
	m.stopped = false
	m.mu.Unlock()

	args := m.buildArgs()

	log.Printf("llama-server %s: n_ctx=%d n_batch=%d n_ubatch=%d n_parallel=%d kv_unified=%s extra_args=%v",
		m.modelCfg.Name, m.modelCfg.NCtx, m.modelCfg.NBatch, m.modelCfg.NUbatch, m.modelCfg.NParallel,
		kvUnifiedState(m.modelCfg.KVUnified), m.modelCfg.ExtraArgs)
	log.Printf("llama-server %s args: %s", m.modelCfg.Name, strings.Join(redactArgs(args), " "))

	m.cmd = exec.CommandContext(ctx, m.serverBinary, args...)
	m.cmd.Stdout = os.Stdout
	m.cmd.Stderr = os.Stderr

	if err := m.cmd.Start(); err != nil {
		cancel()
		return fmt.Errorf("start llama-server: %w", err)
	}

	m.done = make(chan struct{})
	go m.monitor()

	if err := m.WaitForReady(60 * time.Second); err != nil {
		return err
	}

	m.fetchServerModelName()

	return nil
}

func (m *LlamaServerManager) WaitForReady(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		m.mu.Lock()
		stopped := m.stopped
		m.mu.Unlock()
		if stopped {
			return fmt.Errorf("llama-server exited during startup")
		}

		resp, err := m.client.Get("http://localhost/health")
		if err == nil && resp.StatusCode == 200 {
			resp.Body.Close()
			m.mu.Lock()
			m.ready = true
			m.mu.Unlock()
			log.Printf("llama-server ready at %s", m.socketPath)
			return nil
		}
		if resp != nil {
			resp.Body.Close()
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("llama-server not ready after %v", timeout)
}

func (m *LlamaServerManager) fetchServerModelName() {
	resp, err := m.client.Get("http://localhost/v1/models")
	if err != nil {
		return
	}
	defer resp.Body.Close()

	var result struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return
	}
	if len(result.Data) > 0 {
		m.mu.Lock()
		m.serverModelName = result.Data[0].ID
		m.mu.Unlock()
		log.Printf("llama-server model name: %s", result.Data[0].ID)
	}
}

func (m *LlamaServerManager) ServerModelName() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.serverModelName
}

func (m *LlamaServerManager) monitor() {
	err := m.cmd.Wait()
	close(m.done)
	m.mu.Lock()
	m.ready = false
	stopped := m.stopped
	m.mu.Unlock()

	if stopped {
		return
	}

	log.Printf("llama-server exited unexpectedly: %v", err)
	time.Sleep(5 * time.Second)
	m.mu.Lock()
	if m.stopped {
		m.mu.Unlock()
		return
	}
	m.mu.Unlock()
	m.restart()
}

func (m *LlamaServerManager) restart() {
	m.mu.Lock()
	if m.stopped {
		m.mu.Unlock()
		return
	}
	if m.cancelFn != nil {
		m.cancelFn()
	}
	m.mu.Unlock()

	os.Remove(m.socketPath)
	if err := m.Start(); err != nil {
		log.Printf("failed to restart llama-server: %v", err)
	}
}

func (m *LlamaServerManager) Shutdown() {
	m.mu.Lock()
	m.stopped = true
	if m.cancelFn != nil {
		m.cancelFn()
	}
	cmd := m.cmd
	done := m.done
	m.mu.Unlock()

	if cmd != nil && cmd.Process != nil {
		cmd.Process.Signal(syscall.SIGTERM)
		select {
		case <-done:
		case <-time.After(10 * time.Second):
			log.Printf("llama-server did not exit gracefully, killing (pid=%d)", cmd.Process.Pid)
			cmd.Process.Kill()
			<-done
		}
	}
	os.Remove(m.socketPath)
	log.Printf("llama-server shutdown complete: %s", m.socketPath)
}

func (m *LlamaServerManager) ProxyRequest(method, path string, body io.Reader) (*http.Response, error) {
	req, err := http.NewRequestWithContext(context.Background(), method, "http://localhost"+path, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return m.client.Do(req)
}

func (m *LlamaServerManager) ModelName() string {
	return m.modelCfg.Name
}

// kvUnifiedState 把 kv_unified 配置转成日志用的可读状态
func kvUnifiedState(p *bool) string {
	if p == nil {
		return "auto"
	}
	if *p {
		return "on"
	}
	return "off"
}

// sensitiveArgs 需要在日志中脱敏的参数（值紧跟在参数名之后）
var sensitiveArgs = map[string]bool{
	"--api-key":  true,
	"--hf-token": true,
	"--hft":      true,
}

// redactArgs 复制参数列表并把敏感参数的值替换掉，避免写进日志
func redactArgs(args []string) []string {
	out := make([]string, len(args))
	copy(out, args)
	for i := 0; i < len(out)-1; i++ {
		if sensitiveArgs[out[i]] {
			out[i+1] = "***"
		}
	}
	return out
}
