package model

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sction-ai/go/internal/config"
)

func writeFakeServer(t *testing.T, help string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "llama-server")
	script := "#!/bin/sh\ncat <<'EOF'\n" + help + "\nEOF\nexit 0\n"
	if err := os.WriteFile(path, []byte(script), 0755); err != nil {
		t.Fatalf("write fake server: %v", err)
	}
	return path
}

const fakeHelp = `usage: llama-server [options]
-h, --help
-lv, --verbosity, --log-verbosity    N
-cram, --cache-ram                    N
--cache-idle-slots
--no-cache-idle-slots
`

func TestProbeServerFlagsDetectsSupport(t *testing.T) {
	bin := writeFakeServer(t, fakeHelp)
	sup := probeServerFlags(bin)
	if !sup.known {
		t.Fatalf("expected known=true")
	}
	for _, f := range []string{"-lv", "--cache-idle-slots", "--cache-ram"} {
		if !sup.supports(f) {
			t.Errorf("expected support for %s", f)
		}
	}
}

func TestProbeServerFlagsOldBinary(t *testing.T) {
	bin := writeFakeServer(t, "usage: llama-server [options]\n-h, --help\n-lv, --verbosity N\n")
	sup := probeServerFlags(bin)
	if !sup.known {
		t.Fatalf("expected known=true")
	}
	if !sup.supports("-lv") {
		t.Errorf("expected -lv support")
	}
	if sup.supports("--cache-idle-slots") {
		t.Errorf("expected --cache-idle-slots unsupported")
	}
	if sup.supports("--cache-ram") {
		t.Errorf("expected --cache-ram unsupported")
	}
}

func TestProbeServerFlagsMissingBinary(t *testing.T) {
	sup := probeServerFlags("/nonexistent/llama-server")
	if sup.known {
		t.Fatalf("expected known=false")
	}
	if !sup.supports("-lv") {
		t.Errorf("expected -lv to still be allowed")
	}
	if sup.supports("--cache-ram") {
		t.Errorf("expected --cache-ram unsupported")
	}
}

func TestAppendServerArgs(t *testing.T) {
	bin := writeFakeServer(t, fakeHelp)
	v := 3
	yes := true
	mgr := NewLlamaServerManager(config.ModelConfig{Name: "m", LogVerbosity: &v, CacheIdleSlots: &yes}, bin, nil)
	args := mgr.appendServerArgs([]string{"-m", "x.gguf"})
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "-lv 3") {
		t.Errorf("missing -lv 3: %s", joined)
	}
	if !strings.Contains(joined, "--cache-idle-slots") {
		t.Errorf("missing --cache-idle-slots: %s", joined)
	}
	if strings.Contains(joined, "--no-cache-idle-slots") {
		t.Errorf("unexpected --no-cache-idle-slots: %s", joined)
	}
}

func TestAppendServerArgsSkipsUnsupported(t *testing.T) {
	bin := writeFakeServer(t, "usage: llama-server [options]\n-h, --help\n-lv, --verbosity N\n")
	no := false
	mgr := NewLlamaServerManager(config.ModelConfig{Name: "m", CacheIdleSlots: &no}, bin, nil)
	args := mgr.appendServerArgs([]string{"-m", "x.gguf"})
	joined := strings.Join(args, " ")
	if strings.Contains(joined, "cache-idle-slots") {
		t.Errorf("expected cache flag skipped: %s", joined)
	}
}

func TestLlamaServerEnv(t *testing.T) {
	t.Setenv("LLAMA_ARG_LOG_VERBOSITY", "4")
	t.Setenv("LLAMA_ARG_CACHE_IDLE_SLOTS", "1")
	env := llamaServerEnv(3)
	joined := strings.Join(env, "\n")
	if strings.Contains(joined, "LLAMA_ARG_LOG_VERBOSITY=4") {
		t.Errorf("inherited verbosity not stripped: %s", joined)
	}
	if !strings.Contains(joined, "LLAMA_ARG_LOG_VERBOSITY=3") {
		t.Errorf("verbosity not set: %s", joined)
	}
	if !strings.Contains(joined, "LLAMA_ARG_CACHE_IDLE_SLOTS=1") {
		t.Errorf("other env vars should be preserved: %s", joined)
	}
}

func TestLlamaServerLogWriterFile(t *testing.T) {
	dir := t.TempDir()
	logging := &config.LoggingConfig{LlamaServerLog: "file", LlamaServerLogDir: dir, MaxSizeMB: 1, MaxBackups: 1, MaxAgeDays: 1}
	w, closer, err := llamaServerLogWriter(logging, "qwen3.5")
	if err != nil {
		t.Fatalf("writer: %v", err)
	}
	if closer == nil {
		t.Fatalf("expected closer")
	}
	if _, err := w.Write([]byte("hello\n")); err != nil {
		t.Fatalf("write: %v", err)
	}
	closer.Close()
	p := filepath.Join(dir, "llama-server-qwen3.5.log")
	data, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("read log: %v", err)
	}
	if !strings.Contains(string(data), "hello") {
		t.Errorf("unexpected log content: %s", string(data))
	}
}

func TestLlamaServerLogWriterStdout(t *testing.T) {
	w, closer, err := llamaServerLogWriter(&config.LoggingConfig{LlamaServerLog: "stdout"}, "m")
	if err != nil {
		t.Fatalf("writer: %v", err)
	}
	if closer != nil {
		t.Errorf("expected nil closer for stdout mode")
	}
	if w != os.Stdout {
		t.Errorf("expected os.Stdout")
	}
}
