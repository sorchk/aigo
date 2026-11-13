package model

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"gopkg.in/natefinch/lumberjack.v2"

	"github.com/sction-ai/go/internal/config"
)

// llamaServerFlags 是 aigo 需要探测的 llama-server 参数。
// 上游不同版本支持的参数并不一致（例如 --cache-idle-slots 是较新版本才有的），
// 直接透传可能让 llama-server 启动失败，因此先探测再使用。
var llamaServerFlags = []string{"-lv", "--cache-idle-slots", "--cache-ram"}

// flagSupport 描述当前 llama-server 二进制对参数的支持情况。
type flagSupport struct {
	known   bool            // 是否成功读到 --help 输出
	present map[string]bool // 参数是否出现在帮助信息中
}

// supports 判断参数是否可用。
// 探测失败时只放行 -lv（各历史版本都支持），新增参数宁可跳过也不能让服务起不来。
func (s flagSupport) supports(flag string) bool {
	if !s.known {
		return flag == "-lv"
	}
	return s.present[flag]
}

var probeCache sync.Map // binary path -> flagSupport

// probeServerFlags 读取 `llama-server --help`，判断二进制支持哪些参数。
// 结果按二进制路径缓存，一个进程内只探测一次。
func probeServerFlags(binary string) flagSupport {
	if v, hit := probeCache.Load(binary); hit {
		return v.(flagSupport)
	}

	sup := flagSupport{present: map[string]bool{}}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, binary, "--help")
	out, err := cmd.CombinedOutput()
	help := string(out)
	if err != nil && ctx.Err() != nil {
		log.Printf("llama-server %s: --help probe timed out, only -lv will be passed", binary)
	}
	// 用帮助文本本身判断，而不是退出码：部分版本的 --help 返回非 0，
	// 但只要打印了用法（其中一定包含 "--help"），参数支持信息就是可信的。
	if strings.Contains(help, "--help") {
		sup.known = true
		for _, f := range llamaServerFlags {
			sup.present[f] = strings.Contains(help, f)
		}
	}

	probeCache.Store(binary, sup)
	return sup
}

// llamaServerEnv 返回子进程环境变量。
// 移除可能把日志级别抬到 TRACE 的 LLAMA_ARG_LOG_VERBOSITY，再写入本次配置生效的级别。
// 上游的解析顺序是 环境变量 → 命令行，两者保持一致最稳妥。
func llamaServerEnv(verbosity int) []string {
	env := make([]string, 0, len(os.Environ())+1)
	for _, kv := range os.Environ() {
		if strings.HasPrefix(kv, "LLAMA_ARG_LOG_VERBOSITY=") {
			continue
		}
		env = append(env, kv)
	}
	return append(env, fmt.Sprintf("LLAMA_ARG_LOG_VERBOSITY=%d", verbosity))
}

// llamaServerLogWriter 按配置决定 llama-server 子进程日志的落盘方式。
// 默认写独立的轮转文件：子进程日志与业务请求无关，
// 继承父进程 stdout 时会被 systemd 收进 journald/系统日志并迅速撑爆磁盘。
func llamaServerLogWriter(logging *config.LoggingConfig, modelName string) (io.Writer, io.Closer, error) {
	mode := "file"
	dir := "logs"
	maxSize, maxBackups, maxAge, compress := 100, 7, 30, true

	if logging != nil {
		if logging.LlamaServerLog != "" {
			mode = logging.LlamaServerLog
		}
		if logging.LlamaServerLogDir != "" {
			dir = logging.LlamaServerLogDir
		}
		maxSize, maxBackups, maxAge, compress = logging.MaxSizeMB, logging.MaxBackups, logging.MaxAgeDays, logging.Compress
	}

	switch mode {
	case "stdout":
		return os.Stdout, nil, nil
	case "discard":
		return io.Discard, nil, nil
	}

	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, nil, fmt.Errorf("create llama-server log dir: %w", err)
	}

	path := filepath.Join(dir, fmt.Sprintf("llama-server-%s.log", sanitizeLogFileName(modelName)))
	lj := &lumberjack.Logger{
		Filename:   path,
		MaxSize:    maxSize,
		MaxBackups: maxBackups,
		MaxAge:     maxAge,
		Compress:   compress,
	}
	return lj, lj, nil
}

// sanitizeLogFileName 把模型名转换成安全的日志文件名
func sanitizeLogFileName(name string) string {
	safe := strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9',
			r == '-', r == '_', r == '.':
			return r
		default:
			return '_'
		}
	}, name)
	if safe == "" || safe == "." || safe == ".." {
		safe = "model"
	}
	return safe
}
