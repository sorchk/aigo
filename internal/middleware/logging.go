package middleware

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/sction-ai/go/internal/config"

	"gopkg.in/natefinch/lumberjack.v2"
)

type responseWriter struct {
	gin.ResponseWriter
	body       *bytes.Buffer
	statusCode int
}

func (w *responseWriter) Write(b []byte) (int, error) {
	w.body.Write(b)
	return w.ResponseWriter.Write(b)
}

func (w *responseWriter) WriteHeader(statusCode int) {
	w.statusCode = statusCode
	w.ResponseWriter.WriteHeader(statusCode)
}

type RequestLog struct {
	RequestID string      `json:"request_id"`
	Time      string      `json:"time"`
	Type      string      `json:"type"`
	Method    string      `json:"method"`
	Path      string      `json:"path"`
	Query     string      `json:"query,omitempty"`
	Body      interface{} `json:"body"`
	BodySize  int         `json:"body_size"`
}

type ResponseLog struct {
	RequestID  string      `json:"request_id"`
	Time       string      `json:"time"`
	Type       string      `json:"type"`
	Status     int         `json:"status"`
	Body       interface{} `json:"body"`
	BodySize   int         `json:"body_size"`
	DurationMs int64       `json:"duration_ms"`
	Stream     bool        `json:"stream,omitempty"`
}

var (
	logger        *slog.Logger
	logConfig     *config.LoggingConfig
	sensitiveKeys = []*regexp.Regexp{
		regexp.MustCompile(`(?i)(token|password|secret|api_?key|authorization|Bearer|x-api-key)`),
		regexp.MustCompile(`(?i).*(token|secret|password|key).*`),
	}
	stringSanitizer = regexp.MustCompile(`(?i)(token|password|secret|api_?key|authorization)\s*[:=]\s*["']?[^"'\s&]+["']?`)
)

func InitLogging(cfg *config.LoggingConfig) error {
	logConfig = cfg
	if !cfg.Enabled {
		logger = slog.New(slog.DiscardHandler)
		return nil
	}

	if err := os.MkdirAll(filepath.Dir(cfg.Path), 0755); err != nil {
		return err
	}

	writer := &lumberjack.Logger{
		Filename:   cfg.Path,
		MaxSize:    cfg.MaxSizeMB,
		MaxBackups: cfg.MaxBackups,
		MaxAge:     cfg.MaxAgeDays,
		Compress:   cfg.Compress,
	}

	logger = slog.New(slog.NewJSONHandler(writer, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))

	return nil
}

func RequestLogger() gin.HandlerFunc {
	return func(c *gin.Context) {
		if logConfig == nil || !logConfig.Enabled {
			c.Next()
			return
		}

		requestID := uuid.New().String()
		c.Set("request_id", requestID)
		c.Header("X-Request-ID", requestID)

		startTime := time.Now()

		bodyBytes, err := io.ReadAll(c.Request.Body)
		if err != nil {
			bodyBytes = []byte{}
		}
		c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

		sanitizedBody := sanitizeBody(bodyBytes)

		reqLog := RequestLog{
			RequestID: requestID,
			Time:      startTime.UTC().Format(time.RFC3339Nano),
			Type:      "request",
			Method:    c.Request.Method,
			Path:      c.Request.URL.Path,
			Query:     c.Request.URL.RawQuery,
			Body:      sanitizedBody,
			BodySize:  len(bodyBytes),
		}

		if c.Request.ContentLength > 0 {
			reqLog.BodySize = int(c.Request.ContentLength)
		}

		rw := &responseWriter{
			ResponseWriter: c.Writer,
			body:           bytes.NewBuffer(nil),
			statusCode:     http.StatusOK,
		}
		c.Writer = rw

		c.Next()

		duration := time.Since(startTime)

		stream := c.GetHeader("Accept") == "text/event-stream" ||
			strings.Contains(c.GetHeader("Content-Type"), "text/event-stream")

		var respBody interface{}
		if stream {
			respBody = string(rw.body.Bytes())
		} else {
			respBody = sanitizeBody(rw.body.Bytes())
		}

		respLog := ResponseLog{
			RequestID:  requestID,
			Time:       time.Now().UTC().Format(time.RFC3339Nano),
			Type:       "response",
			Status:     rw.statusCode,
			Body:       respBody,
			BodySize:   rw.body.Len(),
			DurationMs: duration.Milliseconds(),
			Stream:     stream,
		}

		if rw.statusCode == 0 {
			respLog.Status = http.StatusOK
		}

		logger.Info("request completed",
			slog.String("request_id", requestID),
			slog.String("type", "request"),
			slog.String("method", c.Request.Method),
			slog.String("path", c.Request.URL.Path),
			slog.Any("body", sanitizedBody),
			slog.Int("body_size", len(bodyBytes)),
		)

		logger.Info("response sent",
			slog.String("request_id", requestID),
			slog.String("type", "response"),
			slog.Int("status", respLog.Status),
			slog.Any("body", respBody),
			slog.Int("body_size", rw.body.Len()),
			slog.Int64("duration_ms", duration.Milliseconds()),
			slog.Bool("stream", stream),
		)
	}
}

func sanitizeBody(data []byte) interface{} {
	if len(data) == 0 {
		return nil
	}

	var raw interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		return string(data)
	}

	return sanitizeValue(raw)
}

func sanitizeValue(v interface{}) interface{} {
	switch val := v.(type) {
	case map[string]interface{}:
		result := make(map[string]interface{})
		for k, v := range val {
			if isSensitiveKey(k) {
				result[k] = "***"
			} else {
				result[k] = sanitizeValue(v)
			}
		}
		return result
	case []interface{}:
		result := make([]interface{}, len(val))
		for i, v := range val {
			result[i] = sanitizeValue(v)
		}
		return result
	case string:
		return sanitizeString(val)
	default:
		return val
	}
}

func isSensitiveKey(key string) bool {
	for _, re := range sensitiveKeys {
		if re.MatchString(key) {
			return true
		}
	}
	return false
}

func sanitizeString(s string) string {
	return stringSanitizer.ReplaceAllStringFunc(s, func(match string) string {
		parts := regexp.MustCompile(`([:=]\s*)["']?[^"'\s&]+["']?`).FindStringSubmatch(match)
		if len(parts) > 1 {
			return parts[1] + "***"
		}
		return "***"
	})
}
