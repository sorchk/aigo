package handler

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"

	"github.com/sction-ai/go/internal/model"
	"github.com/sction-ai/go/internal/schema"
)

func ChatCompletions(hub *model.ModelHub) gin.HandlerFunc {
	return func(c *gin.Context) {
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"message": "failed to read request body", "type": "invalid_request_error"}})
			return
		}

		var req schema.ChatCompletionRequest
		if err := parseJSON(body, &req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"message": err.Error(), "type": "invalid_request_error"}})
			return
		}

		modelName := req.Model

		llmSvc, err := hub.GetAnyLLM(modelName)
		if err != nil {
			handleModelErrors(c, err)
			return
		}

		switch svc := llmSvc.(type) {
		case *model.OnnxLLMService:
			handleOnnxChatCompletion(c, hub, svc, req)
		case *model.LlamaServerManager:
			resp, err := svc.ProxyRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
			if err != nil {
				c.JSON(http.StatusBadGateway, gin.H{"error": gin.H{"message": err.Error(), "type": "server_error"}})
				return
			}
			defer resp.Body.Close()
			for key, values := range resp.Header {
				for _, value := range values {
					c.Writer.Header().Add(key, value)
				}
			}
			c.Writer.WriteHeader(resp.StatusCode)
			io.Copy(c.Writer, resp.Body)
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": gin.H{"message": "unknown LLM type", "type": "server_error"}})
		}
	}
}

func hasMultimodalContent(messages []schema.ChatMessage) bool {
	for _, msg := range messages {
		_, parts := msg.ParseContent()
		for _, p := range parts {
			if p.Type == "image_url" && p.ImageURL != nil {
				return true
			}
		}
	}
	return false
}

func handleOnnxChatCompletion(c *gin.Context, hub *model.ModelHub, svc *model.OnnxLLMService, req schema.ChatCompletionRequest) {
	if hasMultimodalContent(req.Messages) && svc.HasVisionEncoder() {
		handleOnnxMultimodalChatCompletion(c, hub, svc, req)
		return
	}

	prompt := formatChatPrompt(req.Messages)

	tokenizer := svc.Tokenizer()
	inputIDs, err := tokenizer.Encode(prompt)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": gin.H{"message": fmt.Sprintf("tokenize: %v", err), "type": "server_error"}})
		return
	}

	inputIDsInt64 := make([]int64, len(inputIDs))
	for i, id := range inputIDs {
		inputIDsInt64[i] = int64(id)
	}

	params := buildGenerateParams(req)

	resolvedName := hub.ResolveName(req.Model)
	if resolvedName == "" {
		resolvedName = req.Model
	}

	stream := false
	if req.Stream != nil {
		stream = *req.Stream
	}

	if stream {
		handleOnnxChatStream(c, svc, inputIDsInt64, params, resolvedName, len(inputIDs), req)
		return
	}

	generatedIDs, err := svc.Generate(inputIDsInt64, params)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": gin.H{"message": fmt.Sprintf("generate: %v", err), "type": "server_error"}})
		return
	}

	generatedIDs = filterStopTokens(generatedIDs, req.Stop)

	text, err := tokenizer.Decode(toIntSlice(generatedIDs))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": gin.H{"message": fmt.Sprintf("decode: %v", err), "type": "server_error"}})
		return
	}

	finishReason := "stop"
	c.JSON(http.StatusOK, schema.ChatCompletionResponse{
		ID:      "chatcmpl-" + uuid.New().String()[:8],
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   resolvedName,
		Choices: []schema.ChatChoice{
			{
				Index: 0,
				Message: schema.ChatMessage{
					Role:    "assistant",
					Content: json.RawMessage(`"` + jsonEscape(text) + `"`),
				},
				FinishReason: finishReason,
			},
		},
		Usage: schema.UsageInfo{
			PromptTokens:     len(inputIDs),
			CompletionTokens: len(generatedIDs),
			TotalTokens:      len(inputIDs) + len(generatedIDs),
		},
	})
}

func handleOnnxMultimodalChatCompletion(c *gin.Context, hub *model.ModelHub, svc *model.OnnxLLMService, req schema.ChatCompletionRequest) {
	var images []imageData
	var multimodalMsgs []multimodalMessage

	for _, msg := range req.Messages {
		text, parts := msg.ParseContent()
		mmMsg := multimodalMessage{role: msg.Role}

		if len(parts) > 0 {
			for _, p := range parts {
				switch p.Type {
				case "text":
					mmMsg.parts = append(mmMsg.parts, contentPart{typ: "text", text: p.Text})
				case "image_url":
					if p.ImageURL == nil || p.ImageURL.URL == "" {
						continue
					}
					img, err := model.DecodeImageFromURL(p.ImageURL.URL)
					if err != nil {
						c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"message": fmt.Sprintf("load image: %v", err), "type": "invalid_request_error"}})
						return
					}
					result, err := model.PreprocessImage(img)
					if err != nil {
						c.JSON(http.StatusInternalServerError, gin.H{"error": gin.H{"message": fmt.Sprintf("preprocess image: %v", err), "type": "server_error"}})
						return
					}
					images = append(images, imageData{result: result})
					mmMsg.parts = append(mmMsg.parts, contentPart{
						typ:         "image",
						imgIdx:      len(images) - 1,
						numFeatures: result.NumFeatures,
					})
				}
			}
		} else {
			mmMsg.parts = append(mmMsg.parts, contentPart{typ: "text", text: text})
		}
		multimodalMsgs = append(multimodalMsgs, mmMsg)
	}

	tokenizer := svc.Tokenizer()
	inputIDs, err := buildMultimodalTokenIDs(multimodalMsgs, images, tokenizer)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": gin.H{"message": fmt.Sprintf("build tokens: %v", err), "type": "server_error"}})
		return
	}

	var allPixelValues []float32
	var allGridTHW []int64
	totalFeatures := 0
	for _, img := range images {
		allPixelValues = append(allPixelValues, img.result.PixelValues...)
		allGridTHW = append(allGridTHW, img.result.GridTHW...)
		totalFeatures += img.result.NumFeatures
	}

	imageFeatures, err := svc.EncodeImage(allPixelValues, allGridTHW, totalFeatures)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": gin.H{"message": fmt.Sprintf("encode image: %v", err), "type": "server_error"}})
		return
	}

	params := buildGenerateParams(req)
	resolvedName := hub.ResolveName(req.Model)
	if resolvedName == "" {
		resolvedName = req.Model
	}

	stream := false
	if req.Stream != nil {
		stream = *req.Stream
	}

	if stream {
		handleOnnxMultimodalChatStream(c, svc, inputIDs, imageFeatures, params, resolvedName, len(inputIDs), req)
		return
	}

	generatedIDs, err := svc.GenerateMultimodal(inputIDs, imageFeatures, params)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": gin.H{"message": fmt.Sprintf("generate: %v", err), "type": "server_error"}})
		return
	}

	generatedIDs = filterStopTokens(generatedIDs, req.Stop)
	text, err := tokenizer.Decode(toIntSlice(generatedIDs))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": gin.H{"message": fmt.Sprintf("decode: %v", err), "type": "server_error"}})
		return
	}

	finishReason := "stop"
	c.JSON(http.StatusOK, schema.ChatCompletionResponse{
		ID:      "chatcmpl-" + uuid.New().String()[:8],
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   resolvedName,
		Choices: []schema.ChatChoice{
			{
				Index: 0,
				Message: schema.ChatMessage{
					Role:    "assistant",
					Content: json.RawMessage(`"` + jsonEscape(text) + `"`),
				},
				FinishReason: finishReason,
			},
		},
		Usage: schema.UsageInfo{
			PromptTokens:     len(inputIDs),
			CompletionTokens: len(generatedIDs),
			TotalTokens:      len(inputIDs) + len(generatedIDs),
		},
	})
}

type contentPart struct {
	typ         string
	text        string
	imgIdx      int
	numFeatures int
}

type multimodalMessage struct {
	role  string
	parts []contentPart
}

type imageData struct {
	result *model.VisionPreprocessResult
}

func buildMultimodalTokenIDs(messages []multimodalMessage, images []imageData, tokenizer interface{ Encode(string) ([]int, error) }) ([]int64, error) {
	const (
		visionStartID = 248053
		visionEndID   = 248054
		imagePadID    = 248056
	)

	var allIDs []int64

	for _, msg := range messages {
		prefix := "<|im_start|>" + msg.role + "\n"
		prefixIDs, err := tokenizer.Encode(prefix)
		if err != nil {
			return nil, fmt.Errorf("encode prefix: %w", err)
		}
		for _, id := range prefixIDs {
			allIDs = append(allIDs, int64(id))
		}

		for _, part := range msg.parts {
			switch part.typ {
			case "text":
				ids, err := tokenizer.Encode(part.text)
				if err != nil {
					return nil, fmt.Errorf("encode text: %w", err)
				}
				for _, id := range ids {
					allIDs = append(allIDs, int64(id))
				}
			case "image":
				allIDs = append(allIDs, visionStartID)
				for i := 0; i < part.numFeatures; i++ {
					allIDs = append(allIDs, imagePadID)
				}
				allIDs = append(allIDs, visionEndID)
			}
		}

		suffix := "<|im_end|>\n"
		suffixIDs, err := tokenizer.Encode(suffix)
		if err != nil {
			return nil, fmt.Errorf("encode suffix: %w", err)
		}
		for _, id := range suffixIDs {
			allIDs = append(allIDs, int64(id))
		}
	}

	assistantIDs, err := tokenizer.Encode("<|im_start|>assistant\n")
	if err != nil {
		return nil, fmt.Errorf("encode assistant prefix: %w", err)
	}
	for _, id := range assistantIDs {
		allIDs = append(allIDs, int64(id))
	}

	return allIDs, nil
}

func handleOnnxChatStream(c *gin.Context, svc *model.OnnxLLMService, inputIDs []int64, params model.GenerateParams, modelName string, promptTokens int, req schema.ChatCompletionRequest) {
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")

	completionID := "chatcmpl-" + uuid.New().String()[:8]
	created := time.Now().Unix()

	chunk := schema.ChatCompletionChunk{
		ID:      completionID,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   modelName,
		Choices: []schema.ChatStreamChoice{
			{
				Index: 0,
				Delta: schema.ChatStreamDelta{
					Role: "assistant",
				},
			},
		},
	}
	writeSSE(c, chunk)

	tokenizer := svc.Tokenizer()
	stopTokenIDs := getStopTokenIDs(req.Stop)

	var generatedCount int
	params.Callback = func(tokenID int64) error {
		for _, stopID := range stopTokenIDs {
			if tokenID == stopID {
				return fmt.Errorf("stop")
			}
		}

		text, err := tokenizer.Decode([]int{int(tokenID)})
		if err != nil {
			return nil
		}

		chunk := schema.ChatCompletionChunk{
			ID:      completionID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   modelName,
			Choices: []schema.ChatStreamChoice{
				{
					Index: 0,
					Delta: schema.ChatStreamDelta{
						Content: text,
					},
				},
			},
		}
		writeSSE(c, chunk)
		generatedCount++
		return nil
	}

	_, _ = svc.Generate(inputIDs, params)

	finishReason := "stop"
	finalChunk := schema.ChatCompletionChunk{
		ID:      completionID,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   modelName,
		Choices: []schema.ChatStreamChoice{
			{
				Index:        0,
				FinishReason: &finishReason,
			},
		},
	}
	writeSSE(c, finalChunk)

	usageChunk := fmt.Sprintf("data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%d,\"model\":\"%s\",\"choices\":[],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}\n\n",
		completionID, created, modelName, promptTokens, generatedCount, promptTokens+generatedCount)
	c.Writer.WriteString(usageChunk)
	c.Writer.WriteString("data: [DONE]\n\n")
	c.Writer.Flush()
}

func handleOnnxMultimodalChatStream(c *gin.Context, svc *model.OnnxLLMService, inputIDs []int64, imageFeatures [][]float32, params model.GenerateParams, modelName string, promptTokens int, req schema.ChatCompletionRequest) {
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")

	completionID := "chatcmpl-" + uuid.New().String()[:8]
	created := time.Now().Unix()

	chunk := schema.ChatCompletionChunk{
		ID:      completionID,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   modelName,
		Choices: []schema.ChatStreamChoice{
			{
				Index: 0,
				Delta: schema.ChatStreamDelta{
					Role: "assistant",
				},
			},
		},
	}
	writeSSE(c, chunk)

	tokenizer := svc.Tokenizer()
	stopTokenIDs := getStopTokenIDs(req.Stop)

	var generatedCount int
	params.Callback = func(tokenID int64) error {
		for _, stopID := range stopTokenIDs {
			if tokenID == stopID {
				return fmt.Errorf("stop")
			}
		}

		text, err := tokenizer.Decode([]int{int(tokenID)})
		if err != nil {
			return nil
		}

		chunk := schema.ChatCompletionChunk{
			ID:      completionID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   modelName,
			Choices: []schema.ChatStreamChoice{
				{
					Index: 0,
					Delta: schema.ChatStreamDelta{
						Content: text,
					},
				},
			},
		}
		writeSSE(c, chunk)
		generatedCount++
		return nil
	}

	_, _ = svc.GenerateMultimodal(inputIDs, imageFeatures, params)

	finishReason := "stop"
	finalChunk := schema.ChatCompletionChunk{
		ID:      completionID,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   modelName,
		Choices: []schema.ChatStreamChoice{
			{
				Index:        0,
				FinishReason: &finishReason,
			},
		},
	}
	writeSSE(c, finalChunk)

	usageChunk := fmt.Sprintf("data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%d,\"model\":\"%s\",\"choices\":[],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}\n\n",
		completionID, created, modelName, promptTokens, generatedCount, promptTokens+generatedCount)
	c.Writer.WriteString(usageChunk)
	c.Writer.WriteString("data: [DONE]\n\n")
	c.Writer.Flush()
}

func writeSSE(c *gin.Context, v interface{}) {
	data, _ := json.Marshal(v)
	c.Writer.WriteString("data: " + string(data) + "\n\n")
	c.Writer.Flush()
}

func formatChatPrompt(messages []schema.ChatMessage) string {
	var sb strings.Builder
	for _, msg := range messages {
		text, _ := msg.ParseContent()
		sb.WriteString("<|im_start|>")
		sb.WriteString(msg.Role)
		sb.WriteString("\n")
		sb.WriteString(text)
		sb.WriteString("<|im_end|>\n")
	}
	sb.WriteString("<|im_start|>assistant\n")
	return sb.String()
}

func filterStopTokens(ids []int64, stops []string) []int64 {
	return ids
}

func getStopTokenIDs(stops []string) []int64 {
	var result []int64
	for _, s := range stops {
		if s == "<|im_end|>" {
			result = append(result, 248044)
		}
	}
	return result
}

func buildGenerateParams(req schema.ChatCompletionRequest) model.GenerateParams {
	maxTokens := 512
	if req.MaxTokens != nil && *req.MaxTokens > 0 {
		maxTokens = *req.MaxTokens
	}

	temperature := 0.6
	if req.Temperature != nil && *req.Temperature >= 0 {
		temperature = *req.Temperature
	}

	topP := 0.0
	if req.TopP != nil && *req.TopP > 0 {
		topP = *req.TopP
	}

	topK := 0
	if req.TopK != nil && *req.TopK > 0 {
		topK = *req.TopK
	}

	return model.GenerateParams{
		MaxTokens:   maxTokens,
		Temperature: temperature,
		TopP:        topP,
		TopK:        topK,
	}
}

func toIntSlice(ids []int64) []int {
	result := make([]int, len(ids))
	for i, id := range ids {
		result[i] = int(id)
	}
	return result
}

func jsonEscape(s string) string {
	b, _ := json.Marshal(s)
	return string(b[1 : len(b)-1])
}
