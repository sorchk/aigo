package handler

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/sction-ai/go/internal/model"
	"github.com/sction-ai/go/internal/schema"
)

func CreateEmbeddings(hub *model.ModelHub) gin.HandlerFunc {
	return func(c *gin.Context) {
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"detail": "failed to read request body"})
			return
		}

		var modelName string
		type modelReq struct {
			Model *string `json:"model"`
		}
		var partial modelReq
		if err := parseJSON(body, &partial); err == nil && partial.Model != nil {
			modelName = *partial.Model
		}

		modelCfg := hub.GetModelConfig(modelName)
		if modelCfg != nil && modelCfg.Runtime == "onnx" {
			handleONNXEmbedding(c, hub, modelName, body)
			return
		}

		mgr, err := hub.GetEmbedding(modelName)
		if err != nil {
			handleModelErrors(c, err)
			return
		}

		resp, err := mgr.ProxyRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(body))
		if err != nil {
			c.JSON(http.StatusBadGateway, gin.H{"detail": err.Error()})
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
	}
}

func handleONNXEmbedding(c *gin.Context, hub *model.ModelHub, modelName string, body []byte) {
	var req schema.EmbeddingRequest
	if err := parseJSON(body, &req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"detail": err.Error()})
		return
	}

	svc, err := hub.GetONNXEmbedding(modelName)
	if err != nil {
		handleModelErrors(c, err)
		return
	}

	texts := extractInputTexts(req.Input)

	embeddings, totalTokens, err := svc.Embed(texts)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
		return
	}

	resolvedName := hub.ResolveName(modelName)
	if resolvedName == "" {
		resolvedName = modelName
	}

	data := make([]schema.EmbeddingData, len(embeddings))
	for i, emb := range embeddings {
		data[i] = schema.EmbeddingData{
			Object:    "embedding",
			Embedding: emb,
			Index:     i,
		}
	}

	c.JSON(http.StatusOK, schema.EmbeddingResponse{
		Object: "list",
		Data:   data,
		Model:  resolvedName,
		Usage: schema.EmbeddingUsage{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	})
}

func extractInputTexts(input interface{}) []string {
	switch v := input.(type) {
	case string:
		return []string{v}
	case []interface{}:
		texts := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok {
				texts = append(texts, s)
			}
		}
		return texts
	default:
		return nil
	}
}

func parseJSON(data []byte, v interface{}) error {
	return json.Unmarshal(data, v)
}
