package handler

import (
	"bytes"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"sort"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"

	"github.com/sction-ai/go/internal/model"
	"github.com/sction-ai/go/internal/schema"
)

func Rerank(hub *model.ModelHub) gin.HandlerFunc {
	return func(c *gin.Context) {
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"detail": "failed to read request body"})
			return
		}

		var req schema.RerankRequest
		if err := parseJSON(body, &req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"detail": err.Error()})
			return
		}

		modelName := ""
		if req.Model != nil {
			modelName = *req.Model
		}

		modelCfg := hub.GetModelConfig(modelName)
		if modelCfg != nil && modelCfg.Runtime == "llama" {
			handleLlamaRerank(c, hub, modelName, body)
			return
		}

		reranker, err := hub.GetReranker(modelName)
		if err != nil {
			handleModelErrors(c, err)
			return
		}

		topN := len(req.Documents)
		if req.TopN != nil && *req.TopN > 0 && *req.TopN < topN {
			topN = *req.TopN
		}
		if req.TopK != nil && *req.TopK > 0 && *req.TopK < topN {
			topN = *req.TopK
		}

		results, totalTokens, err := reranker.Rerank(req.Query, req.Documents, topN)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"detail": err.Error()})
			return
		}

		sort.Slice(results, func(i, j int) bool {
			return results[i].RelevanceScore > results[j].RelevanceScore
		})

		rerankResults := make([]schema.RerankResult, len(results))
		for i, r := range results {
			rerankResults[i] = schema.RerankResult{
				Index:          r.Index,
				RelevanceScore: r.RelevanceScore,
			}
			if req.ReturnDocuments == nil || *req.ReturnDocuments {
				rerankResults[i].Document = &schema.Document{
					Text: req.Documents[r.Index],
				}
			}
		}

		resolvedName := hub.ResolveName(modelName)
		if resolvedName == "" {
			resolvedName = modelName
		}

		c.JSON(http.StatusOK, schema.RerankResponse{
			ID:      "rerank-" + uuid.New().String()[:8],
			Model:   resolvedName,
			Results: rerankResults,
			Usage: schema.UsageInfo{
				PromptTokens: totalTokens,
				TotalTokens:  totalTokens,
			},
			Meta: schema.RerankMeta{
				APIVersion:  schema.RerankMetaAPIVersion{Version: "2"},
				BilledUnits: schema.RerankMetaBilledUnits{SearchUnits: int(math.Ceil(float64(len(req.Documents)) / 100.0))},
				Tokens:      schema.RerankMetaTokens{InputTokens: totalTokens},
			},
		})
	}
}

func handleLlamaRerank(c *gin.Context, hub *model.ModelHub, modelName string, body []byte) {
	mgr, err := hub.GetLlamaReranker(modelName)
	if err != nil {
		handleModelErrors(c, err)
		return
	}

	var reqMap map[string]interface{}
	if err := json.Unmarshal(body, &reqMap); err == nil {
		serverName := mgr.ServerModelName()
		if serverName != "" {
			reqMap["model"] = serverName
		} else {
			delete(reqMap, "model")
		}
		if stripped, err := json.Marshal(reqMap); err == nil {
			body = stripped
		}
	}

	resp, err := mgr.ProxyRequest(http.MethodPost, "/v1/rerank", bytes.NewReader(body))
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
