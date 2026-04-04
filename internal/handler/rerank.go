package handler

import (
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
		var req schema.RerankRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"detail": err.Error()})
			return
		}

		modelName := ""
		if req.Model != nil {
			modelName = *req.Model
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
