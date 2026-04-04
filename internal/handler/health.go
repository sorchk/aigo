package handler

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/sction-ai/go/internal/model"
)

func Health(hub *model.ModelHub) gin.HandlerFunc {
	return func(c *gin.Context) {
		statuses := hub.GetStatuses()

		rerankModels := convertStatuses(statuses["rerank"])
		embedModels := convertStatuses(statuses["embedding"])
		llmModels := convertStatuses(statuses["llm"])

		allReady := true
		for _, m := range rerankModels {
			if m.Status != "ready" && m.Status != "not_loaded" {
				allReady = false
				break
			}
		}
		if allReady {
			for _, m := range embedModels {
				if m.Status != "ready" && m.Status != "not_loaded" {
					allReady = false
					break
				}
			}
		}
		if allReady {
			for _, m := range llmModels {
				if m.Status != "ready" && m.Status != "not_loaded" {
					allReady = false
					break
				}
			}
		}

		status := "ok"
		if !allReady {
			status = "degraded"
		}

		c.JSON(http.StatusOK, gin.H{
			"status": status,
			"system": getSystemInfo(),
			"models": gin.H{
				"rerank":    rerankModels,
				"embedding": embedModels,
				"llm":       llmModels,
			},
		})
	}
}

type modelStatus struct {
	Name   string `json:"name"`
	Status string `json:"status"`
}

func convertStatuses(in []model.ModelStatus) []modelStatus {
	if in == nil {
		return []modelStatus{}
	}
	var out []modelStatus
	for _, m := range in {
		out = append(out, modelStatus{Name: m.Name, Status: m.Status})
		for _, a := range m.Aliases {
			out = append(out, modelStatus{Name: a, Status: m.Status})
		}
	}
	return out
}

func ListModels(hub *model.ModelHub) gin.HandlerFunc {
	return func(c *gin.Context) {
		allModels := hub.ListModels()

		c.JSON(http.StatusOK, gin.H{
			"object": "list",
			"data":   allModels,
		})
	}
}
