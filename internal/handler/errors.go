package handler

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/sction-ai/go/internal/model"
	"github.com/sction-ai/go/internal/schema"
)

func handleModelErrors(c *gin.Context, err error) {
	switch err.(type) {
	case *schema.ModelNotReadyError:
		c.Header("Retry-After", "30")
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error":   "model_not_ready",
			"message": err.Error(),
		})
	case *model.ValueError:
		c.JSON(http.StatusNotFound, gin.H{
			"error":   "model_not_found",
			"message": err.Error(),
		})
	default:
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "internal_error",
			"message": err.Error(),
		})
	}
}
