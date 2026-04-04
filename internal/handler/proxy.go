package handler

import (
	"io"
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/sction-ai/go/internal/model"
)

func Proxy(hub *model.ModelHub) gin.HandlerFunc {
	return func(c *gin.Context) {
		llmModel, err := hub.GetDefaultLLM()
		if err != nil {
			handleModelErrors(c, err)
			return
		}

		path := c.Request.URL.Path
		if c.Request.URL.RawQuery != "" {
			path = path + "?" + c.Request.URL.RawQuery
		}

		var body io.Reader
		if c.Request.Body != nil {
			body = c.Request.Body
		}

		resp, err := llmModel.ProxyRequest(c.Request.Method, path, body)
		if err != nil {
			c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
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
