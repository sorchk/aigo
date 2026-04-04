package middleware

import (
	"strings"

	"github.com/gin-gonic/gin"
)

func AuthMiddleware(token string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if token == "" {
			c.Next()
			return
		}

		auth := c.GetHeader("Authorization")
		if !strings.HasPrefix(auth, "Bearer ") || auth[7:] != token {
			c.AbortWithStatusJSON(401, gin.H{"detail": "Invalid or missing API token"})
			return
		}
		c.Next()
	}
}
