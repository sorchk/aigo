package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/sction-ai/go/internal/config"
	"github.com/sction-ai/go/internal/handler"
	"github.com/sction-ai/go/internal/middleware"
	"github.com/sction-ai/go/internal/model"
	"github.com/sction-ai/go/internal/service"
)

func main() {
	if len(os.Args) < 2 {
		runServer()
		return
	}

	switch os.Args[1] {
	case "run":
		runServer()
	case "install":
		installCmd := flag.NewFlagSet("install", flag.ExitOnError)
		user := installCmd.String("user", "", "User to run the service as (default: current user)")
		workDir := installCmd.String("workdir", "", "Working directory (default: binary directory)")
		configPath := installCmd.String("config", "", "Path to config.yaml (default: config.yaml in workdir)")
		host := installCmd.String("host", "0.0.0.0", "Listen host")
		port := installCmd.Int("port", 8000, "Listen port")
		installCmd.Parse(os.Args[2:])

		opts := &service.InstallOptions{
			User:       *user,
			WorkingDir: *workDir,
			ConfigPath: *configPath,
			Host:       *host,
			Port:       *port,
		}
		if err := service.Install(opts); err != nil {
			log.Fatalf("Install failed: %v", err)
		}
	case "uninstall":
		if err := service.Uninstall(); err != nil {
			log.Fatalf("Uninstall failed: %v", err)
		}
	case "help", "--help", "-h":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Printf(`Usage: %s <command> [options]

Commands:
  run        Start the API server (default when no command is given)
  install    Install as a systemd service (requires sudo)
  uninstall  Uninstall the systemd service (requires sudo)
  help       Show this help message


Examples:
  %s                                          # Run server with defaults
  %s run                                      # Same as above
  sudo %s install                             # Install as systemd service
  sudo %s uninstall                           # Remove systemd service
`, os.Args[0], os.Args[0], os.Args[0], os.Args[0], os.Args[0])
}

func getBinDir() string {
	exe, err := os.Executable()
	if err != nil {
		wd, _ := os.Getwd()
		return wd
	}
	link, err := filepath.EvalSymlinks(exe)
	if err != nil {
		link = exe
	}
	return filepath.Dir(link)
}

func runServer() {
	confDir := config.GetEnvString("CONF_DIR", "conf")
	host := config.GetEnvString("HOST", "0.0.0.0")
	port := config.GetEnvInt("PORT", 8000)

	if err := config.EnsureConfDir(confDir); err != nil {
		log.Fatalf("Failed to ensure conf dir: %v", err)
	}

	configPath := config.GetEnvString("CONFIG_PATH", filepath.Join(confDir, "config.yaml"))

	if !filepath.IsAbs(configPath) {
		if _, err := os.Stat(configPath); err != nil {
			binDir := getBinDir()
			candidate := filepath.Join(binDir, configPath)
			if _, err := os.Stat(candidate); err == nil {
				configPath = candidate
			}
		}
	}

	cfg, err := config.Load(configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	if cfg.APIToken != "" {
		log.Println("API token authentication enabled")
	} else {
		log.Println("WARNING: API token not configured, authentication disabled")
	}

	hub := model.NewModelHub(cfg)

	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Recovery())

	auth := middleware.AuthMiddleware(cfg.APIToken)
	proxyHandler := handler.Proxy(hub)

	v1 := router.Group("/v1", auth)
	{
		if hub.HasRerank() {
			v1.POST("/rerank", handler.Rerank(hub))
		}
		if hub.HasEmbedding() {
			v1.POST("/embeddings", handler.CreateEmbeddings(hub))
		}
		if hub.HasLLM() {
			v1.POST("/chat/completions", handler.ChatCompletions(hub))
		}
		v1.GET("/health", handler.Health(hub))
		v1.GET("/models", handler.ListModels(hub))
	}

	router.NoRoute(func(c *gin.Context) {
		if len(c.Request.URL.Path) >= 3 && c.Request.URL.Path[:3] == "/v1" {
			auth(c)
			if c.IsAborted() {
				return
			}
		}
		proxyHandler(c)
	})

	srv := &http.Server{
		Addr:    fmt.Sprintf("%s:%d", host, port),
		Handler: router,
	}

	go func() {
		log.Printf("Starting server on %s:%d", host, port)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server failed: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Printf("Server shutdown error: %v", err)
	}

	hub.Shutdown()
	log.Println("Server exited")
}
