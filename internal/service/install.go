package service

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const serviceName = "aigo"

type InstallOptions struct {
	User         string
	WorkingDir   string
	ConfigPath   string
	BinaryPath   string
	Host         string
	Port         int
	AfterInstall bool
}

func resolveBinaryPath() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", fmt.Errorf("resolve executable path: %w", err)
	}
	return filepath.EvalSymlinks(exe)
}

func detectUser() string {
	if u := os.Getenv("SUDO_USER"); u != "" {
		return u
	}
	return os.Getenv("USER")
}

func systemdUnit(opts *InstallOptions) string {
	envLines := []string{}
	if opts.Host != "" && opts.Host != "0.0.0.0" {
		envLines = append(envLines, fmt.Sprintf("Environment=HOST=%s", opts.Host))
	}
	if opts.Port != 0 && opts.Port != 8000 {
		envLines = append(envLines, fmt.Sprintf("Environment=PORT=%d", opts.Port))
	}
	if opts.ConfigPath != "" && opts.ConfigPath != "conf/config.yaml" {
		envLines = append(envLines, fmt.Sprintf("Environment=CONFIG_PATH=%s", opts.ConfigPath))
	}
	envBlock := strings.Join(envLines, "\n")

	extraEnv := ""
	if envBlock != "" {
		extraEnv = "\n" + envBlock
	}

	return fmt.Sprintf(`[Unit]
 Description=Go AI Model Hub Server
After=network.target

[Service]
Type=simple
User=%s
WorkingDirectory=%s
ExecStart=%s run
Restart=on-failure
RestartSec=5%s

[Install]
WantedBy=multi-user.target
`, opts.User, opts.WorkingDir, opts.BinaryPath, extraEnv)
}

func Install(opts *InstallOptions) error {
	if os.Getuid() != 0 {
		return fmt.Errorf("install requires root privileges, run with sudo")
	}

	binPath, err := resolveBinaryPath()
	if err != nil {
		return err
	}
	opts.BinaryPath = binPath

	if opts.WorkingDir == "" {
		opts.WorkingDir = filepath.Dir(binPath)
	}
	if opts.User == "" {
		opts.User = detectUser()
	}

	unitContent := systemdUnit(opts)

	unitPath := fmt.Sprintf("/etc/systemd/system/%s.service", serviceName)
	if err := os.WriteFile(unitPath, []byte(unitContent), 0644); err != nil {
		return fmt.Errorf("write unit file: %w", err)
	}
	fmt.Printf("Created %s\n", unitPath)

	commands := [][]string{
		{"systemctl", "daemon-reload"},
		{"systemctl", "enable", serviceName},
		{"systemctl", "start", serviceName},
	}

	for _, cmd := range commands {
		if err := runCommand(cmd[0], cmd[1:]...); err != nil {
			return fmt.Errorf("run %s: %w", strings.Join(cmd, " "), err)
		}
		fmt.Printf("Executed: %s\n", strings.Join(cmd, " "))
	}

	fmt.Printf("\nService '%s' installed and started successfully.\n", serviceName)
	fmt.Printf("Useful commands:\n")
	fmt.Printf("  systemctl status %s\n", serviceName)
	fmt.Printf("  systemctl stop %s\n", serviceName)
	fmt.Printf("  systemctl restart %s\n", serviceName)
	fmt.Printf("  journalctl -u %s -f\n", serviceName)
	return nil
}

func Uninstall() error {
	if os.Getuid() != 0 {
		return fmt.Errorf("uninstall requires root privileges, run with sudo")
	}

	commands := [][]string{
		{"systemctl", "stop", serviceName},
		{"systemctl", "disable", serviceName},
	}

	for _, cmd := range commands {
		if err := runCommand(cmd[0], cmd[1:]...); err != nil {
			fmt.Printf("Warning: %s: %v\n", strings.Join(cmd, " "), err)
		} else {
			fmt.Printf("Executed: %s\n", strings.Join(cmd, " "))
		}
	}

	unitPath := fmt.Sprintf("/etc/systemd/system/%s.service", serviceName)
	if err := os.Remove(unitPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("remove unit file: %w", err)
	}
	fmt.Printf("Removed %s\n", unitPath)

	if err := runCommand("systemctl", "daemon-reload"); err != nil {
		fmt.Printf("Warning: daemon-reload: %v\n", err)
	}

	fmt.Printf("\nService '%s' uninstalled successfully.\n", serviceName)
	return nil
}

func runCommand(name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
