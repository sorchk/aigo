package handler

import (
	"os"
	"os/exec"
	"runtime"
	"strings"
)

func getSystemInfo() map[string]interface{} {
	info := map[string]interface{}{
		"cpu_cores": runtime.NumCPU(),
		"os":        runtime.GOOS,
		"arch":      runtime.GOARCH,
	}

	gpus := detectGPUs()
	if len(gpus) > 0 {
		info["gpus"] = gpus
	}

	return info
}

type gpuInfo struct {
	Name     string `json:"name"`
	Vendor   string `json:"vendor,omitempty"`
	Memory   string `json:"memory,omitempty"`
	Driver   string `json:"driver,omitempty"`
	Platform string `json:"platform,omitempty"`
}

func detectGPUs() []gpuInfo {
	var gpus []gpuInfo

	if cudaGPUs := detectCUDAGPUs(); len(cudaGPUs) > 0 {
		gpus = append(gpus, cudaGPUs...)
	}
	if rocmGPUs := detectROCmGPUs(); len(rocmGPUs) > 0 {
		gpus = append(gpus, rocmGPUs...)
	}
	if vulkanGPUs := detectVulkanGPUs(); len(vulkanGPUs) > 0 {
		gpus = append(gpus, vulkanGPUs...)
	}

	return gpus
}

func detectCUDAGPUs() []gpuInfo {
	path, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return nil
	}

	out, err := exec.Command(path, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil
	}

	var gpus []gpuInfo
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line == "" {
			continue
		}
		parts := strings.Split(line, ", ")
		g := gpuInfo{Platform: "cuda"}
		if len(parts) >= 1 {
			g.Name = strings.TrimSpace(parts[0])
		}
		if len(parts) >= 2 {
			g.Memory = strings.TrimSpace(parts[1]) + " MiB"
		}
		if len(parts) >= 3 {
			g.Driver = strings.TrimSpace(parts[2])
		}
		g.Vendor = "nvidia"
		gpus = append(gpus, g)
	}
	return gpus
}

func detectROCmGPUs() []gpuInfo {
	if _, err := os.Stat("/dev/kfd"); os.IsNotExist(err) {
		if _, err := os.Stat("/opt/rocm/bin/rocm-smi"); err != nil {
			return nil
		}
	}

	path, err := exec.LookPath("rocm-smi")
	if err != nil {
		return nil
	}

	out, err := exec.Command(path, "--showproductname", "--csv").Output()
	if err != nil {
		return nil
	}

	var gpus []gpuInfo
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line == "" || strings.HasPrefix(line, "device") {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) < 2 {
			continue
		}
		gpus = append(gpus, gpuInfo{
			Name:     strings.TrimSpace(parts[len(parts)-1]),
			Vendor:   "amd",
			Platform: "rocm",
		})
	}
	return gpus
}

func detectVulkanGPUs() []gpuInfo {
	path, err := exec.LookPath("vulkaninfo")
	if err != nil {
		return nil
	}

	out, err := exec.Command(path, "--summary").Output()
	if err != nil {
		return nil
	}

	var gpus []gpuInfo
	var currentName, currentType string
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "deviceName") {
			currentName = strings.TrimPrefix(line, "deviceName")
			currentName = strings.TrimSpace(strings.TrimLeft(currentName, "= "))
		}
		if strings.HasPrefix(line, "deviceType") {
			currentType = strings.TrimPrefix(line, "deviceType")
			currentType = strings.TrimSpace(strings.TrimLeft(currentType, "= "))
		}
		if currentName != "" && currentType != "" {
			vendor := "unknown"
			if strings.Contains(strings.ToLower(currentName), "nvidia") {
				vendor = "nvidia"
			} else if strings.Contains(strings.ToLower(currentName), "amd") ||
				strings.Contains(strings.ToLower(currentName), "radeon") {
				vendor = "amd"
			} else if strings.Contains(strings.ToLower(currentName), "intel") {
				vendor = "intel"
			} else if strings.Contains(strings.ToLower(currentName), "apple") ||
				strings.Contains(strings.ToLower(currentName), "m1") ||
				strings.Contains(strings.ToLower(currentName), "m2") ||
				strings.Contains(strings.ToLower(currentName), "m3") ||
				strings.Contains(strings.ToLower(currentName), "m4") {
				vendor = "apple"
			} else if strings.Contains(strings.ToLower(currentName), "adreno") ||
				strings.Contains(strings.ToLower(currentName), "mali") {
				vendor = "arm"
			}
			gpus = append(gpus, gpuInfo{
				Name:     currentName,
				Vendor:   vendor,
				Platform: "vulkan",
			})
			currentName = ""
			currentType = ""
		}
	}
	return gpus
}
