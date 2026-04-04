package handler

import (
	"runtime"
)

func getSystemInfo() map[string]interface{} {
	return map[string]interface{}{
		"cpu_cores": runtime.NumCPU(),
		"os":        runtime.GOOS,
		"arch":      runtime.GOARCH,
	}
}
