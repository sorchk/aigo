package conf

import "embed"

//go:embed config.yaml
var ConfigFiles embed.FS
