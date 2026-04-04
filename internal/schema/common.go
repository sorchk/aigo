package schema

type ModelNotReadyError struct {
	ModelName string
}

func (e *ModelNotReadyError) Error() string {
	return "Model '" + e.ModelName + "' is loading, please retry later"
}

type UsageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
	Type    string `json:"type"`
	Loaded  bool   `json:"loaded"`
}
