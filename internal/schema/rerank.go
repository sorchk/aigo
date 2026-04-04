package schema

type Document struct {
	Text string `json:"text"`
}

type RerankResult struct {
	Index          int       `json:"index"`
	Document       *Document `json:"document,omitempty"`
	RelevanceScore float64   `json:"relevance_score"`
}

type RerankRequest struct {
	Model           *string  `json:"model,omitempty"`
	Query           string   `json:"query"`
	Documents       []string `json:"documents"`
	TopN            *int     `json:"top_n,omitempty"`
	TopK            *int     `json:"top_k,omitempty"`
	ReturnDocuments *bool    `json:"return_documents,omitempty"`
}

type RerankMetaAPIVersion struct {
	Version string `json:"version"`
}

type RerankMetaBilledUnits struct {
	SearchUnits int `json:"search_units"`
}

type RerankMetaTokens struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type RerankMeta struct {
	APIVersion  RerankMetaAPIVersion  `json:"api_version"`
	BilledUnits RerankMetaBilledUnits `json:"billed_units"`
	Tokens      RerankMetaTokens      `json:"tokens"`
}

type RerankResponse struct {
	ID      string         `json:"id"`
	Model   string         `json:"model"`
	Results []RerankResult `json:"results"`
	Usage   UsageInfo      `json:"usage"`
	Meta    RerankMeta     `json:"meta"`
}
