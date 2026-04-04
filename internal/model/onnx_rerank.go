package model

import (
	"fmt"
	"math"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/sction-ai/go/internal/config"
)

type OnnxRerankService struct {
	*OnnxSession
}

func NewOnnxRerankService(cfg config.ModelConfig, onnxLibPath string) (*OnnxRerankService, error) {
	outputName := cfg.OnnxOutputName
	if outputName == "" {
		outputName = "logits"
	}
	session, err := NewOnnxSession(cfg, onnxLibPath, []string{"input_ids", "attention_mask"}, []string{outputName})
	if err != nil {
		return nil, err
	}
	return &OnnxRerankService{OnnxSession: session}, nil
}

type RerankResult struct {
	Index          int
	RelevanceScore float64
}

func (s *OnnxRerankService) Rerank(query string, documents []string, topN int) ([]RerankResult, int, error) {
	type pairResult struct {
		index  int
		score  float64
		tokens int
	}

	results := make([]pairResult, len(documents))
	var batchSize = 8

	for i := 0; i < len(documents); i += batchSize {
		end := i + batchSize
		if end > len(documents) {
			end = len(documents)
		}
		batch := documents[i:end]

		inputIDs, attentionMasks, totalTokens, err := s.encodeRerankBatch(query, batch)
		if err != nil {
			return nil, 0, fmt.Errorf("encode batch: %w", err)
		}

		scores, err := s.runRerankInference(inputIDs, attentionMasks, len(batch))
		if err != nil {
			return nil, 0, fmt.Errorf("run inference: %w", err)
		}

		for j := 0; j < len(batch); j++ {
			results[i+j] = pairResult{
				index:  i + j,
				score:  scores[j],
				tokens: totalTokens / len(batch),
			}
		}
	}

	allResults := make([]RerankResult, len(results))
	for i, r := range results {
		allResults[i] = RerankResult{
			Index:          r.index,
			RelevanceScore: r.score,
		}
	}

	if topN > 0 && topN < len(allResults) {
		allResults = allResults[:topN]
	}

	totalTokens := 0
	for _, r := range results {
		totalTokens += r.tokens
	}

	return allResults, totalTokens, nil
}

func (s *OnnxRerankService) encodeRerankPair(query, doc string) ([]int64, []int64, int, error) {
	queryIDs, err := s.tokenizer.Encode(query)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("encode query: %w", err)
	}
	docIDs, err := s.tokenizer.Encode(doc)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("encode document: %w", err)
	}

	const maxLen = 8192
	if len(queryIDs)+len(docIDs)+4 > maxLen {
		remaining := maxLen - len(queryIDs) - 4
		if remaining > 0 {
			docIDs = docIDs[:remaining]
		} else {
			queryIDs = queryIDs[:maxLen/2-2]
			docIDs = docIDs[:maxLen/2-2]
		}
	}

	ids := make([]int64, 0, len(queryIDs)+len(docIDs)+4)
	ids = append(ids, 0)
	for _, id := range queryIDs {
		ids = append(ids, int64(id))
	}
	ids = append(ids, 2, 2)
	for _, id := range docIDs {
		ids = append(ids, int64(id))
	}
	ids = append(ids, 2)

	mask := make([]int64, len(ids))
	for i := range mask {
		mask[i] = 1
	}

	return ids, mask, len(ids), nil
}

func (s *OnnxRerankService) encodeRerankBatch(query string, docs []string) ([][]int64, [][]int64, int, error) {
	allIDs := make([][]int64, len(docs))
	allMasks := make([][]int64, len(docs))
	totalTokens := 0

	maxSeqLen := 0
	for i, doc := range docs {
		ids, mask, n, err := s.encodeRerankPair(query, doc)
		if err != nil {
			return nil, nil, 0, err
		}
		allIDs[i] = ids
		allMasks[i] = mask
		totalTokens += n
		if len(ids) > maxSeqLen {
			maxSeqLen = len(ids)
		}
	}

	for i := range allIDs {
		padLen := maxSeqLen - len(allIDs[i])
		if padLen > 0 {
			padded := make([]int64, maxSeqLen)
			copy(padded, allIDs[i])
			allIDs[i] = padded

			paddedMask := make([]int64, maxSeqLen)
			for j := range allMasks[i] {
				paddedMask[j] = allMasks[i][j]
			}
			allMasks[i] = paddedMask
		}
	}

	return allIDs, allMasks, totalTokens, nil
}

func (s *OnnxRerankService) runRerankInference(inputIDs, attentionMasks [][]int64, batchSize int) ([]float64, error) {
	seqLen := len(inputIDs[0])

	flatInputIDs := make([]int64, batchSize*seqLen)
	flatAttentionMasks := make([]int64, batchSize*seqLen)
	for i := 0; i < batchSize; i++ {
		copy(flatInputIDs[i*seqLen:], inputIDs[i])
		copy(flatAttentionMasks[i*seqLen:], attentionMasks[i])
	}

	inputShape := ort.NewShape(int64(batchSize), int64(seqLen))

	inputIDsTensor, err := ort.NewTensor(inputShape, flatInputIDs)
	if err != nil {
		return nil, fmt.Errorf("create input_ids tensor: %w", err)
	}
	defer inputIDsTensor.Destroy()

	attentionMaskTensor, err := ort.NewTensor(inputShape, flatAttentionMasks)
	if err != nil {
		return nil, fmt.Errorf("create attention_mask tensor: %w", err)
	}
	defer attentionMaskTensor.Destroy()

	outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(batchSize), 1))
	if err != nil {
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	err = s.Run(
		[]ort.ArbitraryTensor{inputIDsTensor, attentionMaskTensor},
		[]ort.ArbitraryTensor{outputTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("onnx run: %w", err)
	}

	outputData := outputTensor.GetData()
	scores := make([]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		scores[i] = sigmoid(float64(outputData[i]))
	}

	return scores, nil
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
