package model

import (
	"fmt"
	"math"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/sction-ai/go/internal/config"
)

type OnnxEmbeddingService struct {
	*OnnxSession
}

func NewOnnxEmbeddingService(cfg config.ModelConfig, onnxLibPath string) (*OnnxEmbeddingService, error) {
	outputName := cfg.OnnxOutputName
	if outputName == "" {
		outputName = "logits"
	}
	session, err := NewOnnxSession(cfg, onnxLibPath, []string{"input_ids", "attention_mask"}, []string{outputName})
	if err != nil {
		return nil, err
	}
	return &OnnxEmbeddingService{OnnxSession: session}, nil
}

func (s *OnnxEmbeddingService) Embed(texts []string) ([][]float32, int, error) {
	inputIDs, attentionMasks, totalTokens, err := s.encodeTexts(texts)
	if err != nil {
		return nil, 0, fmt.Errorf("encode texts: %w", err)
	}

	batchSize := len(texts)
	logits, err := s.runEmbedInference(inputIDs, attentionMasks, batchSize)
	if err != nil {
		return nil, 0, fmt.Errorf("run inference: %w", err)
	}

	seqLen := len(inputIDs[0])
	embeddings := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		embeddings[i] = meanPool(logits[i], attentionMasks[i], seqLen)
		normalize(embeddings[i])
	}

	return embeddings, totalTokens, nil
}

func (s *OnnxEmbeddingService) encodeTexts(texts []string) ([][]int64, [][]int64, int, error) {
	allIDs := make([][]int64, len(texts))
	allMasks := make([][]int64, len(texts))
	totalTokens := 0

	maxSeqLen := 0
	for i, text := range texts {
		ids, err := s.tokenizer.Encode(text)
		if err != nil {
			return nil, nil, 0, fmt.Errorf("encode text %d: %w", i, err)
		}

		tokenIDs := make([]int64, len(ids)+2)
		tokenIDs[0] = 0
		for j, id := range ids {
			tokenIDs[j+1] = int64(id)
		}
		tokenIDs[len(ids)+1] = 2

		mask := make([]int64, len(tokenIDs))
		for j := range mask {
			mask[j] = 1
		}

		allIDs[i] = tokenIDs
		allMasks[i] = mask
		totalTokens += len(tokenIDs)
		if len(tokenIDs) > maxSeqLen {
			maxSeqLen = len(tokenIDs)
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

func (s *OnnxEmbeddingService) runEmbedInference(inputIDs, attentionMasks [][]int64, batchSize int) ([][]float32, error) {
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

	hiddenSize := s.modelCfg.OnnxHiddenSize
	outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(batchSize), int64(seqLen), int64(hiddenSize)))
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

	result := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		row := make([]float32, seqLen*hiddenSize)
		start := i * seqLen * hiddenSize
		copy(row, outputData[start:start+seqLen*hiddenSize])
		result[i] = row
	}

	return result, nil
}

func meanPool(lastHidden []float32, mask []int64, seqLen int) []float32 {
	if len(lastHidden) == 0 {
		return lastHidden
	}
	hiddenSize := len(lastHidden) / seqLen
	result := make([]float32, hiddenSize)

	var count float32
	for t := 0; t < seqLen; t++ {
		if mask[t] == 0 {
			continue
		}
		count++
		for j := 0; j < hiddenSize; j++ {
			result[j] += lastHidden[t*hiddenSize+j]
		}
	}

	if count > 0 {
		for j := range result {
			result[j] /= count
		}
	}

	return result
}

func normalize(vec []float32) {
	var norm float64
	for _, v := range vec {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		return
	}
	for i := range vec {
		vec[i] /= float32(norm)
	}
}
