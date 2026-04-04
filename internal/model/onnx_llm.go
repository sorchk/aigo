package model

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"strings"

	sp "github.com/tggo/goSentencePiece"
	ort "github.com/yalue/onnxruntime_go"

	"github.com/sction-ai/go/internal/config"
)

const vocabSize = 248320
const hiddenSize = 2560

var fullAttentionLayers = []int{3, 7, 11, 15, 19, 23, 27, 31}
var linearAttentionLayers = []int{0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30}

type OnnxLLMService struct {
	embedSession   *OnnxSession
	decoderSession *OnnxSession
	visionSession  *OnnxSession
	modelCfg       config.ModelConfig
	eosTokenID     int64
}

func NewOnnxLLMService(cfg config.ModelConfig, onnxLibPath string) (*OnnxLLMService, error) {
	embedCfg := cfg
	if embedCfg.OnnxPath == "" || !strings.Contains(embedCfg.OnnxPath, "embed_tokens") {
		candidates := []string{}
		for _, f := range embedCfg.OnnxFiles {
			if strings.Contains(f, "embed_tokens") {
				candidates = append(candidates, f)
				break
			}
		}
		if len(candidates) == 0 {
			return nil, fmt.Errorf("embed_tokens model not found for %s", cfg.Name)
		}
		embedCfg.OnnxPath = candidates[0]
		embedCfg.OnnxFiles = candidates
	}

	embedSession, err := NewOnnxSession(embedCfg, onnxLibPath, []string{"input_ids"}, []string{"inputs_embeds"})
	if err != nil {
		return nil, fmt.Errorf("create embed session: %w", err)
	}

	decoderCfg := cfg
	if decoderCfg.OnnxPath == "" || strings.Contains(decoderCfg.OnnxPath, "embed_tokens") {
		candidates := []string{}
		for _, f := range decoderCfg.OnnxFiles {
			if strings.Contains(f, "decoder_model") && !strings.Contains(f, "embed_tokens") {
				candidates = append(candidates, f)
			}
		}
		if len(candidates) == 0 {
			embedSession.Destroy()
			return nil, fmt.Errorf("decoder model not found for %s", cfg.Name)
		}
		decoderCfg.OnnxPath = candidates[0]
		decoderCfg.OnnxFiles = candidates
	}

	inputNames := buildDecoderInputNames()
	outputNames := buildDecoderOutputNames()

	decoderSession, err := NewOnnxSession(decoderCfg, onnxLibPath, inputNames, outputNames)
	if err != nil {
		embedSession.Destroy()
		return nil, fmt.Errorf("create decoder session: %w", err)
	}

	var visionSession *OnnxSession
	for _, f := range cfg.OnnxFiles {
		if strings.Contains(f, "vision_encoder") {
			visionCfg := cfg
			visionCfg.OnnxPath = f
			visionCfg.OnnxFiles = []string{f}
			vs, vErr := NewOnnxSession(visionCfg, onnxLibPath,
				[]string{"pixel_values", "image_grid_thw"},
				[]string{"image_features"},
			)
			if vErr != nil {
				log.Printf("Warning: failed to load vision encoder for %s: %v", cfg.Name, vErr)
			} else {
				visionSession = vs
			}
			break
		}
	}

	eosTokenID := int64(248044)

	log.Printf("ONNX LLM service created for model %s", cfg.Name)

	return &OnnxLLMService{
		embedSession:   embedSession,
		decoderSession: decoderSession,
		visionSession:  visionSession,
		modelCfg:       cfg,
		eosTokenID:     eosTokenID,
	}, nil
}

func (s *OnnxLLMService) Destroy() {
	if s.embedSession != nil {
		s.embedSession.Destroy()
	}
	if s.decoderSession != nil {
		s.decoderSession.Destroy()
	}
	if s.visionSession != nil {
		s.visionSession.Destroy()
	}
}

func (s *OnnxLLMService) HasVisionEncoder() bool {
	return s.visionSession != nil
}

func (s *OnnxLLMService) Tokenizer() *sp.Tokenizer {
	return s.embedSession.Tokenizer()
}

func (s *OnnxLLMService) EncodeImage(pixelValues []float32, gridTHW []int64, numFeatures int) ([][]float32, error) {
	if s.visionSession == nil {
		return nil, fmt.Errorf("vision encoder not loaded")
	}

	numPatches := int64(len(pixelValues)) / visionPatchDim

	pixelTensor, err := ort.NewTensor(ort.NewShape(numPatches, visionPatchDim), pixelValues)
	if err != nil {
		return nil, fmt.Errorf("create pixel tensor: %w", err)
	}
	defer pixelTensor.Destroy()

	numImages := int64(len(gridTHW)) / 3
	gridTensor, err := ort.NewTensor(ort.NewShape(numImages, 3), gridTHW)
	if err != nil {
		return nil, fmt.Errorf("create grid tensor: %w", err)
	}
	defer gridTensor.Destroy()

	featuresTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(numFeatures), hiddenSize))
	if err != nil {
		return nil, fmt.Errorf("create features tensor: %w", err)
	}
	defer featuresTensor.Destroy()

	err = s.visionSession.Run(
		[]ort.ArbitraryTensor{pixelTensor, gridTensor},
		[]ort.ArbitraryTensor{featuresTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("vision encoder run: %w", err)
	}

	data := featuresTensor.GetData()
	result := make([][]float32, numFeatures)
	for i := 0; i < numFeatures; i++ {
		row := make([]float32, hiddenSize)
		copy(row, data[i*hiddenSize:(i+1)*hiddenSize])
		result[i] = row
	}
	return result, nil
}

const imagePadTokenID = int64(248056)

func (s *OnnxLLMService) GenerateMultimodal(inputIDs []int64, imageFeatures [][]float32, params GenerateParams) ([]int64, error) {
	if params.MaxTokens <= 0 {
		params.MaxTokens = 512
	}
	if params.Temperature <= 0 {
		params.Temperature = 0.6
	}

	textEmbeddings, err := s.getEmbeddings(inputIDs)
	if err != nil {
		return nil, fmt.Errorf("get text embeddings: %w", err)
	}

	embeddings := make([][]float32, len(textEmbeddings))
	featureIdx := 0
	for i, id := range inputIDs {
		if id == imagePadTokenID && featureIdx < len(imageFeatures) {
			embeddings[i] = imageFeatures[featureIdx]
			featureIdx++
		} else {
			embeddings[i] = textEmbeddings[i]
		}
	}

	return s.generateFromEmbeddings(embeddings, params)
}

func (s *OnnxLLMService) generateFromEmbeddings(embeddings [][]float32, params GenerateParams) ([]int64, error) {
	pastKV, err := s.initKVCache()
	if err != nil {
		return nil, fmt.Errorf("init kv cache: %w", err)
	}

	seqLen := int64(len(embeddings))
	totalSeqLen := seqLen

	generatedTokens := make([]int64, 0, params.MaxTokens)

	for step := 0; step < params.MaxTokens; step++ {
		var stepEmbeddings [][]float32
		var stepPosIDs []int64

		if step == 0 {
			stepEmbeddings = embeddings
			stepPosIDs = make([]int64, seqLen)
			for i := range stepPosIDs {
				stepPosIDs[i] = int64(i)
			}
		} else {
			stepEmbeddings = [][]float32{embeddings[len(embeddings)-1]}
			stepPosIDs = []int64{totalSeqLen - 1}
		}

		stepSeqLen := int64(len(stepPosIDs))

		attentionMask := make([]int64, totalSeqLen)
		for i := range attentionMask {
			attentionMask[i] = 1
		}

		logits, newPastKV, err := s.runDecoderStep(stepEmbeddings, attentionMask, stepPosIDs, pastKV, stepSeqLen, totalSeqLen)
		if err != nil {
			return nil, fmt.Errorf("decoder step %d: %w", step, err)
		}

		pastKV = newPastKV

		nextToken := s.sampleToken(logits, params)
		generatedTokens = append(generatedTokens, nextToken)
		totalSeqLen++

		if nextToken == s.eosTokenID {
			break
		}

		if step+1 < params.MaxTokens {
			embeddings, err = s.getEmbeddings([]int64{nextToken})
			if err != nil {
				return nil, fmt.Errorf("get next token embedding: %w", err)
			}
		}

		if params.Callback != nil {
			if err := params.Callback(nextToken); err != nil {
				break
			}
		}
	}

	return generatedTokens, nil
}

func (s *OnnxLLMService) Generate(inputIDs []int64, params GenerateParams) ([]int64, error) {
	if params.MaxTokens <= 0 {
		params.MaxTokens = 512
	}
	if params.Temperature <= 0 {
		params.Temperature = 0.6
	}

	embeddings, err := s.getEmbeddings(inputIDs)
	if err != nil {
		return nil, fmt.Errorf("get embeddings: %w", err)
	}

	pastKV, err := s.initKVCache()
	if err != nil {
		return nil, fmt.Errorf("init kv cache: %w", err)
	}

	seqLen := int64(len(inputIDs))
	totalSeqLen := seqLen

	generatedTokens := make([]int64, 0, params.MaxTokens)

	for step := 0; step < params.MaxTokens; step++ {
		var stepEmbeddings [][]float32
		var stepPosIDs []int64

		if step == 0 {
			stepEmbeddings = embeddings
			stepPosIDs = make([]int64, seqLen)
			for i := range stepPosIDs {
				stepPosIDs[i] = int64(i)
			}
		} else {
			stepEmbeddings = [][]float32{embeddings[len(embeddings)-1]}
			stepPosIDs = []int64{totalSeqLen - 1}
		}

		stepSeqLen := int64(len(stepPosIDs))

		attentionMask := make([]int64, totalSeqLen)
		for i := range attentionMask {
			attentionMask[i] = 1
		}

		logits, newPastKV, err := s.runDecoderStep(stepEmbeddings, attentionMask, stepPosIDs, pastKV, stepSeqLen, totalSeqLen)
		if err != nil {
			return nil, fmt.Errorf("decoder step %d: %w", step, err)
		}

		pastKV = newPastKV

		nextToken := s.sampleToken(logits, params)
		generatedTokens = append(generatedTokens, nextToken)
		totalSeqLen++

		if nextToken == s.eosTokenID {
			break
		}

		if step+1 < params.MaxTokens {
			embeddings, err = s.getEmbeddings([]int64{nextToken})
			if err != nil {
				return nil, fmt.Errorf("get next token embedding: %w", err)
			}
		}

		if params.Callback != nil {
			if err := params.Callback(nextToken); err != nil {
				break
			}
		}
	}

	return generatedTokens, nil
}

type GenerateParams struct {
	MaxTokens   int
	Temperature float64
	TopP        float64
	TopK        int
	Callback    func(tokenID int64) error
}

func (s *OnnxLLMService) getEmbeddings(ids []int64) ([][]float32, error) {
	seqLen := int64(len(ids))
	shape := ort.NewShape(1, seqLen)

	inputTensor, err := ort.NewTensor(shape, ids)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, seqLen, hiddenSize))
	if err != nil {
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	err = s.embedSession.Run(
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("embed run: %w", err)
	}

	data := outputTensor.GetData()
	result := make([][]float32, seqLen)
	for i := int64(0); i < seqLen; i++ {
		row := make([]float32, hiddenSize)
		copy(row, data[i*hiddenSize:(i+1)*hiddenSize])
		result[i] = row
	}
	return result, nil
}

type kvCache struct {
	convStates      map[int][]float32
	recurrentStates map[int][]float32
	keyStates       map[int][]float32
	valueStates     map[int][]float32
	keySeqLens      map[int]int64
}

func (s *OnnxLLMService) initKVCache() (*kvCache, error) {
	cache := &kvCache{
		convStates:      make(map[int][]float32),
		recurrentStates: make(map[int][]float32),
		keyStates:       make(map[int][]float32),
		valueStates:     make(map[int][]float32),
		keySeqLens:      make(map[int]int64),
	}

	for _, layer := range linearAttentionLayers {
		cache.convStates[layer] = make([]float32, 8192*4)
		cache.recurrentStates[layer] = make([]float32, 32*128*128)
	}

	for _, layer := range fullAttentionLayers {
		cache.keyStates[layer] = nil
		cache.valueStates[layer] = nil
		cache.keySeqLens[layer] = 0
	}

	return cache, nil
}

func (s *OnnxLLMService) runDecoderStep(
	embeddings [][]float32,
	attentionMask []int64,
	positionIDs []int64,
	pastKV *kvCache,
	seqLen int64,
	totalSeqLen int64,
) ([]float32, *kvCache, error) {
	flatEmbeddings := make([]float32, len(embeddings)*hiddenSize)
	for i, emb := range embeddings {
		copy(flatEmbeddings[i*hiddenSize:], emb)
	}

	embedShape := ort.NewShape(1, seqLen, hiddenSize)
	embedTensor, err := ort.NewTensor(embedShape, flatEmbeddings)
	if err != nil {
		return nil, nil, fmt.Errorf("create embed tensor: %w", err)
	}
	defer embedTensor.Destroy()

	maskShape := ort.NewShape(1, totalSeqLen)
	maskTensor, err := ort.NewTensor(maskShape, attentionMask)
	if err != nil {
		return nil, nil, fmt.Errorf("create mask tensor: %w", err)
	}
	defer maskTensor.Destroy()

	posShape := ort.NewShape(3, 1, seqLen)
	posData := make([]int64, 3*seqLen)
	for i := int64(0); i < seqLen; i++ {
		posData[i] = positionIDs[i]
		posData[seqLen+i] = positionIDs[i]
		posData[2*seqLen+i] = positionIDs[i]
	}
	posTensor, err := ort.NewTensor(posShape, posData)
	if err != nil {
		return nil, nil, fmt.Errorf("create pos tensor: %w", err)
	}
	defer posTensor.Destroy()

	inputs := []ort.ArbitraryTensor{embedTensor, maskTensor, posTensor}

	pastConvTensors := make(map[int]ort.ArbitraryTensor)
	pastRecurrentTensors := make(map[int]ort.ArbitraryTensor)
	pastKeyTensors := make(map[int]ort.ArbitraryTensor)
	pastValueTensors := make(map[int]ort.ArbitraryTensor)

	for _, layer := range linearAttentionLayers {
		convData := pastKV.convStates[layer]
		convTensor, err := ort.NewTensor(ort.NewShape(1, 8192, 4), convData)
		if err != nil {
			return nil, nil, fmt.Errorf("create past_conv.%d tensor: %w", layer, err)
		}
		pastConvTensors[layer] = convTensor
		inputs = append(inputs, convTensor)

		recData := pastKV.recurrentStates[layer]
		recTensor, err := ort.NewTensor(ort.NewShape(1, 32, 128, 128), recData)
		if err != nil {
			return nil, nil, fmt.Errorf("create past_recurrent.%d tensor: %w", layer, err)
		}
		pastRecurrentTensors[layer] = recTensor
		inputs = append(inputs, recTensor)
	}

	for _, layer := range fullAttentionLayers {
		pastSeqLen := pastKV.keySeqLens[layer]

		if pastSeqLen > 0 && pastKV.keyStates[layer] != nil {
			keyTensor, err := ort.NewTensor(ort.NewShape(1, 4, pastSeqLen, 256), pastKV.keyStates[layer])
			if err != nil {
				return nil, nil, fmt.Errorf("create past_key.%d tensor: %w", layer, err)
			}
			pastKeyTensors[layer] = keyTensor
			inputs = append(inputs, keyTensor)

			valueTensor, err := ort.NewTensor(ort.NewShape(1, 4, pastSeqLen, 256), pastKV.valueStates[layer])
			if err != nil {
				return nil, nil, fmt.Errorf("create past_value.%d tensor: %w", layer, err)
			}
			pastValueTensors[layer] = valueTensor
			inputs = append(inputs, valueTensor)
		} else {
			emptyKey, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 4, 0, 256))
			if err != nil {
				return nil, nil, fmt.Errorf("create empty past_key.%d tensor: %w", layer, err)
			}
			pastKeyTensors[layer] = emptyKey
			inputs = append(inputs, emptyKey)

			emptyValue, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 4, 0, 256))
			if err != nil {
				return nil, nil, fmt.Errorf("create empty past_value.%d tensor: %w", layer, err)
			}
			pastValueTensors[layer] = emptyValue
			inputs = append(inputs, emptyValue)
		}
	}

	outputNames := buildDecoderOutputNames()
	outputs := make([]ort.ArbitraryTensor, len(outputNames))

	logitsTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, seqLen, vocabSize))
	if err != nil {
		return nil, nil, fmt.Errorf("create logits tensor: %w", err)
	}
	outputs[0] = logitsTensor

	presentConvTensors := make(map[int]ort.ArbitraryTensor)
	presentRecurrentTensors := make(map[int]ort.ArbitraryTensor)
	presentKeyTensors := make(map[int]ort.ArbitraryTensor)
	presentValueTensors := make(map[int]ort.ArbitraryTensor)

	outIdx := 1
	for _, layer := range linearAttentionLayers {
		conv, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 8192, 4))
		if err != nil {
			return nil, nil, fmt.Errorf("create present_conv.%d tensor: %w", layer, err)
		}
		presentConvTensors[layer] = conv
		outputs[outIdx] = conv
		outIdx++

		rec, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 32, 128, 128))
		if err != nil {
			return nil, nil, fmt.Errorf("create present_recurrent.%d tensor: %w", layer, err)
		}
		presentRecurrentTensors[layer] = rec
		outputs[outIdx] = rec
		outIdx++
	}

	for _, layer := range fullAttentionLayers {
		newSeqLen := pastKV.keySeqLens[layer] + seqLen
		key, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 4, newSeqLen, 256))
		if err != nil {
			return nil, nil, fmt.Errorf("create present.%d.key tensor: %w", layer, err)
		}
		presentKeyTensors[layer] = key
		outputs[outIdx] = key
		outIdx++

		value, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 4, newSeqLen, 256))
		if err != nil {
			return nil, nil, fmt.Errorf("create present.%d.value tensor: %w", layer, err)
		}
		presentValueTensors[layer] = value
		outputs[outIdx] = value
		outIdx++
	}

	err = s.decoderSession.Run(inputs, outputs)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder run: %w", err)
	}

	logitsData := logitsTensor.GetData()
	lastTokenLogits := make([]float32, vocabSize)
	if seqLen == 1 {
		copy(lastTokenLogits, logitsData)
	} else {
		start := (seqLen - 1) * vocabSize
		copy(lastTokenLogits, logitsData[start:start+vocabSize])
	}

	newKV := &kvCache{
		convStates:      make(map[int][]float32),
		recurrentStates: make(map[int][]float32),
		keyStates:       make(map[int][]float32),
		valueStates:     make(map[int][]float32),
		keySeqLens:      make(map[int]int64),
	}

	for _, layer := range linearAttentionLayers {
		convData := presentConvTensors[layer].(*ort.Tensor[float32]).GetData()
		newKV.convStates[layer] = make([]float32, len(convData))
		copy(newKV.convStates[layer], convData)

		recData := presentRecurrentTensors[layer].(*ort.Tensor[float32]).GetData()
		newKV.recurrentStates[layer] = make([]float32, len(recData))
		copy(newKV.recurrentStates[layer], recData)
	}

	for _, layer := range fullAttentionLayers {
		keyData := presentKeyTensors[layer].(*ort.Tensor[float32]).GetData()
		newKV.keyStates[layer] = make([]float32, len(keyData))
		copy(newKV.keyStates[layer], keyData)

		valueData := presentValueTensors[layer].(*ort.Tensor[float32]).GetData()
		newKV.valueStates[layer] = make([]float32, len(valueData))
		copy(newKV.valueStates[layer], valueData)

		newKV.keySeqLens[layer] = pastKV.keySeqLens[layer] + seqLen
	}

	logitsTensor.Destroy()
	for _, t := range pastConvTensors {
		t.Destroy()
	}
	for _, t := range pastRecurrentTensors {
		t.Destroy()
	}
	for _, t := range pastKeyTensors {
		t.Destroy()
	}
	for _, t := range pastValueTensors {
		t.Destroy()
	}
	for _, t := range presentConvTensors {
		t.Destroy()
	}
	for _, t := range presentRecurrentTensors {
		t.Destroy()
	}
	for _, t := range presentKeyTensors {
		t.Destroy()
	}
	for _, t := range presentValueTensors {
		t.Destroy()
	}

	return lastTokenLogits, newKV, nil
}

func (s *OnnxLLMService) sampleToken(logits []float32, params GenerateParams) int64 {
	if params.Temperature > 0 && params.Temperature != 1.0 {
		for i := range logits {
			logits[i] = logits[i] / float32(params.Temperature)
		}
	}

	maxLogit := float32(math.Inf(-1))
	for _, l := range logits {
		if l > maxLogit {
			maxLogit = l
		}
	}

	var sumExp float64
	for i := range logits {
		logits[i] = float32(math.Exp(float64(logits[i] - maxLogit)))
		sumExp += float64(logits[i])
	}
	for i := range logits {
		logits[i] /= float32(sumExp)
	}

	if params.TopK > 0 {
		type tokenProb struct {
			id   int
			prob float32
		}
		probs := make([]tokenProb, len(logits))
		for i, p := range logits {
			probs[i] = tokenProb{id: i, prob: p}
		}
		sort.Slice(probs, func(i, j int) bool {
			return probs[i].prob > probs[j].prob
		})

		k := params.TopK
		if k > len(probs) {
			k = len(probs)
		}

		filteredProbs := make([]float32, len(logits))
		var filteredSum float64
		for i := 0; i < k; i++ {
			filteredProbs[probs[i].id] = probs[i].prob
			filteredSum += float64(probs[i].prob)
		}

		for i := range filteredProbs {
			filteredProbs[i] /= float32(filteredSum)
		}
		copy(logits, filteredProbs)
	}

	if params.TopP > 0 && params.TopP < 1.0 {
		type tokenProb struct {
			id   int
			prob float32
		}
		probs := make([]tokenProb, len(logits))
		for i, p := range logits {
			probs[i] = tokenProb{id: i, prob: p}
		}
		sort.Slice(probs, func(i, j int) bool {
			return probs[i].prob > probs[j].prob
		})

		var cumSum float64
		cutoffIdx := len(probs)
		for i, p := range probs {
			cumSum += float64(p.prob)
			if cumSum > float64(params.TopP) {
				cutoffIdx = i + 1
				break
			}
		}

		filteredProbs := make([]float32, len(logits))
		var filteredSum float64
		for i := 0; i < cutoffIdx; i++ {
			filteredProbs[probs[i].id] = probs[i].prob
			filteredSum += float64(probs[i].prob)
		}

		for i := range filteredProbs {
			filteredProbs[i] /= float32(filteredSum)
		}
		copy(logits, filteredProbs)
	}

	r := rand.Float64()
	var cumSum float64
	for i, p := range logits {
		cumSum += float64(p)
		if cumSum >= r {
			return int64(i)
		}
	}

	return int64(len(logits) - 1)
}

func buildDecoderInputNames() []string {
	names := []string{"inputs_embeds", "attention_mask", "position_ids"}

	for _, layer := range linearAttentionLayers {
		names = append(names, fmt.Sprintf("past_conv.%d", layer))
		names = append(names, fmt.Sprintf("past_recurrent.%d", layer))
	}

	for _, layer := range fullAttentionLayers {
		names = append(names, fmt.Sprintf("past_key_values.%d.key", layer))
		names = append(names, fmt.Sprintf("past_key_values.%d.value", layer))
	}

	return names
}

func buildDecoderOutputNames() []string {
	names := []string{"logits"}

	for _, layer := range linearAttentionLayers {
		names = append(names, fmt.Sprintf("present_conv.%d", layer))
		names = append(names, fmt.Sprintf("present_recurrent.%d", layer))
	}

	for _, layer := range fullAttentionLayers {
		names = append(names, fmt.Sprintf("present.%d.key", layer))
		names = append(names, fmt.Sprintf("present.%d.value", layer))
	}

	return names
}
