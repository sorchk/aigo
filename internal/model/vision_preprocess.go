package model

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"net/http"
	"strings"
)

const (
	visionPatchSize         = 16
	visionTemporalPatchSize = 2
	visionMergeSize         = 2
	visionPatchDim          = 3 * visionTemporalPatchSize * visionPatchSize * visionPatchSize
	visionImageMean         = 0.5
	visionImageStd          = 0.5
	visionFactor            = visionPatchSize * visionMergeSize
	visionMinPixels         = 256 * 28 * 28
	visionMaxPixels         = 1280 * 28 * 28
)

type VisionPreprocessResult struct {
	PixelValues []float32
	GridTHW     []int64
	NumFeatures int
}

func smartResize(height, width int) (int, int) {
	hBar := roundByFactor(height, visionFactor)
	wBar := roundByFactor(width, visionFactor)
	if hBar*wBar > visionMaxPixels {
		scale := math.Sqrt(float64(visionMaxPixels) / float64(hBar*wBar))
		hBar = roundByFactor(int(float64(hBar)*scale), visionFactor)
		wBar = roundByFactor(int(float64(wBar)*scale), visionFactor)
	} else if hBar*wBar < visionMinPixels {
		scale := math.Sqrt(float64(visionMinPixels) / float64(hBar*wBar))
		hBar = roundByFactor(int(float64(hBar)*scale), visionFactor)
		wBar = roundByFactor(int(float64(wBar)*scale), visionFactor)
	}
	return hBar, wBar
}

func roundByFactor(x, factor int) int {
	return ((x + factor/2) / factor) * factor
}

func DecodeImageFromURL(urlStr string) (image.Image, error) {
	if strings.HasPrefix(urlStr, "data:") {
		idx := strings.Index(urlStr, ",")
		if idx < 0 {
			return nil, fmt.Errorf("invalid data URL")
		}
		b64Data := urlStr[idx+1:]
		data, err := base64.StdEncoding.DecodeString(b64Data)
		if err != nil {
			return nil, fmt.Errorf("base64 decode: %w", err)
		}
		return decodeImage(data)
	}

	resp, err := http.Get(urlStr)
	if err != nil {
		return nil, fmt.Errorf("fetch image: %w", err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read image: %w", err)
	}
	return decodeImage(data)
}

func decodeImage(data []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	return img, err
}

func PreprocessImage(img image.Image) (*VisionPreprocessResult, error) {
	bounds := img.Bounds()
	origH := bounds.Dy()
	origW := bounds.Dx()

	newH, newW := smartResize(origH, origW)

	resized := resizeImage(img, newW, newH)

	hPatches := newH / visionPatchSize
	wPatches := newW / visionPatchSize
	gridH := newH / visionFactor
	gridW := newW / visionFactor
	gridT := 1

	numPatches := gridT * hPatches * wPatches
	numFeatures := gridT * gridH * gridW

	pixelValues := make([]float32, numPatches*visionPatchDim)

	for hi := 0; hi < hPatches; hi++ {
		for wi := 0; wi < wPatches; wi++ {
			patchIdx := hi*wPatches + wi
			patchOffset := patchIdx * visionPatchDim

			for t := 0; t < visionTemporalPatchSize; t++ {
				for py := 0; py < visionPatchSize; py++ {
					for px := 0; px < visionPatchSize; px++ {
						y := hi*visionPatchSize + py
						x := wi*visionPatchSize + px
						r, g, b, _ := resized.At(x, y).RGBA()

						rf := (float32(float64(r)/65535.0) - visionImageMean) / visionImageStd
						gf := (float32(float64(g)/65535.0) - visionImageMean) / visionImageStd
						bf := (float32(float64(b)/65535.0) - visionImageMean) / visionImageStd

						tOffset := t * visionPatchSize * visionPatchSize * 3
						yxOffset := py*visionPatchSize*3 + px*3

						pixelValues[patchOffset+tOffset+yxOffset+0] = rf
						pixelValues[patchOffset+tOffset+yxOffset+1] = gf
						pixelValues[patchOffset+tOffset+yxOffset+2] = bf
					}
				}
			}
		}
	}

	gridTHW := []int64{int64(gridT), int64(gridH), int64(gridW)}

	return &VisionPreprocessResult{
		PixelValues: pixelValues,
		GridTHW:     gridTHW,
		NumFeatures: numFeatures,
	}, nil
}

func resizeImage(img image.Image, newW, newH int) *image.RGBA {
	srcBounds := img.Bounds()
	srcW := srcBounds.Dx()
	srcH := srcBounds.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, newW, newH))

	sx := float64(srcW) / float64(newW)
	sy := float64(srcH) / float64(newH)

	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			srcX := float64(x) * sx
			srcY := float64(y) * sy
			c := bilinearInterpolate(img, srcX, srcY, srcBounds)
			dst.Set(x, y, c)
		}
	}
	return dst
}

func bilinearInterpolate(img image.Image, x, y float64, bounds image.Rectangle) color.Color {
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	x1 := x0 + 1
	y1 := y0 + 1

	if x0 < bounds.Min.X {
		x0 = bounds.Min.X
	}
	if x1 > bounds.Max.X-1 {
		x1 = bounds.Max.X - 1
	}
	if y0 < bounds.Min.Y {
		y0 = bounds.Min.Y
	}
	if y1 > bounds.Max.Y-1 {
		y1 = bounds.Max.Y - 1
	}

	fx := x - float64(x0)
	fy := y - float64(y0)

	c00 := color.NRGBAModel.Convert(img.At(x0, y0)).(color.NRGBA)
	c10 := color.NRGBAModel.Convert(img.At(x1, y0)).(color.NRGBA)
	c01 := color.NRGBAModel.Convert(img.At(x0, y1)).(color.NRGBA)
	c11 := color.NRGBAModel.Convert(img.At(x1, y1)).(color.NRGBA)

	lerp := func(a, b uint8, t float64) uint8 {
		return uint8(float64(a)*(1-t) + float64(b)*t)
	}

	return color.NRGBA{
		R: lerp(lerp(c00.R, c10.R, fx), lerp(c01.R, c11.R, fx), fy),
		G: lerp(lerp(c00.G, c10.G, fx), lerp(c01.G, c11.G, fx), fy),
		B: lerp(lerp(c00.B, c10.B, fx), lerp(c01.B, c11.B, fx), fy),
		A: 255,
	}
}
