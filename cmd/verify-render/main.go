package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"path/filepath"
	"runtime/pprof"
	"strings"
	"time"

	"mangatrans/internal/render"
)

type result struct {
	Name      string `json:"name"`
	Passed    bool   `json:"passed"`
	Changed   int    `json:"changed_pixels"`
	Outside   int    `json:"outside_mask_pixels"`
	ElapsedMS int64  `json:"elapsed_ms"`
	Error     string `json:"error,omitempty"`
}

func main() {
	out := flag.String("out", ".cache/render-verification", "verification artifact directory")
	font := flag.String("font", `C:\Windows\Fonts\msyhbd.ttc`, "font used for rendering")
	deadline := flag.Duration("deadline", 8*time.Second, "maximum time for each render")
	profile := flag.String("cpu-profile", "", "optional CPU profile for diagnosis")
	flag.Parse()
	if *profile != "" {
		f, err := os.Create(*profile)
		if err != nil {
			panic(err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			panic(err)
		}
		defer f.Close()
		defer pprof.StopCPUProfile()
	}
	if err := run(*out, *font, *deadline); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(dir, font string, deadline time.Duration) error {
	if deadline <= 0 {
		return fmt.Errorf("deadline must be positive")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	background := color.RGBA{R: 210, G: 210, B: 210, A: 255}
	var results []result
	for _, vertical := range []bool{true, false} {
		for _, shape := range []string{"paragraph", "compact", "lobes", "shared", "punctuation", "empty"} {
			name := shape + "-horizontal"
			if vertical {
				name = shape + "-vertical"
			}
			page := image.NewRGBA(image.Rect(0, 0, 860, 1200))
			draw.Draw(page, page.Bounds(), image.NewUniform(background), image.Point{}, draw.Src)
			area := image.NewAlpha(image.Rect(70, 440, 790, 780))
			for y := area.Rect.Min.Y; y < area.Rect.Max.Y; y++ {
				for x := area.Rect.Min.X; x < area.Rect.Max.X; x++ {
					if shape == "empty" || shape == "lobes" && x > 400 && x < 460 && (y < 600 || y > 620) {
						continue
					}
					area.SetAlpha(x, y, color.Alpha{A: 255})
				}
			}
			text := strings.Repeat("\u6570\u5b57\u7248\u6536\u5f55\u7eb8\u8d28\u7248\u5185\u5bb9\u3002", 10)
			if shape == "punctuation" {
				text = strings.Repeat("\u300c\u6d4b\u8bd5\u300d\u3001ABC 2026\u3002", 8)
			}
			blocks := []render.Block{{Box: image.Rect(79, 463, 775, 755), Text: text, Vertical: vertical, Area: area}}
			if shape == "shared" {
				blocks = []render.Block{
					{Box: image.Rect(100, 480, 380, 730), Anchor: image.Rect(100, 480, 380, 730), Text: text[:len(text)/2], Vertical: vertical, Area: area},
					{Box: image.Rect(480, 480, 760, 730), Anchor: image.Rect(480, 480, 760, 730), Text: text[len(text)/2:], Vertical: vertical, Area: area},
				}
			}
			started := time.Now()
			minSize := 1.2
			if shape == "compact" {
				minSize = 6
			}
			completed := append([]result(nil), results...)
			timer := time.AfterFunc(deadline, func() {
				if f, err := os.Create(filepath.Join(dir, "stacks.txt")); err == nil {
					_ = pprof.Lookup("goroutine").WriteTo(f, 2)
					_ = f.Close()
				}
				pprof.StopCPUProfile()
				_ = writeReport(dir, append(completed, result{Name: name, Error: "render exceeded deadline"}))
				fmt.Fprintln(os.Stderr, name+": render exceeded deadline")
				os.Exit(124)
			})
			output, err := render.Render(page, blocks, render.Options{
				FontPath: font, MinSize: minSize, MaxSize: 18, Padding: 3, Spacing: 0.5,
				TextColor: color.RGBA{A: 255},
			})
			timer.Stop()
			r := result{Name: name, ElapsedMS: time.Since(started).Milliseconds()}
			if shape == "empty" {
				r.Passed = err != nil
			} else if err != nil {
				r.Error = err.Error()
			} else {
				for y := 0; y < 1200; y++ {
					for x := 0; x < 860; x++ {
						if color.RGBAModel.Convert(output.At(x, y)) != background {
							r.Changed++
							if area.AlphaAt(x, y).A != 255 {
								r.Outside++
							}
						}
					}
				}
				r.Passed = r.Changed > 0 && r.Outside == 0
				f, e := os.Create(filepath.Join(dir, name+".png"))
				if e != nil {
					return e
				}
				e = png.Encode(f, output)
				closeErr := f.Close()
				if e != nil {
					return e
				}
				if closeErr != nil {
					return closeErr
				}
			}
			results = append(results, r)
			fmt.Printf("%s passed=%t changed=%d outside=%d elapsed_ms=%d\n", r.Name, r.Passed, r.Changed, r.Outside, r.ElapsedMS)
		}
	}
	if err := writeReport(dir, results); err != nil {
		return err
	}
	for _, r := range results {
		if !r.Passed {
			return fmt.Errorf("render verification failed: %s: %s", r.Name, r.Error)
		}
	}
	return nil
}

func writeReport(dir string, results []result) error {
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "report.json"), data, 0o644)
}
