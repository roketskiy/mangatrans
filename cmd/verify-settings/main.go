package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"mangatrans/settings"
)

type check struct {
	Name string `json:"name"`
	Pass bool   `json:"pass"`
}

func main() {
	if err := verify(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func verify() error {
	dir := filepath.Join(".cache", "settings", "config-verification")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	var checks []check
	assert := func(name string, pass bool) {
		checks = append(checks, check{Name: name, Pass: pass})
	}
	d := settings.Default()
	assert("defaults validate", d.Validate() == nil)
	missing, err := settings.Load(filepath.Join(dir, "missing", "settings.json"))
	assert("missing file uses defaults", err == nil && missing == d)
	partial, err := settings.Parse(strings.NewReader(`{"output":{"format":"webp","webp_lossless":true}}`))
	assert("partial object retains nested defaults", err == nil && partial.Output.WebPLossless && partial.Output.WebPQuality == 90 && partial.Layout == d.Layout)
	for _, input := range []string{"null", "[]", "", "{} {}", `{"unknown":true}`, `{"version":3}`, `{"output":{"format":"avif"}}`, `{"output":{"webp_quality":101}}`, `{"layout":{"color":"#ffzz11"}}`, `{"processing":{"batch_pages":0}}`, `{"recognition":{"confidence":0}}`, `{"translation":{"batch_attempts":8}}`, `{"repair":{"tile_size":12}}`} {
		_, err := settings.Parse(strings.NewReader(input))
		assert("reject "+input, err != nil)
	}
	_, err = settings.Parse(strings.NewReader(`{}` + strings.Repeat(" ", 1<<20)))
	assert("reject oversized configuration", err != nil)
	legacy, err := settings.Parse(strings.NewReader(`{"version":1,"recognition":{"sensitivity":"sensitive","precision":"fine"},"translation":{"retry":"enhanced"},"repair":{"mask":"large","quality":"fine"},"layout":{"stroke":"thin"}}`))
	assert("legacy settings migrate to numeric parameters", err == nil && legacy.Version == 2 && legacy.Recognition.Confidence == 0.2 && legacy.Recognition.TileSize == 1024 && legacy.Translation.BatchAttempts == 4 && legacy.Repair.MaskDilation == 9 && legacy.Repair.TileSize == 512 && legacy.Layout.StrokeWidth == 1)
	custom := d
	custom.Output.Format, custom.Output.WebPLossless = "webp", true
	custom.Layout.Color, custom.Layout.Direction = "#12ABef", "horizontal"
	path := filepath.Join(dir, "settings.json")
	if err := settings.Save(path, custom); err != nil {
		return err
	}
	reloaded, err := settings.Load(path)
	assert("saved settings survive reload", err == nil && reloaded == custom)
	old, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	bad := custom
	bad.Output.JPEGQuality = 1
	err = settings.Save(path, bad)
	after, readErr := os.ReadFile(path)
	assert("invalid save preserves existing bytes", err != nil && readErr == nil && bytes.Equal(old, after))
	custom.Output.WebPLossless = false
	if err := settings.Save(path, custom); err != nil {
		return err
	}
	reloaded, err = settings.Load(path)
	assert("atomic save replaces existing file", err == nil && reloaded == custom)
	malformedPath := filepath.Join(dir, "malformed.json")
	if err := os.WriteFile(malformedPath, []byte("{broken"), 0o644); err != nil {
		return err
	}
	_, err = settings.Load(malformedPath)
	malformed, readErr := os.ReadFile(malformedPath)
	assert("malformed load reports error and preserves file", err != nil && readErr == nil && string(malformed) == "{broken")
	passed := true
	for _, result := range checks {
		passed = passed && result.Pass
	}
	report := struct {
		Time   time.Time `json:"time"`
		Passed bool      `json:"passed"`
		Checks []check   `json:"checks"`
	}{time.Now().UTC(), passed, checks}
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "report.json"), data, 0o644); err != nil {
		return err
	}
	if !passed {
		return fmt.Errorf("settings verification failed; see %s", filepath.Join(dir, "report.json"))
	}
	fmt.Printf("PASS %d settings checks; report %s\n", len(checks), filepath.Join(dir, "report.json"))
	return nil
}
