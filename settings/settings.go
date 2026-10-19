package settings

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

type Config struct {
	Version     int         `json:"version"`
	Processing  Processing  `json:"processing"`
	Recognition Recognition `json:"recognition"`
	Translation Translation `json:"translation"`
	Repair      Repair      `json:"repair"`
	Layout      Layout      `json:"layout"`
	Output      Output      `json:"output"`
}

type Processing struct {
	ReuseResults bool `json:"reuse_results"`
	BatchPages   int  `json:"batch_pages"`
}

type Recognition struct {
	Confidence float64 `json:"confidence"`
	TileSize   int     `json:"tile_size"`
}

type Translation struct {
	ModelPath     string `json:"model_path"`
	BatchAttempts int    `json:"batch_attempts"`
}

type Repair struct {
	MaskDilation   int `json:"mask_dilation"`
	ContextPadding int `json:"context_padding"`
	TileSize       int `json:"tile_size"`
}

type Layout struct {
	Direction   string  `json:"direction"`
	FontPath    string  `json:"font_path"`
	FontScale   float64 `json:"font_scale"`
	MinFontSize float64 `json:"min_font_size"`
	Spacing     float64 `json:"spacing"`
	Color       string  `json:"color"`
	StrokeWidth float64 `json:"stroke_width"`
}

type Output struct {
	Format          string `json:"format"`
	JPEGQuality     int    `json:"jpeg_quality"`
	WebPQuality     int    `json:"webp_quality"`
	WebPLossless    bool   `json:"webp_lossless"`
	MaxLongEdge     int    `json:"max_long_edge"`
	PreserveNoText  bool   `json:"preserve_no_text"`
	SaveEditable    bool   `json:"save_editable"`
	SaveDiagnostics bool   `json:"save_diagnostics"`
	Existing        string `json:"existing"`
}

func Default() Config {
	return Config{
		Version:     2,
		Processing:  Processing{ReuseResults: true, BatchPages: 2},
		Recognition: Recognition{Confidence: 0.35},
		Translation: Translation{BatchAttempts: 2},
		Repair:      Repair{MaskDilation: 6, ContextPadding: 64},
		Layout:      Layout{Direction: "vertical", FontScale: 1, Spacing: 1, Color: "auto", StrokeWidth: -1},
		Output:      Output{Format: "jpeg", JPEGQuality: 92, WebPQuality: 90, PreserveNoText: true, Existing: "ask"},
	}
}

func DefaultPath() (string, error) {
	if path := os.Getenv("MANGATRANS_SETTINGS"); path != "" {
		return path, nil
	}
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "MangaTrans", "settings.json"), nil
}

func Parse(r io.Reader) (Config, error) {
	data, err := io.ReadAll(io.LimitReader(r, (1<<20)+1))
	if err != nil {
		return Config{}, err
	}
	if len(data) > 1<<20 {
		return Config{}, fmt.Errorf("settings exceed 1 MiB")
	}
	if text := strings.TrimSpace(string(data)); len(text) == 0 || text[0] != '{' {
		return Config{}, fmt.Errorf("settings must be a JSON object")
	}
	data, err = migrate(data)
	if err != nil {
		return Config{}, err
	}
	c := Default()
	d := json.NewDecoder(strings.NewReader(string(data)))
	d.DisallowUnknownFields()
	if err := d.Decode(&c); err != nil {
		return Config{}, fmt.Errorf("read settings: %w", err)
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return Config{}, fmt.Errorf("settings must contain one JSON object")
	}
	if err := c.Validate(); err != nil {
		return Config{}, err
	}
	return c, nil
}

func Load(path string) (Config, error) {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return Default(), nil
	}
	if err != nil {
		return Config{}, err
	}
	defer f.Close()
	return Parse(f)
}

func Save(path string, c Config) error {
	if err := c.Validate(); err != nil {
		return err
	}
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".settings-*.json")
	if err != nil {
		return err
	}
	name := f.Name()
	defer os.Remove(name)
	if _, err := f.Write(append(data, '\n')); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(name, path)
}

func (c Config) Validate() error {
	if c.Version != 2 {
		return fmt.Errorf("unsupported settings version: %d", c.Version)
	}
	choices := []struct{ name, value, allowed string }{
		{"text direction", c.Layout.Direction, "vertical horizontal auto"},
		{"output format", c.Output.Format, "jpeg png webp"},
		{"existing output policy", c.Output.Existing, "ask rename"},
	}
	for _, choice := range choices {
		found := false
		for _, allowed := range strings.Fields(choice.allowed) {
			found = found || choice.value == allowed
		}
		if !found {
			return fmt.Errorf("invalid %s: %q", choice.name, choice.value)
		}
	}
	if c.Processing.BatchPages < 1 || c.Processing.BatchPages > 4 {
		return fmt.Errorf("translation batch pages must be 1..4")
	}
	if !(c.Recognition.Confidence >= 0.05 && c.Recognition.Confidence <= 0.95) {
		return fmt.Errorf("detection confidence must be 0.05..0.95")
	}
	if c.Recognition.TileSize != 0 && (c.Recognition.TileSize < 256 || c.Recognition.TileSize > 4096) {
		return fmt.Errorf("detection tile size must be 0 or 256..4096 pixels")
	}
	if c.Translation.BatchAttempts < 1 || c.Translation.BatchAttempts > 5 {
		return fmt.Errorf("translation attempts must be 1..5")
	}
	if c.Repair.MaskDilation < 0 || c.Repair.MaskDilation > 24 || c.Repair.ContextPadding < 16 || c.Repair.ContextPadding > 256 {
		return fmt.Errorf("mask dilation must be 0..24 pixels and repair context 16..256 pixels")
	}
	if c.Repair.TileSize != 0 && (c.Repair.TileSize < 128 || c.Repair.TileSize > 2048) {
		return fmt.Errorf("repair tile size must be 0 or 128..2048 pixels")
	}
	if !(c.Layout.StrokeWidth >= -1 && c.Layout.StrokeWidth <= 4) || (c.Layout.StrokeWidth < 0 && c.Layout.StrokeWidth != -1) {
		return fmt.Errorf("stroke width must be -1 for auto, or 0..4 pixels")
	}
	if c.Output.JPEGQuality < 80 || c.Output.JPEGQuality > 100 || c.Output.WebPQuality < 1 || c.Output.WebPQuality > 100 {
		return fmt.Errorf("JPEG quality must be 80..100 and WebP quality must be 1..100")
	}
	if c.Output.MaxLongEdge != 0 && (c.Output.MaxLongEdge < 640 || c.Output.MaxLongEdge > 16383) {
		return fmt.Errorf("output long edge must be 0 or 640..16383")
	}
	if !(c.Layout.FontScale >= 0.7 && c.Layout.FontScale <= 1.4) ||
		!(c.Layout.Spacing >= 0.5 && c.Layout.Spacing <= 2) ||
		!(c.Layout.MinFontSize >= 0 && c.Layout.MinFontSize <= 72) {
		return fmt.Errorf("font scale must be 0.7..1.4, spacing 0.5..2, minimum font size 0..72")
	}
	if c.Layout.Color != "auto" {
		if len(c.Layout.Color) != 7 || c.Layout.Color[0] != '#' {
			return fmt.Errorf("text color must be auto or #RRGGBB")
		}
		if _, err := hex.DecodeString(c.Layout.Color[1:]); err != nil {
			return fmt.Errorf("text color must be auto or #RRGGBB")
		}
	}
	return nil
}
