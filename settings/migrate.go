package settings

import (
	"encoding/json"
	"fmt"
)

func migrate(data []byte) ([]byte, error) {
	var root map[string]json.RawMessage
	if err := json.Unmarshal(data, &root); err != nil {
		return nil, err
	}
	var version int
	if raw, ok := root["version"]; ok {
		if err := json.Unmarshal(raw, &version); err != nil {
			return nil, err
		}
	}
	if version != 1 {
		return data, nil
	}
	type legacyField struct {
		old, current string
		values       map[string]any
	}
	groups := []struct {
		name   string
		fields []legacyField
	}{
		{"recognition", []legacyField{
			{"sensitivity", "confidence", map[string]any{"conservative": 0.5, "standard": 0.35, "sensitive": 0.2}},
			{"precision", "tile_size", map[string]any{"standard": 0, "fine": 1024}},
		}},
		{"translation", []legacyField{
			{"retry", "batch_attempts", map[string]any{"standard": 2, "enhanced": 4}},
		}},
		{"repair", []legacyField{
			{"mask", "mask_dilation", map[string]any{"small": 2, "standard": 6, "large": 9}},
			{"quality", "tile_size", map[string]any{"standard": 0, "fine": 512}},
		}},
		{"layout", []legacyField{
			{"stroke", "stroke_width", map[string]any{"auto": -1, "off": 0, "thin": 1, "thick": 2}},
		}},
	}
	for _, group := range groups {
		raw, exists := root[group.name]
		if !exists {
			continue
		}
		var object map[string]json.RawMessage
		if err := json.Unmarshal(raw, &object); err != nil {
			return nil, err
		}
		for _, field := range group.fields {
			raw, exists := object[field.old]
			if !exists {
				continue
			}
			var old string
			if err := json.Unmarshal(raw, &old); err != nil {
				return nil, err
			}
			value, valid := field.values[old]
			if !valid {
				return nil, fmt.Errorf("invalid legacy %s.%s", group.name, field.old)
			}
			if _, duplicate := object[field.current]; duplicate {
				return nil, fmt.Errorf("ambiguous legacy field %s.%s", group.name, field.current)
			}
			object[field.current], _ = json.Marshal(value)
			delete(object, field.old)
		}
		root[group.name], _ = json.Marshal(object)
	}
	root["version"] = json.RawMessage("2")
	return json.Marshal(root)
}
