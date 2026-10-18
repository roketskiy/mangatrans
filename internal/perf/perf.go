// Package perf records host-observed latency with Go's monotonic clock.
// It does not claim to measure CUDA kernel execution independently of its callers.
package perf

import (
	"context"
	"encoding/json"
	"os"
	"runtime"
	"sync"
)

type Event struct {
	Kind       string `json:"kind"`
	Stage      string `json:"stage"`
	Page       int    `json:"page"`
	Item       int    `json:"item"`
	StartNS    int64  `json:"start_ns"`
	DurationNS int64  `json:"duration_ns"`
	Error      string `json:"error,omitempty"`
	Count      int64  `json:"count,omitempty"`
}

type Page struct {
	Index       int    `json:"index"`
	Name        string `json:"name"`
	Width       int    `json:"width"`
	Height      int    `json:"height"`
	Detected    int    `json:"detected"`
	OCRRegions  int    `json:"ocr_regions"`
	OCRAccepted int    `json:"ocr_accepted"`
	OCRFailed   int    `json:"ocr_failed"`
	OCRRejected int    `json:"ocr_rejected"`
	CropSkipped int    `json:"crop_skipped"`
	Translated  int    `json:"translated"`
	Rendered    int    `json:"rendered"`
	Outcome     string `json:"outcome"`
}

type Recorder struct {
	mu        sync.Mutex
	start     int64
	path      string
	Events    []Event         `json:"events"`
	Pages     []*Page         `json:"pages"`
	Schema    int             `json:"schema"`
	TotalNS   int64           `json:"total_ns"`
	Error     string          `json:"error,omitempty"`
	GoVersion string          `json:"go_version"`
	Files     map[string]File `json:"files"`
	Clock     *ClockAnchor    `json:"clock"`
}

type File struct {
	Path       string `json:"path"`
	Bytes      int64  `json:"bytes"`
	ModifiedNS int64  `json:"modified_ns"`
	Error      string `json:"error,omitempty"`
}

func (r *Recorder) FilesUsed(paths map[string]string) {
	if r == nil {
		return
	}
	r.Files = make(map[string]File, len(paths))
	for role, path := range paths {
		f := File{Path: path}
		if st, err := os.Stat(path); err != nil {
			f.Error = err.Error()
		} else {
			f.Bytes, f.ModifiedNS = st.Size(), st.ModTime().UnixNano()
		}
		r.Files[role] = f
	}
}

func New(path string, start int64) *Recorder {
	if path == "" {
		return nil
	}
	return &Recorder{path: path, start: start, Schema: 1, GoVersion: runtime.Version(), Clock: clockAnchor(start), Events: []Event{}, Pages: []*Page{}}
}

// Span returns a completion function. Call it exactly once, including on errors.
// Page and item are zero-based; -1 means not associated with a page/item.
func (r *Recorder) Span(kind, stage string, page, item int) func(error) {
	if r == nil {
		return func(error) {}
	}
	start := Now()
	return func(err error) {
		end := Now()
		e := Event{Kind: kind, Stage: stage, Page: page, Item: item,
			StartNS: start - r.start, DurationNS: end - start}
		if err != nil {
			e.Error = err.Error()
		}
		r.mu.Lock()
		r.Events = append(r.Events, e)
		r.mu.Unlock()
	}
}

func (r *Recorder) Page(index int, name string) *Page {
	p := &Page{Index: index, Name: name, Outcome: "unprocessed"}
	if r != nil {
		r.Pages = append(r.Pages, p)
	}
	return p
}

// Finish is deferred before resource cleanup so total includes that cleanup.
// Report serialization and writing happen after TotalNS is captured.
func (r *Recorder) Finish(err error) error {
	if r == nil {
		return nil
	}
	r.TotalNS = Now() - r.start
	if err != nil {
		r.Error = err.Error()
	}
	data, e := json.MarshalIndent(r, "", "  ")
	if e != nil {
		return e
	}
	return os.WriteFile(r.path, data, 0o644)
}

type contextKey struct{}
type locationKey struct{}
type location struct{ page, item int }

func WithLocation(ctx context.Context, page, item int) context.Context {
	return context.WithValue(ctx, locationKey{}, location{page, item})
}

func WithItem(ctx context.Context, item int) context.Context {
	loc, ok := ctx.Value(locationKey{}).(location)
	if !ok {
		loc.page = -1
	}
	return WithLocation(ctx, loc.page, item)
}

func WithRecorder(ctx context.Context, r *Recorder) context.Context {
	return context.WithValue(ctx, contextKey{}, r)
}

func Request(ctx context.Context, stage string) func(error) {
	return ContextSpan(ctx, "request", stage)
}

func ContextSpan(ctx context.Context, kind, stage string) func(error) {
	r, _ := ctx.Value(contextKey{}).(*Recorder)
	loc, ok := ctx.Value(locationKey{}).(location)
	if !ok {
		loc = location{-1, -1}
	}
	return r.Span(kind, stage, loc.page, loc.item)
}

func Count(ctx context.Context, stage string, count int) {
	r, _ := ctx.Value(contextKey{}).(*Recorder)
	if r == nil {
		return
	}
	loc, ok := ctx.Value(locationKey{}).(location)
	if !ok {
		loc = location{-1, -1}
	}
	e := Event{Kind: "counter", Stage: stage, Page: loc.page, Item: loc.item, StartNS: Now() - r.start, Count: int64(count)}
	r.mu.Lock()
	r.Events = append(r.Events, e)
	r.mu.Unlock()
}
