//go:build !windows

package perf

import "time"

type ClockAnchor struct{}

var clockOrigin = time.Now()

func Now() int64 { return time.Since(clockOrigin).Nanoseconds() }

func clockAnchor(int64) *ClockAnchor { return nil }
