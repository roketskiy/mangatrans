package perf

import (
	"syscall"
	"unsafe"
)

// CPython's perf_counter_ns and these events use the same Windows QPC clock.
type ClockAnchor struct {
	QPCNS         int64 `json:"qpc_ns"`
	GoNS          int64 `json:"go_ns"`
	UncertaintyNS int64 `json:"uncertainty_ns"`
	FrequencyHz   int64 `json:"frequency_hz"`
}

var counterProc = syscall.NewLazyDLL("kernel32.dll").NewProc("QueryPerformanceCounter")
var counterFrequency = func() int64 {
	proc := syscall.NewLazyDLL("kernel32.dll").NewProc("QueryPerformanceFrequency")
	var frequency int64
	if ok, _, _ := proc.Call(uintptr(unsafe.Pointer(&frequency))); ok == 0 || frequency <= 0 {
		panic("QueryPerformanceFrequency failed")
	}
	return frequency
}()

// Now uses QPC directly rather than assuming time.Now's update granularity.
// Conversion preserves ticks without overflowing ticks*1e9.
func Now() int64 {
	var ticks int64
	if ok, _, _ := counterProc.Call(uintptr(unsafe.Pointer(&ticks))); ok == 0 {
		panic("QueryPerformanceCounter failed")
	}
	return ticks/counterFrequency*1e9 + ticks%counterFrequency*1e9/counterFrequency
}

func clockAnchor(start int64) *ClockAnchor {
	return &ClockAnchor{QPCNS: start, GoNS: 0, FrequencyHz: counterFrequency,
		UncertaintyNS: (1e9 + counterFrequency - 1) / counterFrequency}
}
