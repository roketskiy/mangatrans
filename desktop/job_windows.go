package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	jobMu     sync.Mutex
	jobHandle windows.Handle
)

// initKillJob creates a job object whose processes die when the UI process
// exits, even if the exit is a forced kill from Task Manager.
func initKillJob() error {
	jobMu.Lock()
	defer jobMu.Unlock()
	if jobHandle != 0 {
		return nil
	}

	h, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return err
	}
	info := windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION{}
	info.BasicLimitInformation.LimitFlags = windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
	if _, err := windows.SetInformationJobObject(
		h,
		windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&info)),
		uint32(unsafe.Sizeof(info)),
	); err != nil {
		windows.CloseHandle(h)
		return err
	}
	jobHandle = h
	return nil
}

// assignKillJob puts the engine, and later its llama-server children, in the
// job. Children inherit the job.
func assignKillJob(pid int) error {
	jobMu.Lock()
	defer jobMu.Unlock()
	if jobHandle == 0 {
		return fmt.Errorf("job object is not initialized")
	}
	h, err := windows.OpenProcess(
		windows.PROCESS_SET_QUOTA|windows.PROCESS_TERMINATE,
		false,
		uint32(pid),
	)
	if err != nil {
		return err
	}
	defer windows.CloseHandle(h)
	return windows.AssignProcessToJobObject(jobHandle, h)
}

func terminateKillJob() error {
	jobMu.Lock()
	defer jobMu.Unlock()
	if jobHandle == 0 {
		return fmt.Errorf("job object is not initialized")
	}
	return windows.TerminateJobObject(jobHandle, 1)
}

func closeKillJob() {
	jobMu.Lock()
	defer jobMu.Unlock()
	if jobHandle != 0 {
		windows.CloseHandle(jobHandle)
		jobHandle = 0
	}
}

type jobSelfTestResult struct {
	Assigned bool   `json:"assigned"`
	Killed   bool   `json:"killed"`
	ExitCode int    `json:"exit_code"`
	Error    string `json:"error,omitempty"`
}

// jobSelfTest starts a hidden parent/child process tree, assigns it to the
// job, closes the job handle, and verifies both processes die.
func jobSelfTest() {
	result := jobSelfTestResult{ExitCode: -1}
	outPath := filepath.Join(".cache", "out", "job-selftest.json")
	_ = os.MkdirAll(filepath.Dir(outPath), 0o755)

	writeResult := func() {
		data, _ := json.MarshalIndent(result, "", "  ")
		_ = os.WriteFile(outPath, data, 0o644)
	}

	if err := initKillJob(); err != nil {
		result.Error = err.Error()
		writeResult()
		return
	}

	cmd := exec.Command("cmd.exe", "/c", "ping", "-n", "30", "127.0.0.1")
	cmd.SysProcAttr = &syscall.SysProcAttr{
		HideWindow:    true,
		CreationFlags: 0x08000000, // CREATE_NO_WINDOW
	}
	if err := cmd.Start(); err != nil {
		result.Error = err.Error()
		writeResult()
		return
	}

	if err := assignKillJob(cmd.Process.Pid); err != nil {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
		result.Error = err.Error()
		writeResult()
		return
	}
	result.Assigned = true

	// Closing the job handle is what Windows does when the UI process exits.
	closeKillJob()

	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case <-done:
		result.Killed = cmd.ProcessState != nil
		if cmd.ProcessState != nil {
			result.ExitCode = cmd.ProcessState.ExitCode()
		}
	case <-time.After(5 * time.Second):
		_ = cmd.Process.Kill()
		result.Killed = false
		result.Error = "process tree survived job close"
	}
	writeResult()
}
