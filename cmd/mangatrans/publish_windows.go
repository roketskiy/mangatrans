package main

import (
	"os"

	"golang.org/x/sys/windows"
)

func publishArchive(staged, output string, overwrite bool) error {
	if overwrite {
		return os.Rename(staged, output)
	}
	from, err := windows.UTF16PtrFromString(staged)
	if err != nil {
		return err
	}
	to, err := windows.UTF16PtrFromString(output)
	if err != nil {
		return err
	}
	return windows.MoveFile(from, to)
}
