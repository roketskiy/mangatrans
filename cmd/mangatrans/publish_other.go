//go:build !windows

package main

import "os"

func publishArchive(staged, output string, overwrite bool) error {
	if overwrite {
		return os.Rename(staged, output)
	}
	return os.Link(staged, output)
}
