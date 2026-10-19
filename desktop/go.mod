module mangatrans-desktop

go 1.27.1

require (
	github.com/egoist/mygo v0.5.5
	golang.org/x/sys v0.49.0
	mangatrans v0.0.0
)

replace mangatrans => ..

require (
	github.com/ebitengine/purego v0.11.1 // indirect
	github.com/go-text/typesetting v0.3.5 // indirect
	golang.org/x/image v0.47.0 // indirect
	golang.org/x/text v0.43.0 // indirect
)
