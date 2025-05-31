build:
	go build -o dist/openai-client *.go

clean:
	rm -rf dist

.PHONY: build clean
