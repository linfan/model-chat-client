build:
	cargo build --release
	mkdir -p dist
	cp target/release/chat-client dist/

clean:
	cargo clean
	rm -rf dist

dev:
	cargo run

.PHONY: build clean dev