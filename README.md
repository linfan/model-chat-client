# Model Chat Client

A Rust-based command-line client for interacting with OpenAI-compatible APIs.

## Features

- Interactive chat with OpenAI-compatible APIs
- Support for streaming responses with Server-Sent Events (SSE)
- File input support for batch processing
- Command-line arguments for quick queries
- Configurable model and temperature settings
- Support for reasoning content display

## Installation

Make sure you have Rust installed. If not, install it from [rustup.rs](https://rustup.rs/).

```bash
# Build the project
make build

# Or use cargo directly
cargo build --release
```

## Configuration

Set the following environment variables:

- `OPENAI_URL`: API endpoint (default: `http://127.0.0.1:8000/v1/chat/completions`)
- `OPENAI_API_KEY`: Your API key (default: `none`)
- `OPENAI_MODEL`: Model name (default: uses the `-m` flag value or `plus`)
- `OPENAI_TEMPERATURE`: Temperature for responses (default: `0.7`)

## Usage

```bash
# Interactive mode
./dist/chat-client

# Quick query
./dist/chat-client "What is Rust programming language?"

# With specific model
./dist/chat-client -m gpt-4 "Explain async programming"

# Read from file
./dist/chat-client -f input.txt

# Debug mode
./dist/chat-client -d "Debug this code"

# Development mode (using cargo)
make dev
```

## Command Line Options

- `-f, --file <FILE>`: Specify file path to read content from
- `-m, --model <MODEL>`: Specify model name (default: plus)
- `-d, --debug`: Enable debug mode
- `--help`: Show help information

## Building

```bash
# Build release version
make build

# Clean build artifacts
make clean

# Run in development mode
make dev
```