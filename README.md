# Emji

On-device emoji semantic lookup CLI powered by vector search

## What it does

It's a simple CLI that uses [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main/onnx) quantized ONNX model and a vector search powered by [sqlite-vec](https://github.com/asg017/sqlite-vec) to help you to find emoji you are looking for

## Installation

### Prerequisites

- Python 3.12 or higher

### Install from source

```bash
uv pip install -e .
```

### Data Source

All data is stored under `~/.emji`:

- Model file: `~/.emji/model_qint8_arm64.onnx`
- Default emoji data: `~/.emji/shortnames.json`
- Optional override data: `~/.emji/shortnames_override.json` (takes precedence if present)
- Vector index: `~/.emji/emoji_index.db`
- Config: `~/.emji/.config`

On first run, the CLI will automatically download:

- The emoji data from this [gist](https://gist.github.com/subpath/13bd5c15f76f451dfcb85421a53f0666)
- The quantized ONNX model from `sentence-transformers/all-MiniLM-L6-v2`

### Model

The CLI will automatically download the quantized ONNX model from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main/onnx) when needed. You can also use any other bi-encoder of your choice that fits your device by replacing the ONNX file. If you are using a model that requires different prompts to encode a query and a searched entity (emojis in our case) then you will need to modify the source code.

The tiny dedicated finetuned MiniLM is coming soon.

## Quick Start

1. **Search for emojis** (the CLI will automatically download dependencies and build the index if needed):

  ```bash
  emji happy birthday
  emji coffee break
  emji celebration party
  ```

2. **Select and copy**: Choose from the interactive list, and the emoji will be copied to your clipboard!

3. **Force rebuild index** (if needed):

  ```bash
  emji --build-index
  ```

4. **Cleanup all Emji data** (removes `~/.emji`):

  ```bash
  emji --cleanup
  ```

## Usage

### Commands

- `emji <text>` - Search for emojis matching your description
- `emji --build-index` - Force rebuild the semantic search index
- `emji --cleanup` - Delete all Emji data and config under `~/.emji`

**Options:**

- `--n <number>`: Number of results to return (default: 3)

### Examples

```bash
# Find celebration emojis
emji party celebration

# Find food-related emojis
emji delicious food

# Find weather emojis
emji sunny day

# Get more results
emji animals --n 5

# Force rebuild the index
emji --build-index
```

## How It Works

1. **Automatic Setup**: On first run, the CLI automatically downloads the required model and emoji data
2. **Embedding Generation**: Uses a quantized version of `sentence-transformers/all-MiniLM-L6-v2` to convert emoji names and descriptions into 384-dimensional vectors
3. **Vector Database**: Stores embeddings in SQLite with `sqlite-vec` extension for fast similarity search
4. **Semantic Matching**: When you search, your query is converted to an embedding and compared against all emoji embeddings
5. **Interactive Selection**: The best matches are presented in an interactive menu for you to choose from

## Development

### Setup

```bash
# create virtual env
uv venv
# activate
source .venv/bin/activate
# sync dependencies
uv sync --all-groups
# Set up pre-commit hooks
pre-commit install
```

## License

Licensed under the Apache License 2.0\. See <LICENSE> for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
