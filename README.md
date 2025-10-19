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

### Data Souce

You need can you emjies description from this [gist](https://gist.github.com/subpath/13bd5c15f76f451dfcb85421a53f0666), on use your own

```
wget https://gist.githubusercontent.com/subpath/13bd5c15f76f451dfcb85421a53f0666/raw/1d362e4b4addfcd920b88f949090c6e82bf2c791/emojies_shortnames.json
```

### Model

At the moment you can us vanilla [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main/onnx) or any other biencoder of your chouse that fits your device. Just use ONNX file. If you are using a model that requires different prompts to encode a query and a searched entity (emojies in our case) then you will need to modify the source code.

The tiny dedicated finetuned MiniLM is coming soon.

## Quick Start

1. **Build the emoji index** (first time only):

  ```bash
  emji build-index
  ```

2. **Search for emojis**:

  ```bash
  emji query "happy birthday"
  emji query "coffee break"
  emji query "celebration party"
  ```

3. **Select and copy**: Choose from the interactive list, and the emoji will be copied to your clipboard!

## Usage

### Commands

- `emji build-index` - Build the semantic search index from emoji data
- `emji query <text>` - Search for emojis matching your description

  - `--n` or `-n`: Number of results to return (default: 3)

### Examples

```bash
# Find celebration emojis
emji query "party celebration"

# Find food-related emojis
emji query "delicious food"

# Find weather emojis
emji query "sunny day"

# Get more results
emji query "animals" --n 5
```

## How It Works

1. **Embedding Generation**: Uses a quantized version of `sentence-transformers/all-MiniLM-L6-v2` to convert emoji names and descriptions into 384-dimensional vectors
2. **Vector Database**: Stores embeddings in SQLite with `sqlite-vec` extension for fast similarity search
3. **Semantic Matching**: When you search, your query is converted to an embedding and compared against all emoji embeddings
4. **Interactive Selection**: The best matches are presented in an interactive menu for you to choose from

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
