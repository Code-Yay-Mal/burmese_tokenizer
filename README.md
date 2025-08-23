# Burmese Tokenizer

Tokenize Burmese text like a pro. No fancy stuff, just gets the job done.

## Quick Start

```bash
# Using pip
pip install burmese-tokenizer

# Using uv (faster)
uv add burmese-tokenizer
```

```python
from burmese_tokenizer import BurmeseTokenizer

tokenizer = BurmeseTokenizer()
text = "မင်္ဂလာပါ။ နေကောင်းပါသလား။"

# Tokenize
result = tokenizer.encode(text)
print(result["pieces"])  # ['▁မင်္ဂလာ', '▁ပါ', '။', '▁နေ', '▁ကောင်း', '▁ပါ', '▁သလား', '။']

# Decode
decoded = tokenizer.decode(result["pieces"])
print(decoded)  # မင်္ဂလာပါ။ နေကောင်းပါသလား။
```

## CLI

```bash
# Tokenize
burmese-tokenizer "မင်္ဂလာပါ။"

# Verbose mode (shows all the details)
burmese-tokenizer -v "မင်္ဂလာပါ။"

# Decode tokens back to text
burmese-tokenizer -d -t "▁မင်္ဂလာ,▁ပါ,။"
```

## API

- `encode(text)` - Chop text into tokens
- `decode(pieces)` - Glue tokens back together
- `decode_ids(ids)` - Convert IDs back to text
- `get_vocab_size()` - How many tokens we know
- `get_vocab()` - The whole vocabulary

## Dev Setup

```bash
git clone git@github.com:Code-Yay-Mal/burmese_tokenizer.git
cd burmese_tokenizer
uv sync --dev
uv run pytest

uv build
uv build --no-sources 
# make sure to have pypirc
uv run twine upload dist/*  or uv publish

# or publish with gh-action
git tag v0.1.2 
git push origin v0.1.2 

# if something goes wrong delete and restart all over again
git tag -d v0.1.2 && git push origin :refs/tags/v0.1.2 

```

## License

MIT - do whatever you want with it.
