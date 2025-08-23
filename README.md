# Burmese Tokenizer

Simple, fast Burmese text tokenization. No fancy stuff, just gets the job done.

## Install

```bash
pip install burmese-tokenizer
```

## Quick Start

```python
from burmese_tokenizer import BurmeseTokenizer

tokenizer = BurmeseTokenizer()
text = "မင်္ဂလာပါ။ နေကောင်းပါသလား။"

# tokenize
tokens = tokenizer.encode(text)
print(tokens["pieces"])
# ['▁မင်္ဂလာ', '▁ပါ', '။', '▁နေ', '▁ကောင်း', '▁ပါ', '▁သလား', '။']

# decode back
text = tokenizer.decode(tokens["pieces"])
print(text)
# မင်္ဂလာပါ။ နေကောင်းပါသလား။
```

## CLI

```bash
# tokenize
burmese-tokenizer "မင်္ဂလာပါ။"

# show details
burmese-tokenizer -v "မင်္ဂလာပါ။"

# decode tokens
burmese-tokenizer -d -t "▁မင်္ဂလာ,▁ပါ,။"
```

## API

- `encode(text)` - tokenize text
- `decode(pieces)` - convert tokens back to text  
- `decode_ids(ids)` - convert ids to text
- `get_vocab_size()` - vocabulary size
- `get_vocab()` - full vocabulary

## Links

- [PyPI](https://pypi.org/project/burmese-tokenizer/)
- [Contributing](docs/how_to_contribute.md)

## License

MIT - Do whatever you want with it.