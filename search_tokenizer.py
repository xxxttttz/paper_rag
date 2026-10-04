"""Dependency-free lexical tokenizer for mixed Chinese and English paper search."""

import re
import unicodedata


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]+")


def tokenize_search_text(text: str) -> list[str]:
    """Normalize names and emit Latin words plus overlapping Chinese bigrams."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = []
    for match in _TOKEN_PATTERN.finditer(normalized):
        value = match.group()
        if "\u3400" <= value[0] <= "\u9fff" and len(value) > 1:
            tokens.extend(value[index:index + 2] for index in range(len(value) - 1))
        else:
            tokens.append(value)
    return tokens
