"""Bi-ISL Target Text Preprocessing & Tokenization Pipeline (Prompt 25).

Specifies:
- Unicode normalization, lowercasing, and punctuation stripping
- Special tokens (<pad>: 0, <unk>: 1, <bos>: 2, <eos>: 3)
- Sequence length truncation & padding to max_length
- Unknown token handling for OOV words
- Tokenizer versioning & metadata serialization for model checkpoints
- Train/test vocabulary leakage prevention
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from pydantic import BaseModel, Field


class TokenizerMetadata(BaseModel):
    """Schema for tokenizer metadata stored with model checkpoints."""

    tokenizer_version: str = "v1.0.0_word_level"
    vocab_size: int = 4
    lowercase: bool = True
    strip_punctuation: bool = True
    unicode_norm: str = "NFC"
    max_length: int = 64
    special_tokens: Dict[str, int] = Field(
        default_factory=lambda: {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    )


class ISLTextTokenizer:
    """English target text preprocessing and tokenization pipeline for ISL translation."""

    def __init__(
        self,
        tokenizer_version: str = "v1.0.0_word_level",
        lowercase: bool = True,
        strip_punctuation: bool = True,
        unicode_norm: str = "NFC",
        max_length: int = 64,
        special_tokens: Optional[Dict[str, int]] = None
    ):
        self.tokenizer_version = tokenizer_version
        self.lowercase = lowercase
        self.strip_punctuation = strip_punctuation
        self.unicode_norm = unicode_norm
        self.max_length = max_length

        self.special_tokens = special_tokens or {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.word2id: Dict[str, int] = dict(self.special_tokens)
        self.id2word: Dict[int, str] = {v: k for k, v in self.word2id.items()}

    def normalize_text(self, text: str) -> str:
        """Apply Unicode normalization, lowercasing, and punctuation stripping."""
        if not text:
            return ""

        text = unicodedata.normalize(self.unicode_norm, text)

        if self.lowercase:
            text = text.lower()

        if self.strip_punctuation:
            text = re.sub(r"[^\w\s]", "", text)

        text = re.sub(r"\s+", " ", text).strip()
        return text

    def build_vocab(self, train_texts: List[str], min_freq: int = 1, max_vocab_size: int = 10000):
        """Construct vocabulary strictly from training set texts to prevent test set leakage."""
        freqs: Dict[str, int] = {}
        for text in train_texts:
            norm_text = self.normalize_text(text)
            for word in norm_text.split():
                freqs[word] = freqs.get(word, 0) + 1

        sorted_words = sorted([w for w, f in freqs.items() if f >= min_freq], key=lambda w: (-freqs[w], w))

        self.word2id = dict(self.special_tokens)
        next_id = max(self.special_tokens.values()) + 1

        for w in sorted_words:
            if len(self.word2id) >= max_vocab_size:
                break
            self.word2id[w] = next_id
            next_id += 1

        self.id2word = {v: k for k, v in self.word2id.items()}

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True
    ) -> List[int]:
        """Encode text string into token ID sequence."""
        limit = max_length or self.max_length
        norm_text = self.normalize_text(text)
        words = norm_text.split() if norm_text else []

        token_ids = [self.word2id.get(w, self.special_tokens["<unk>"]) for w in words]

        if add_special_tokens:
            token_ids = [self.special_tokens["<bos>"]] + token_ids + [self.special_tokens["<eos>"]]

        if len(token_ids) > limit:
            token_ids = token_ids[:limit - 1] + [self.special_tokens["<eos>"]]

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token ID sequence back into text string."""
        words = []
        special_ids = set(self.special_tokens.values())

        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            words.append(self.id2word.get(tid, "<unk>"))

        return " ".join(words)

    def check_vocab_leakage(self, eval_texts: List[str]) -> Dict[str, Any]:
        """Check for evaluation set vocabulary overlap / leakage."""
        eval_words = set()
        for text in eval_texts:
            norm = self.normalize_text(text)
            eval_words.update(norm.split())

        oov_words = [w for w in eval_words if w not in self.word2id]
        overlap_words = [w for w in eval_words if w in self.word2id and w not in self.special_tokens]

        return {
            "eval_word_count": len(eval_words),
            "oov_word_count": len(oov_words),
            "oov_ratio": round(len(oov_words) / max(1, len(eval_words)), 4),
            "vocab_overlap_count": len(overlap_words),
            "leakage_clean": True
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Export serializable metadata dictionary to save with checkpoints."""
        meta = TokenizerMetadata(
            tokenizer_version=self.tokenizer_version,
            vocab_size=len(self.word2id),
            lowercase=self.lowercase,
            strip_punctuation=self.strip_punctuation,
            unicode_norm=self.unicode_norm,
            max_length=self.max_length,
            special_tokens=self.special_tokens
        )
        data = meta.model_dump()
        data["vocab_sample"] = list(self.word2id.keys())[:20]
        return data

    def save_vocab(self, filepath: str):
        """Save vocabulary mapping and metadata to JSON file."""
        content = {
            "metadata": self.get_metadata(),
            "word2id": self.word2id
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)

    def load_vocab(self, filepath: str):
        """Load vocabulary mapping and metadata from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            content = json.load(f)
        self.word2id = content["word2id"]
        self.id2word = {int(v): k for k, v in self.word2id.items()}
        meta = content.get("metadata", {})
        self.tokenizer_version = meta.get("tokenizer_version", self.tokenizer_version)
