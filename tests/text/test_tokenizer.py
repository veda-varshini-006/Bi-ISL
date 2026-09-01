"""Unit tests for Bi-ISL Target Text Preprocessing & Tokenization Pipeline."""

import os
import tempfile
import pytest

from src.text.tokenizer import ISLTextTokenizer, TokenizerMetadata


def test_text_normalization_and_stripping():
    """Test text normalization, lowercasing, and punctuation stripping."""
    tokenizer = ISLTextTokenizer(lowercase=True, strip_punctuation=True)
    raw_text = "  Hello, World! This is ISL Translation.  "
    norm = tokenizer.normalize_text(raw_text)
    assert norm == "hello world this is isl translation"


def test_build_vocab_strictly_from_train():
    """Test building vocabulary strictly from training texts."""
    tokenizer = ISLTextTokenizer()
    train_texts = ["hello world", "welcome to indian sign language"]
    tokenizer.build_vocab(train_texts)

    assert "<pad>" in tokenizer.word2id
    assert "<unk>" in tokenizer.word2id
    assert "<bos>" in tokenizer.word2id
    assert "<eos>" in tokenizer.word2id
    assert "hello" in tokenizer.word2id
    assert "indian" in tokenizer.word2id


def test_encode_decode_round_trip():
    """Test sequence encoding, padding, truncation, and decoding round-trip."""
    tokenizer = ISLTextTokenizer(max_length=10)
    tokenizer.build_vocab(["good morning everyone"])

    ids = tokenizer.encode("good morning everyone", add_special_tokens=True)
    assert ids[0] == tokenizer.special_tokens["<bos>"]
    assert ids[-1] == tokenizer.special_tokens["<eos>"]

    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    assert decoded == "good morning everyone"


def test_vocab_leakage_check():
    """Test checking evaluation set vocabulary overlap and OOV words."""
    tokenizer = ISLTextTokenizer()
    tokenizer.build_vocab(["apple banana orange"])

    eval_texts = ["apple mango pineapple"]
    leakage_stats = tokenizer.check_vocab_leakage(eval_texts)

    assert leakage_stats["eval_word_count"] == 3
    assert leakage_stats["oov_word_count"] == 2  # mango, pineapple
    assert leakage_stats["vocab_overlap_count"] == 1  # apple
    assert leakage_stats["leakage_clean"] is True


def test_tokenizer_metadata_checkpoint_export():
    """Test exporting metadata dict for model checkpoints, saving, and loading JSON vocab."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tokenizer = ISLTextTokenizer(tokenizer_version="v1.0.0_test")
        tokenizer.build_vocab(["test vocabulary serialization"])

        meta = tokenizer.get_metadata()
        assert meta["tokenizer_version"] == "v1.0.0_test"
        assert meta["vocab_size"] > 4

        json_file = os.path.join(tmp_dir, "vocab.json")
        tokenizer.save_vocab(json_file)
        assert os.path.exists(json_file)

        tokenizer_loaded = ISLTextTokenizer()
        tokenizer_loaded.load_vocab(json_file)
        assert tokenizer_loaded.tokenizer_version == "v1.0.0_test"
        assert "vocabulary" in tokenizer_loaded.word2id
