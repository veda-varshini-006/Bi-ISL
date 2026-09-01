"""Unit tests for Bi-ISL Autoregressive Translation Decoder Head."""

import pytest
import torch

from src.models.translation_decoder import AutoregressiveTranslationDecoder


def test_teacher_forcing_forward_pass():
    """Test teacher forcing forward pass during training."""
    decoder = AutoregressiveTranslationDecoder(vocab_size=50, embed_dim=32, encoder_dim=64, hidden_dim=64)
    dummy_encoder_out = torch.randn(2, 16, 64)
    dummy_targets = torch.randint(0, 50, (2, 10))
    dummy_mask = torch.ones(2, 16, dtype=torch.bool)

    logits = decoder(dummy_encoder_out, dummy_targets, encoder_mask=dummy_mask)
    assert logits.shape == (2, 10, 50)


def test_greedy_inference_decoding():
    """Test greedy autoregressive inference decoding."""
    decoder = AutoregressiveTranslationDecoder(vocab_size=50, embed_dim=32, encoder_dim=64, hidden_dim=64)
    dummy_encoder_out = torch.randn(2, 16, 64)

    tokens, finished = decoder.greedy_decode(dummy_encoder_out, max_len=12)
    assert tokens.shape == (2, 12)
    assert (tokens[:, 0] == decoder.bos_token_id).all()


def test_beam_search_decoding():
    """Test beam search decoding with length penalty."""
    decoder = AutoregressiveTranslationDecoder(vocab_size=50, embed_dim=32, encoder_dim=64, hidden_dim=64)
    dummy_encoder_out = torch.randn(2, 16, 64)

    beam_results = decoder.beam_search_decode(dummy_encoder_out, beam_size=3, max_len=12, length_penalty=0.6)
    assert len(beam_results) == 2
    assert isinstance(beam_results[0], list)


def test_deterministic_inference():
    """Test deterministic output generation given fixed seed and encoder representation."""
    torch.manual_seed(42)
    decoder = AutoregressiveTranslationDecoder(vocab_size=50, embed_dim=32, encoder_dim=64, hidden_dim=64)
    dummy_encoder_out = torch.randn(1, 16, 64)

    out1, _ = decoder.greedy_decode(dummy_encoder_out, max_len=10)
    out2, _ = decoder.greedy_decode(dummy_encoder_out, max_len=10)

    assert torch.equal(out1, out2)
