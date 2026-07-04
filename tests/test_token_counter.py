from token_counter import compression_stats, count_tokens


def test_count_tokens_nonzero_for_nonempty_text():
    assert count_tokens("hello world") > 0


def test_compression_stats_reports_full_reduction_for_empty_after():
    stats = compression_stats("some fairly long original context text", "")
    assert stats["tokens_after"] == 0
    assert stats["reduction_pct"] == 100.0
    assert stats["tokens_saved"] == stats["tokens_before"]


def test_compression_stats_identity_when_unchanged():
    stats = compression_stats("same text", "same text")
    assert stats["tokens_saved"] == 0
    assert stats["compression_ratio"] == 1.0
