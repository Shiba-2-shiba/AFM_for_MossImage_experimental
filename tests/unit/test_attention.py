import unittest
from unittest.mock import patch

import torch

from afm.config import AFMConfig, TokenLayout
from afm.attention import edit_image_rows, replace_rows
from afm.memory import plan_chunks, MemoryGuardError
from afm.spectral import filter_logits, schedule_alphas


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.q, self.k, self.v = [torch.randn(2, 3, 10, 4) for _ in range(3)]
        self.layout = TokenLayout((2, 3), 4)
        self.config = AFMConfig(mode="edit", strength=0.2)

    def test_chunked_kernel_matches_full_joint_oracle(self):
        q, k, v = self.q, self.k, self.v
        logits = q @ k.transpose(-2, -1) * 0.5
        logits[:, :, :6, 6:] = filter_logits(
            logits[:, :, :6, 6:], (2, 3), 1.2, 1, 0.25, True, 0.05, True)
        expected = (logits.softmax(-1) @ v)[:, :, :6]
        for heads, queries in [(1, 1), (2, 3), (3, 6)]:
            chunks = plan_chunks(q.shape, q.element_size(), self.layout, 64, heads, queries)
            got = edit_image_rows(q, k, v, self.layout, self.config, 0, chunks, 0.5)
            torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-5)
        text_only = logits[:, :, :6, 6:].softmax(-1) @ v[:, :, 6:]
        self.assertFalse(torch.allclose(expected, text_only))

    def test_manual_zero_matches_unedited_joint_attention(self):
        c = AFMConfig(mode="edit", strength=0)
        chunks = plan_chunks(self.q.shape, 4, self.layout, 64, 1, 2)
        got = edit_image_rows(self.q, self.k, self.v, self.layout, c, 0.6, chunks, 0.5)
        expected = ((self.q @ self.k.transpose(-2, -1) * 0.5).softmax(-1) @ self.v)[:, :, :6]
        torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-5)

    def test_text_key_product_is_not_recomputed_in_query_chunks(self):
        original = torch.Tensor.__matmul__
        key_lengths = []

        def matmul(left, right):
            # QK products have right's penultimate dimension equal to head_dim.
            if right.shape[-2] == 4:
                key_lengths.append(right.shape[-1])
            return original(left, right)

        chunks = plan_chunks(self.q.shape, 4, self.layout, 64, 1, 2)
        with patch.object(torch.Tensor, "__matmul__", matmul):
            edit_image_rows(self.q, self.k, self.v, self.layout,
                            self.config, 0, chunks, 0.5)
        self.assertNotIn(self.layout.total_tokens, key_lengths)
        self.assertIn(self.layout.image_tokens, key_lengths)

    def test_joint_oracle_all_progress_dtype_scale_and_chunks(self):
        for dtype, tolerance in ((torch.float32, 2e-5), (torch.bfloat16, 0.02),
                                 (torch.float16, 0.002)):
            q, k, v = [x.to(dtype) for x in (self.q, self.k, self.v)]
            for progress in (0, 0.5, 1):
                alpha = schedule_alphas(self.config.strength, progress)
                logits = q.float() @ k.float().transpose(-2, -1) * 0.3
                logits[:, :, :6, 6:] = filter_logits(
                    logits[:, :, :6, 6:], (2, 3), *alpha)
                expected = (logits.softmax(-1) @ v.float())[:, :, :6].to(dtype)
                for heads, queries in ((1, 2), (3, 6)):
                    chunks = plan_chunks(q.shape, q.element_size(), self.layout,
                                         64, heads, queries)
                    got = edit_image_rows(q, k, v, self.layout, self.config,
                                          progress, chunks, 0.3)
                    torch.testing.assert_close(got, expected, atol=tolerance, rtol=tolerance)

    def test_atomic_partial_replacement_preserves_other_rows(self):
        base = torch.randn(2, 10, 12)
        original = base.clone()
        candidate = replace_rows(base, torch.ones(1, 3, 6, 4), [1], self.layout, 3)
        self.assertTrue(torch.equal(base, original))
        self.assertTrue(torch.equal(candidate[0], base[0]))
        self.assertTrue(torch.equal(candidate[1, 6:], base[1, 6:]))
        self.assertTrue(torch.equal(candidate[1, :6], torch.ones(6, 12)))

    def test_nonfinite_candidate_never_changes_baseline(self):
        base = torch.randn(2, 10, 12)
        old = base.clone()
        with self.assertRaises(FloatingPointError):
            replace_rows(base, torch.full((1, 3, 6, 4), float("nan")), [0], self.layout, 3)
        self.assertTrue(torch.equal(base, old))

    def test_budget_rejection_and_chunk_reduction(self):
        layout = TokenLayout((64, 64), 4096)
        with self.assertRaises(MemoryGuardError):
            plan_chunks((2, 24, 8192, 128), 2, layout, 1, 2, 128)
        big = plan_chunks((1, 4, 80, 8), 4, TokenLayout((8, 8), 16), 64, 4, 64)
        small = plan_chunks((1, 4, 80, 8), 4, TokenLayout((8, 8), 16),
                            big.estimated_bytes / 1024**2 * 0.8, 4, 64)
        self.assertLessEqual(small.estimated_bytes, big.estimated_bytes * 0.8)
        self.assertTrue(small.head_chunk_size < 4 or small.query_chunk_size < 64)
