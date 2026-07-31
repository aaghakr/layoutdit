import unittest

import torch

from cgbdm.module import Block


class SpatialAttentionTests(unittest.TestCase):
    def test_inactive_spatial_rows_are_finite_and_exactly_gated(self):
        torch.manual_seed(7)
        block = Block(
            d_model=8,
            nhead=2,
            dim_feedforward=16,
            dropout=0.0,
            diffusion_steps=10,
        ).eval()
        source = torch.randn(2, 3, 8)
        timestep = torch.tensor([1, 1])
        spatial = torch.randn(2, 4, 8)
        # Row 0 represents a prompt without spatial language. One dummy key is
        # deliberately visible so attention remains finite; the row-level gate
        # must remove its complete update.
        spatial_mask = torch.tensor(
            [[False, True, True, True], [False, False, True, True]]
        )
        active = torch.tensor([False, True])

        without_spatial = block(
            source,
            None,
            None,
            None,
            timestep=timestep,
        )
        with_mixed_spatial = block(
            source,
            None,
            None,
            None,
            intent_encode=spatial,
            timestep=timestep,
            spatial_key_padding_mask=spatial_mask,
            spatial_condition_active=active,
        )

        self.assertTrue(torch.isfinite(with_mixed_spatial).all())
        self.assertTrue(
            torch.allclose(without_spatial[0], with_mixed_spatial[0], atol=1e-6)
        )
        self.assertFalse(
            torch.allclose(without_spatial[1], with_mixed_spatial[1], atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
