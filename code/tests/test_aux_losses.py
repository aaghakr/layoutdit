import unittest

import torch

from utils.aux_losses import expected_counts_from_prompts, loss_count, loss_place


class AuxLossTests(unittest.TestCase):
    def test_prompt_counts_exclude_padding_class(self):
        counts = expected_counts_from_prompts(
            ["two texts and one logo"], num_class=4, device="cpu"
        )
        self.assertEqual(tuple(counts.shape), (1, 3))
        self.assertTrue(torch.equal(counts, torch.tensor([[2.0, 1.0, 0.0]])))

    def test_count_loss_backpropagates_to_class_logits(self):
        layout = torch.randn(2, 4, 8, requires_grad=True)
        loss = loss_count(
            layout,
            ["two texts and one logo", "one text"],
            num_class=4,
            device="cpu",
        )
        loss.backward()
        self.assertIsNotNone(layout.grad)
        self.assertGreater(float(layout.grad[:, :, :4].abs().sum()), 0.0)

    def test_place_loss_ignores_negative_one_padding_and_backpropagates(self):
        layout = torch.zeros(1, 2, 8)
        layout[:, :, 1] = 5.0  # valid Text slots
        layout = layout.requires_grad_()
        placement = torch.tensor([[[0.2, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]]])
        loss = loss_place(layout, placement, num_class=4, device="cpu")
        loss.backward()
        self.assertGreater(float(layout.grad[:, :, 4:].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
