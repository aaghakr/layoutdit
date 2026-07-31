"""Differentiable auxiliary losses used by IntentDiT training."""

import torch
import torch.nn.functional as F

from utils.metric import CLASS_INDEX_TO_NAME, _parse_prompt_counts
from utils.util import box_cxcywh_to_xyxy, box_iou, finalize


def layout_expected_counts(layout_pred, num_class):
    """Return differentiable expected counts for non-padding classes.

    ``num_class`` includes class 0 (the padding/no-element class), so the
    returned tensor has shape ``[B, num_class - 1]``.
    """
    class_prob = torch.softmax(layout_pred[:, :, :num_class], dim=-1)
    return class_prob[:, :, 1:].sum(dim=1)


def expected_counts_from_prompts(prompt_texts, num_class, device):
    """Parse requested counts for the non-padding classes."""
    names = [CLASS_INDEX_TO_NAME.get(i, f"Class{i}") for i in range(1, num_class)]
    rows = []
    for prompt in prompt_texts:
        parsed = _parse_prompt_counts(prompt or "")
        rows.append([parsed.get(name, 0) for name in names])
    return torch.tensor(rows, dtype=torch.float32, device=device)


def loss_count(layout_pred, prompt_texts, num_class, device):
    """MSE between differentiable expected and prompt-requested counts."""
    if not prompt_texts or num_class <= 1:
        return torch.tensor(0.0, device=device)
    pred_counts = layout_expected_counts(layout_pred, num_class)
    target_counts = expected_counts_from_prompts(prompt_texts, num_class, device)
    return F.mse_loss(pred_counts, target_counts)


def loss_place(layout_pred, placement_box, num_class, device):
    """One minus mean best IoU between predicted and placement boxes.

    Placement boxes come from connected density regions and do not have a
    guaranteed one-to-one slot ordering with layout tokens.  Set matching by
    best IoU avoids imposing a false diagonal correspondence.
    """
    if placement_box is None or layout_pred is None:
        return torch.tensor(0.0, device=device)
    bbox, label, _ = finalize(layout_pred, num_class)
    valid = label.squeeze(-1) != 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    placement_01 = (placement_box / 2.0 + 0.5).clamp(0.0, 1.0)
    placement_valid = (placement_01[..., 2] > 1e-6) & (placement_01[..., 3] > 1e-6)
    bbox_xyxy = box_cxcywh_to_xyxy(bbox)
    placement_xyxy = box_cxcywh_to_xyxy(placement_01)

    best_ious = []
    for batch_idx in range(layout_pred.shape[0]):
        pred = bbox_xyxy[batch_idx][valid[batch_idx]]
        target = placement_xyxy[batch_idx][placement_valid[batch_idx]]
        if pred.numel() == 0 or target.numel() == 0:
            continue
        iou, _ = box_iou(pred, target)
        best_ious.append(iou.max(dim=1).values)

    if not best_ious:
        return torch.tensor(0.0, device=device)
    mean_iou = torch.cat(best_ious).mean()
    return (1.0 - mean_iou).clamp(min=0.0)
