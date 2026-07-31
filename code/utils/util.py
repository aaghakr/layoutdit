from __future__ import annotations

import sys

import torch
import numpy as np
import random
import os
import re
import fsspec
from torchvision.ops.boxes import box_area
from collections import OrderedDict
from typing import Callable, Optional, Union, Any
from torch import Tensor
import yaml
from datetime import datetime
import pytz
from pathlib import Path


LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVER_PROJECT_ROOT = Path("/home/viplab/Aagha/intent_aware_layout_generation")
PATH_PROFILES = {
    "local": LOCAL_PROJECT_ROOT,
    "server": SERVER_PROJECT_ROOT,
}

class Config:
    def __init__(self, config):
        for key, value in config.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)


def resolve_project_root(path_profile=None):
    """Resolve the active repository root for local/server execution.

    The command-line entrypoints pass ``--path-profile``. Other callers can set
    ``INTENTDIT_PATH_PROFILE=server``; otherwise local is the safe default.
    """
    profile = (path_profile or os.environ.get("INTENTDIT_PATH_PROFILE", "local")).lower()
    if profile not in PATH_PROFILES:
        choices = ", ".join(sorted(PATH_PROFILES))
        raise ValueError(f"Unknown path profile '{profile}'. Choose one of: {choices}")
    return profile, PATH_PROFILES[profile]


def rebase_project_path(path, path_profile=None):
    """Rebase a path from an older IntentDiT checkout to the active profile.

    Explicit external paths are left alone. Only paths recognizable as an
    IntentDiT project path are rewritten.
    """
    if not path:
        return path
    profile, project_root = resolve_project_root(path_profile)
    expanded = os.path.abspath(os.path.expanduser(os.path.expandvars(str(path))))
    project_markers = (
        "intent_aware_layout_generation",
        "intent_latest_backup",
        str(LOCAL_PROJECT_ROOT),
        str(SERVER_PROJECT_ROOT),
    )
    if not any(marker in expanded for marker in project_markers):
        return expanded
    for owned_dir in ("data", "experiments", "reviews_and_rebuttle", "user_study"):
        marker = f"/{owned_dir}/"
        if marker in expanded:
            suffix = expanded.split(marker, 1)[1]
            return str(project_root / owned_dir / suffix)
    return expanded


def apply_path_profile(config, path_profile=None):
    """Rebase project-owned data/checkpoint/output paths to one root."""
    profile, project_root = resolve_project_root(path_profile)
    dataset = config.get("dataset_cls") or config.get("dataset")

    config["path_profile"] = profile
    config["project_root"] = str(project_root)

    if dataset and "paths" in config:
        config["paths"]["base"] = str(project_root / "data" / "dataset" / dataset / "split")
    if dataset and "base_check_dir" in config:
        config["base_check_dir"] = str(project_root / "data" / "checkpoints" / dataset)
    if "imgname_order_dir" in config:
        config["imgname_order_dir"] = str(
            project_root / "data" / "output" / "ptfile" / "image_name_order"
        )
    if "save_imgs_dir" in config:
        config["save_imgs_dir"] = str(project_root / "data" / "output" / "image")
    return config


def process_paths(config):
    base = config['paths']['base']

    sections = ['test']
    if 'train' in config['paths']:
        sections.append('train')

    for section in sections:
        if section in config['paths']:
            for key, value in config['paths'][section].items():
                config['paths'][section][key] = os.path.join(base, value)

    return config

def load_config(config_path, path_profile=None):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config = apply_path_profile(config, path_profile)
    config = process_paths(config)

    china_tz = pytz.timezone('Asia/Shanghai')
    config['datetime'] = datetime.now(china_tz).strftime('%m_%d_%H%M')
    return Config(config)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0) , (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - w/2), (y_c - h/2),
         (x_c + w/2), (y_c + h/2)]
    return torch.stack(b, dim=-1)

def convert_xywh_to_ltrb(
    bbox: Union[Tensor, np.ndarray, list[float]]
) -> Union[list[Tensor], list[np.ndarray], list[float]]:
    # assert len(bbox) == 4
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check

    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # Keep exact LayoutDiT-compatible behavior for positional-embedding loading.
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num

def finalize(layout, num_class):
    bbox = layout[:, :, num_class:]
    bbox = torch.clamp(bbox, min=-1, max=1) / 2 + 0.5
    label = torch.argmax(layout[:, :, :num_class], dim=2)
    mask = (label != 0).clone().detach()
    label = label.unsqueeze(-1)
    return bbox, label, mask

def clamp_w_tol(
    value: float, tolerance: float = 5e-3, vmin: float = 0.0, vmax: float = 1.0
) -> float:
    """
    Clamp the value to [vmin, vmax] range with tolerance.
    """
    assert vmin - tolerance <= value <= vmax + tolerance, value
    return max(vmin, min(vmax, value))

def _compare(low: float, high: float) -> tuple[float, float]:
    if low > high:
        return high, low
    else:
        return low, high

def has_valid_area(width, height, thresh: float = 1e-3) -> bool:
    """
    Check whether the area is smaller than the threshold.
    """
    area = width * height
    valid = area > thresh
    return valid

def natural_sort_cmp(a, b):
    a_match = re.match(r'(\d+)\.(png|jpg)$', a, re.IGNORECASE)
    b_match = re.match(r'(\d+)\.(png|jpg)$', b, re.IGNORECASE)
    if a_match and b_match:
        a_num = int(a_match.group(1))
        b_num = int(b_match.group(1))
        return a_num - b_num
    elif a_match:
        return -1
    elif b_match:
        return 1
    else:
        return 0
