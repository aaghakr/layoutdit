"""
Text-Spatial Grounding Module

Parses position keywords from text prompts and converts them to spatial
guidance boxes. When positions are found in the prompt, text-derived
spatial boxes override intent-map boxes, allowing explicit user control
over element placement.

Workflow:
    Training:
        - Parse positions from prompt -> match to GT element slots
        - Use text_spatial_boxes (class+position) through text_spatial_encoder
        - Mixed: 50% prompts have positions (use text_spatial), 50% don't (use intent)
    Inference:
        - Parse user prompt -> create text_spatial_boxes
        - If positions found -> override intent; if not -> fallback to intent
"""

import re
import torch
from typing import Dict, List, Optional, Tuple


# Canvas 3x3 grid regions: (cx, cy) in [0, 1]
POSITION_REGIONS = {
    'top-left':      (0.167, 0.167),
    'top-center':    (0.500, 0.167),
    'top-right':     (0.833, 0.167),
    'middle-left':   (0.167, 0.500),
    'middle-center': (0.500, 0.500),
    'middle-right':  (0.833, 0.500),
    'bottom-left':   (0.167, 0.833),
    'bottom-center': (0.500, 0.833),
    'bottom-right':  (0.833, 0.833),
}

REGION_W = 0.333
REGION_H = 0.333

# Canonical class name aliases (lowercase -> canonical)
CLASS_NAME_ALIASES = {
    'text': 'Text', 'texts': 'Text',
    'text box': 'Text', 'text boxes': 'Text',
    'text field': 'Text', 'text fields': 'Text',
    'logo': 'Logo', 'logos': 'Logo',
    'icon': 'Logo', 'icons': 'Logo',
    'brand': 'Logo', 'brand mark': 'Logo', 'brand marks': 'Logo',
    'underlay': 'Underlay', 'underlays': 'Underlay',
    'panel': 'Underlay', 'panels': 'Underlay',
    'background panel': 'Underlay', 'background panels': 'Underlay',
    'embellishment': 'Embellishment', 'embellishments': 'Embellishment',
    'decoration': 'Embellishment', 'decorations': 'Embellishment',
    'graphic element': 'Embellishment', 'graphic elements': 'Embellishment',
}

PKU_CLASS_MAP = {'Text': 1, 'Logo': 2, 'Underlay': 3}
CGL_CLASS_MAP = {'Text': 1, 'Logo': 2, 'Underlay': 3, 'Embellishment': 4}

# Count words -> int
_WORD_TO_NUM = {
    'a': 1, 'a single': 1, 'one': 1, 'two': 2, 'three': 3,
    'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
    'nine': 9, 'ten': 10,
}

# Pre-compiled regex patterns
_POSITION_PATTERN = '|'.join(re.escape(pos) for pos in POSITION_REGIONS)
_CLASS_NAMES_SORTED = sorted(CLASS_NAME_ALIASES.keys(), key=lambda x: -len(x))
_CLASS_PATTERN = '|'.join(re.escape(cn) for cn in _CLASS_NAMES_SORTED)
_COUNT_TOKEN_PATTERN = r'\d+|a single|one|two|three|four|five|six|seven|eight|nine|ten|a'
_COUNT_PATTERN = rf'({_COUNT_TOKEN_PATTERN})'
_DESCRIPTORS_PATTERN = r'(?:[\w-]+\s+){0,5}?'
_POST_CLASS_NOUN_PATTERN = (
    r'(?:\s+(?:line|lines|block|blocks|feature|features|'
    r'box|boxes|field|fields|panel|panels|mark|marks|icon|icons))?'
)
_POSITION_LIST_PATTERN = (
    rf'(?:{_POSITION_PATTERN})'
    rf'(?:\s*(?:,\s*(?:and\s+)?|and\s+)(?:the\s+)?(?:{_POSITION_PATTERN}))*'
)
_FULL_PATTERN = re.compile(
    rf'(?P<count>{_COUNT_TOKEN_PATTERN})\s+'
    rf'{_DESCRIPTORS_PATTERN}(?P<class>{_CLASS_PATTERN})'
    rf'{_POST_CLASS_NOUN_PATTERN}\s+(?:at|in|on)\s+(?:the\s+)?'
    rf'(?P<positions>{_POSITION_LIST_PATTERN})',
    re.IGNORECASE
)
_IMPLICIT_PATTERN = re.compile(
    rf'{_DESCRIPTORS_PATTERN}(?P<class>{_CLASS_PATTERN})'
    rf'{_POST_CLASS_NOUN_PATTERN}\s+(?:at|in|on)\s+(?:the\s+)?'
    rf'(?P<positions>{_POSITION_LIST_PATTERN})',
    re.IGNORECASE
)
_GROUPED_PATTERN = re.compile(
    rf'(?P<count>{_COUNT_TOKEN_PATTERN})\s+'
    rf'{_DESCRIPTORS_PATTERN}(?P<class>{_CLASS_PATTERN})'
    rf'{_POST_CLASS_NOUN_PATTERN}\s+with\s+(?P<assignments>.+?)'
    rf'(?=(?:,\s*{_COUNT_PATTERN}\s+(?:{_CLASS_PATTERN})\s+with\b)|[.;]|$)',
    re.IGNORECASE
)
_GROUP_ASSIGNMENT_PATTERN = re.compile(
    rf'{_COUNT_PATTERN}\s+at\s+(?:the\s+)?(?P<position>{_POSITION_PATTERN})',
    re.IGNORECASE
)


def _count_to_int(count_str: str) -> int:
    count = _WORD_TO_NUM.get(count_str.lower().strip())
    if count is not None:
        return count
    try:
        return int(count_str)
    except ValueError:
        return 1


def position_to_box(position_name: str) -> List[float]:
    """Convert position keyword to [cx, cy, w, h] in [0, 1]."""
    cx, cy = POSITION_REGIONS.get(position_name, (0.5, 0.5))
    return [cx, cy, REGION_W, REGION_H]


def _positions_from_text(value: str) -> List[str]:
    return [
        match.group(0).lower()
        for match in re.finditer(_POSITION_PATTERN, value or "", re.IGNORECASE)
    ]


def _spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _append_assignment(
    result: Dict[str, List[str]],
    class_str: str,
    positions: List[str],
    count: Optional[int] = None,
) -> None:
    canonical = CLASS_NAME_ALIASES.get(class_str.lower().strip())
    if canonical is None or not positions:
        return
    target_count = count if count is not None else max(1, len(positions))
    if len(positions) == 1:
        expanded = positions * target_count
    elif target_count <= len(positions):
        expanded = positions[:target_count]
    else:
        expanded = positions + [positions[-1]] * (target_count - len(positions))
    result.setdefault(canonical, []).extend(expanded)


def parse_positions_from_prompt(prompt: str) -> Dict[str, List[str]]:
    """
    Parse prompt to extract per-class position assignments.

    Handles formats like:
        "1 Logo at bottom-right, 2 Texts in the top-center"
        "A layout with 1 Logo in the bottom-right, 2 Texts at top-center."
        "3 Texts with 2 at top-center and 1 at bottom-left."
        "large title text at top-center and a small logo at bottom-left"

    Returns:
        Dict mapping canonical class name to list of position keywords.
        e.g. {'Logo': ['bottom-right'], 'Text': ['top-center', 'top-center']}
    """
    if not prompt:
        return {}

    result: Dict[str, List[str]] = {}
    consumed_spans: List[Tuple[int, int]] = []
    for match in _FULL_PATTERN.finditer(prompt):
        count = _count_to_int(match.group("count"))
        positions = _positions_from_text(match.group("positions"))
        _append_assignment(result, match.group("class"), positions, count)
        consumed_spans.append(match.span())

    # Backward compatibility for earlier generated spatial prompts:
    #   "3 Texts with 2 at top-center and 1 at bottom-left"
    # The text-spatial encoder needs class names attached to the request, so
    # expand each grouped assignment into per-class position occurrences.
    for match in _GROUPED_PATTERN.finditer(prompt):
        class_str = match.group("class")
        assignments = match.group("assignments")
        for assignment in _GROUP_ASSIGNMENT_PATTERN.finditer(assignments):
            count_str = assignment.group(1)
            position_str = assignment.group("position")
            count = _count_to_int(count_str)
            _append_assignment(result, class_str, [position_str.lower()], count)
        consumed_spans.append(match.span())

    # Free-form support: infer one request per position when authors omit a
    # rigid count but write descriptive phrases such as "large title text at
    # top-center" or "small logo at bottom-left".
    for match in _IMPLICIT_PATTERN.finditer(prompt):
        if any(_spans_overlap(match.span(), span) for span in consumed_spans):
            continue
        positions = _positions_from_text(match.group("positions"))
        _append_assignment(result, match.group("class"), positions, None)

    return result


def has_spatial_info(prompt: str) -> bool:
    """Check if prompt contains any parseable spatial position keywords."""
    return bool(parse_positions_from_prompt(prompt))


def get_class_map(num_class: int) -> Dict[str, int]:
    # num_class includes padding class 0: PKU=4, CGL=5.
    return CGL_CLASS_MAP if num_class >= 5 else PKU_CLASS_MAP


def create_text_spatial_boxes(
    prompt: str,
    max_elem: int,
    num_class: int,
    gt_classes: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Create text-spatial guidance boxes from a prompt.

    Each box: [class_one_hot (num_class), cx, cy, w, h] in [-1, 1] space.

    Args:
        prompt: Text prompt string.
        max_elem: Maximum number of layout elements.
        num_class: Number of element classes.
        gt_classes: [max_elem] tensor of GT class indices (1-indexed, 0=pad).
                    Used during training to match positions to correct slots.

    Returns:
        boxes: [max_elem, num_class + 4] or None if no positions found.
        mask:  [max_elem] bool tensor (True = ignore this slot) or None.
    """
    parsed = parse_positions_from_prompt(prompt)
    if not parsed:
        return None, None

    boxes = torch.zeros(max_elem, num_class + 4)
    mask = torch.ones(max_elem, dtype=torch.bool)  # True = ignore
    class_map = get_class_map(num_class)
    idx_to_name = {v: k for k, v in class_map.items()}

    if gt_classes is not None:
        # Training: match parsed positions to GT element slots in order
        queues = {cn: list(positions) for cn, positions in parsed.items()}

        for i in range(max_elem):
            cls_idx = gt_classes[i].item() if isinstance(gt_classes, torch.Tensor) else int(gt_classes[i])
            if cls_idx == 0:
                continue

            cls_name = idx_to_name.get(cls_idx)
            if cls_name and cls_name in queues and queues[cls_name]:
                pos_name = queues[cls_name].pop(0)
                _fill_slot(boxes, mask, i, cls_idx, num_class, pos_name)
    else:
        # Inference: assign positions to slots sequentially
        slot = 0
        for cls_name, positions in parsed.items():
            cls_idx = class_map.get(cls_name)
            if cls_idx is None:
                continue
            for pos_name in positions:
                if slot >= max_elem:
                    break
                _fill_slot(boxes, mask, slot, cls_idx, num_class, pos_name)
                slot += 1

    if mask.all():
        return None, None
    return boxes, mask


def _fill_slot(
    boxes: torch.Tensor,
    mask: torch.Tensor,
    slot: int,
    cls_idx: int,
    num_class: int,
    pos_name: str,
):
    """Fill a single slot in the text_spatial_boxes tensor."""
    pos_box = position_to_box(pos_name)
    # Class one-hot (0-indexed)
    boxes[slot, cls_idx - 1] = 1.0
    # Position in [-1, 1]
    boxes[slot, num_class:] = torch.tensor([2 * (v - 0.5) for v in pos_box])
    mask[slot] = False


def create_text_spatial_boxes_batch(
    prompts: List[str],
    max_elem: int,
    num_class: int,
    gt_classes_batch: Optional[torch.Tensor] = None,
    device: str = 'cpu',
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """
    Batch version of create_text_spatial_boxes.

    Returns:
        boxes:            [B, max_elem, num_class + 4] or None if no item has positions.
        mask:             [B, max_elem] bool or None.
        use_text_spatial: [B] bool — True for items with parsed positions.
    """
    B = len(prompts)
    all_boxes = torch.zeros(B, max_elem, num_class + 4, device=device)
    all_masks = torch.ones(B, max_elem, dtype=torch.bool, device=device)
    use_flag = torch.zeros(B, dtype=torch.bool, device=device)

    any_spatial = False
    for b, prompt in enumerate(prompts):
        if prompt is None:
            continue
        gt = gt_classes_batch[b] if gt_classes_batch is not None else None
        bx, mk = create_text_spatial_boxes(prompt, max_elem, num_class, gt)
        if bx is not None:
            all_boxes[b] = bx.to(device)
            all_masks[b] = mk.to(device)
            use_flag[b] = True
            any_spatial = True

    if not any_spatial:
        return None, None, use_flag
    return all_boxes, all_masks, use_flag
