import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add intent_detect to path for intent map generation
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'intent_detect'))

from PIL import Image
import torch
from transformers import set_seed, BertTokenizer
from torchvision import transforms
import numpy as np
import argparse
import cv2

from utils.visualize import draw_image
from cgbdm.diffusion import Diffusion
from cgbdm.text_spatial import has_spatial_info, create_text_spatial_boxes
from utils.util import box_xyxy_to_cxcywh, finalize, load_config, rebase_project_path
from data_process.generate_sal_box import find_bounding_box

# Import intent detection utilities
try:
    from model import design_intent_detector
    from segmentation_models_pytorch.encoders import get_preprocessing_fn
    INTENT_DETECTION_AVAILABLE = True
except Exception as exc:
    INTENT_DETECTION_AVAILABLE = False
    INTENT_DETECTION_IMPORT_ERROR = exc


def get_sal_box(img_sal_map, width, height, device):
    """Get saliency bounding box from saliency map"""
    sal_box = find_bounding_box(img_sal_map, ifpath=False)
    sal_box = box_xyxy_to_cxcywh(torch.tensor(sal_box)).float()
    sal_box[:, ::2] /= width
    sal_box[:, 1::2] /= height
    sal_box = sal_box.unsqueeze(0).unsqueeze(0).to(device)
    sal_box = 2 * (sal_box - 0.5)
    return sal_box


def get_intent_regions(design_intent_map, pad=False, kernel_n=37):
    """Extract bounding boxes from design intent map (similar to getRegions in map2box.py)"""
    if pad:
        design_intent_map = np.pad(design_intent_map[5:-5, 5:-5], 10, mode='constant', constant_values=0)

    threshold_value = design_intent_map.mean()
    _, binary = cv2.threshold(design_intent_map, threshold_value, 255, cv2.THRESH_BINARY)

    kernel = (kernel_n, kernel_n)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones(kernel, np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones(kernel, np.uint8))
    edges = cv2.Canny(binary, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        bbox.append((x, y, x+w, y+h))

    return bbox


def get_intent_box(img_intent_map, width, height, device):
    """Get intent bounding box from intent map"""
    # Convert PIL to numpy array if needed
    if isinstance(img_intent_map, Image.Image):
        intent_array = np.array(img_intent_map.convert("L"))
    else:
        intent_array = np.array(img_intent_map)

    # Resize to original image size for box extraction
    intent_resized = cv2.resize(intent_array, (width, height))

    # Get bounding boxes
    bboxes = get_intent_regions(intent_resized, pad=False, kernel_n=37)

    # Use the largest box or combine all boxes
    if not bboxes:
        # Fallback: use entire image
        bbox = (0, 0, width, height)
    else:
        # Use the largest box by area
        bbox = max(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))

    intent_box = box_xyxy_to_cxcywh(torch.tensor(bbox)).float()
    intent_box[:, ::2] /= width
    intent_box[:, 1::2] /= height
    intent_box = intent_box.unsqueeze(0).unsqueeze(0).to(device)
    intent_box = 2 * (intent_box - 0.5)
    return intent_box


def get_intent_map(img, intent_model_path, device):
    """Generate intent map from image using intent detection model"""
    if not INTENT_DETECTION_AVAILABLE:
        raise RuntimeError(
            "Intent detection modules are unavailable: "
            f"{INTENT_DETECTION_IMPORT_ERROR}"
        )

    # Load model
    model = design_intent_detector(act='none', action='forward')
    model.load_state_dict(torch.load(intent_model_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()

    # Preprocess image
    pp = get_preprocessing_fn('mit_b1', pretrained='imagenet')
    img_rgb = img.convert('RGB')
    img_resized = img_rgb.resize((224, 224))
    arr = np.array(img_resized)

    transform = transforms.Compose([
        lambda x: cv2.resize(x, (224, 224)),
        pp,
        transforms.ToTensor()
    ])
    image_tensor = transform(arr).float().unsqueeze(0).to(device)

    # Generate intent map
    with torch.no_grad():
        pred = model(image_tensor)
        pred = pred.squeeze().cpu().numpy()

        # Convert to PIL Image (grayscale)
        pred_img = (pred * 255).astype(np.uint8)
        intent_map = Image.fromarray(pred_img).convert("L")

    return intent_map


def tokenize_text(prompt, tokenizer, max_length=128, keep_batch_dim=True):
    """Tokenize text prompt for text control. Keeps batch dim [1, seq_len] for BERT."""
    text_encoding = tokenizer(
        prompt,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    if keep_batch_dim:
        text_features = {
            'input_ids': text_encoding['input_ids'],
            'attention_mask': text_encoding['attention_mask']
        }
    else:
        text_features = {
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0)
        }
    return text_features


def main(opt):
    seed = opt.seed
    set_seed(seed)

    device = torch.device(f"cuda:{opt.gpuid}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpuid)

    cfg = load_config(
        f'configs/{opt.render_style}_anno_test.yaml',
        path_profile=getattr(opt, 'path_profile', None),
    )
    cfg.v_encoder = opt.v_encoder
    cfg.spatial_guidance = opt.spatial_guidance
    cfg.text_control = opt.text_control
    print(f"Path profile: {cfg.path_profile} ({cfg.project_root})")

    print(f"Using {cfg.v_encoder} encoder")
    print(f"Using spatial guidance: {cfg.spatial_guidance}")
    print(f"Text control enabled: {cfg.text_control}")

    # Validate spatial_guidance mode
    if cfg.spatial_guidance not in [0, 1, 2]:
        raise ValueError(f"Invalid spatial_guidance: {cfg.spatial_guidance}. Must be 0 (saliency), 1 (intent), or 2 (both)")

    # Load input image
    img_inp = Image.open(opt.image_path).convert("RGB")
    cfg.width, cfg.height = img_inp.size

    # Initialize transforms
    transform_rgb = transforms.Compose([
        transforms.Resize([384, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    transform_l = transforms.Compose([
        transforms.Resize([384, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
    ])

    # Handle different spatial guidance modes
    sal_box = None
    intent_box = None
    img_tensor_parts = [transform_rgb(img_inp)]

    def _load_precomputed_saliency() -> Image.Image:
        # If user provides precomputed saliency paths, prefer them to avoid requiring saliency model weights.
        if getattr(opt, "saliency_path", ""):
            sal = Image.open(opt.saliency_path).convert("L")
            if getattr(opt, "saliency_sub_path", ""):
                sal_sub = Image.open(opt.saliency_sub_path).convert("L")
                sal = Image.fromarray(np.maximum(np.array(sal), np.array(sal_sub))).convert("L")
            return sal
        # Fallback: run saliency detector (requires local weights configured in data_process/saliency_detection.py).
        from data_process.saliency_detection import get_saliency_map  # lazy import to avoid hardcoded WEIGHT_ROOT assert
        return get_saliency_map(img_inp)

    if cfg.spatial_guidance == 0:  # Saliency only
        img_sal_map = _load_precomputed_saliency()
        sal_box = get_sal_box(img_sal_map, cfg.width, cfg.height, device)
        img_tensor_parts.append(transform_l(img_sal_map))

    elif cfg.spatial_guidance == 1:  # Intent only
        # Auto-detect intent model path if not provided
        if not opt.intent_model_path:
            default_intent_path = os.path.join(
                cfg.project_root,
                "data", "model_weights", "intent_map",
                f"design_intent_{opt.render_style}_epoch100.pth",
            )
            if opt.render_style == 'cgl':
                default_intent_path = default_intent_path.replace('epoch100', 'epoch35')
            if os.path.exists(default_intent_path):
                opt.intent_model_path = default_intent_path
                print(f"Using default intent model: {opt.intent_model_path}")
            else:
                raise ValueError(
                    f"--intent_model_path is required when spatial_guidance=1 (intent only). "
                    f"Default path not found: {default_intent_path}"
                )
        img_intent_map = get_intent_map(img_inp, opt.intent_model_path, device)
        # For intent only mode, sal_box parameter receives intent_box (as per dataloader convention)
        sal_box = get_intent_box(img_intent_map, cfg.width, cfg.height, device)
        intent_box = None  # Not passed separately for mode 1
        img_tensor_parts.append(transform_l(img_intent_map))

    elif cfg.spatial_guidance == 2:  # Both saliency and intent
        img_sal_map = _load_precomputed_saliency()
        sal_box = get_sal_box(img_sal_map, cfg.width, cfg.height, device)
        img_tensor_parts.append(transform_l(img_sal_map))

        # Auto-detect intent model path if not provided
        if not opt.intent_model_path:
            default_intent_path = os.path.join(
                cfg.project_root,
                "data", "model_weights", "intent_map",
                f"design_intent_{opt.render_style}_epoch100.pth",
            )
            if opt.render_style == 'cgl':
                default_intent_path = default_intent_path.replace('epoch100', 'epoch35')
            if os.path.exists(default_intent_path):
                opt.intent_model_path = default_intent_path
                print(f"Using default intent model: {opt.intent_model_path}")
            else:
                raise ValueError(
                    f"--intent_model_path is required when spatial_guidance=2 (both). "
                    f"Default path not found: {default_intent_path}"
                )
        img_intent_map = get_intent_map(img_inp, opt.intent_model_path, device)
        intent_box = get_intent_box(img_intent_map, cfg.width, cfg.height, device)
        img_tensor_parts.append(transform_l(img_intent_map))

    # Concatenate image tensor parts
    img_tensor = torch.concat(img_tensor_parts)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Handle text control
    text_features = None
    if cfg.text_control:
        if not opt.text_prompt:
            print("Warning: text_control enabled but no text_prompt provided. Using empty prompt.")
            opt.text_prompt = ""
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        text_features = tokenize_text(opt.text_prompt, tokenizer)
        text_features = {
            'input_ids': text_features['input_ids'].to(device),
            'attention_mask': text_features['attention_mask'].to(device)
        }

    # Initialize diffusion model
    diffusion_model = Diffusion(num_timesteps=1000,
                                ddim_num_steps=100,
                                n_head=cfg.n_head,
                                dim_model=cfg.d_model,
                                feature_dim=cfg.feature_dim,
                                seq_dim=cfg.num_class + 4,
                                num_layers=cfg.n_layers,
                                device=device,
                                max_elem=cfg.max_elem,
                                v_encoder=cfg.v_encoder,
                                spatial_guidance=cfg.spatial_guidance,
                                text_control=getattr(cfg, 'text_control', False),
                                ddim_schedule=opt.ddim_schedule)

    checkpoint_path = rebase_project_path(
        opt.check_path, getattr(opt, 'path_profile', None)
    )
    weights = torch.load(checkpoint_path, map_location=device)
    diffusion_model.load_diffusion_net(weights)
    diffusion_model.model.eval()

    # Build text-spatial guidance if prompt contains position keywords
    ts_boxes = ts_mask = use_ts = None
    if cfg.text_control and opt.text_prompt and has_spatial_info(opt.text_prompt):
        bx, mk = create_text_spatial_boxes(
            opt.text_prompt, cfg.max_elem, cfg.num_class, gt_classes=None
        )
        if bx is not None:
            ts_boxes = bx.unsqueeze(0).to(device)
            ts_mask = mk.unsqueeze(0).to(device)
            use_ts = torch.ones(1, dtype=torch.bool, device=device)
            print(f"[text-spatial] Detected positions in prompt — overriding intent guidance")

    bbox, cls, _ = diffusion_model.reverse_ddim(
        img_tensor, sal_box, cfg, save_inter=False,
        intent_box=intent_box, text_features=text_features,
        text_spatial_boxes=ts_boxes,
        text_spatial_mask=ts_mask,
        use_text_spatial=use_ts,
    )

    # Save output
    current_dir = os.getcwd()
    imgname = os.path.basename(opt.image_path)
    img_name = 'render_' + imgname
    draw_image(bbox, cls, img_inp, img_name, cfg.width, cfg.height, cfg.num_class, save_dir=current_dir)
    print(f"Output saved to: {os.path.join(current_dir, img_name)}")


# Start with main code
if __name__ == "__main__":
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpuid',
        type=int,
        default=0,
        help='choose gpu')
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='choose seed')
    parser.add_argument(
        '--render_style',
        type=str,
        default='pku',
        help='(pku, cgl)')
    parser.add_argument(
        '--image_path',
        type=str,
        default='',
        help='choose image')
    parser.add_argument(
        '--check_path',
        type=str,
        default='',
        help='choose checkpoint')
    parser.add_argument(
        '--v_encoder',
        type=str,
        default='vit',
        help='choose visual encoder(vit, swin, convnext)')
    parser.add_argument(
        '--spatial_guidance',
        type=int,
        default=0,
        help='Spatial Guidance: 0=saliency only, 1=intent only, 2=both')
    parser.add_argument(
        '--text_control',
        action='store_true',
        help='Enable text control for layout generation')
    parser.add_argument(
        '--text_prompt',
        type=str,
        default='',
        help='Text prompt for layout generation (required if --text_control is set)')
    parser.add_argument(
        '--intent_model_path',
        type=str,
        default='',
        help='Path to intent detection model checkpoint (required if spatial_guidance=1 or 2). '
             'If not provided, will try default path based on render_style.')
    parser.add_argument(
        '--saliency_path',
        type=str,
        default='',
        help='Optional path to a precomputed saliency map (grayscale image). If set, avoids running saliency detector.')
    parser.add_argument(
        '--saliency_sub_path',
        type=str,
        default='',
        help='Optional path to a secondary saliency map; if set, combines with saliency_path via per-pixel max.')
    parser.add_argument(
        '--path-profile', '--path_profile',
        dest='path_profile',
        choices=('local', 'server'),
        default='local',
        help='Path profile: local=current checkout; server=/home/viplab/Aagha/intent_aware_layout_generation')
    parser.add_argument(
        '--ddim-schedule', '--ddim_schedule',
        dest='ddim_schedule',
        choices=('training', 'cosine', 'linear'),
        default='cosine',
        help="Cumulative alpha schedule used by DDIM sampling. Default 'cosine' matches the trained DDPM schedule; 'linear' is retained only for legacy/sensitivity checks.")
    opt = parser.parse_args()
    main(opt)
