import sys
import os
import json
import csv
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from utils import logger
from torch.utils.data import DataLoader
from transformers import set_seed
from utils.metric import metric
from utils.benchmark_metrics import load_ground_truth_layouts
from utils.util import finalize, load_config, rebase_project_path
from utils.visualize import visualize_images

import argparse
from data_process.dataloader import test_uncond_dataset, test_cond_dataset, custom_collate_fn
from cgbdm.diffusion import Diffusion
from cgbdm.text_spatial import create_text_spatial_boxes_batch

def sample_uncond(diffusion_model, testing_dl, cfg):
    sample_output = []
    device = diffusion_model.device
    cnt = 0
    for idx, data in enumerate(testing_dl):
        # Handle different return values based on spatial guidance and text control
        text_features = None
        prompt_texts = None
        if len(data) == 2:  # Mode 0 or 1: single box type
            image, sal_box = data
            intent_box = None
        elif len(data) == 3:  # Mode 2: both saliency and intent, or Mode 0/1 with text
            if isinstance(data[2], dict):  # text_features is a dict
                image, sal_box, text_features = data
                intent_box = None
            else:  # intent_box
                image, sal_box, intent_box = data
        elif len(data) == 4:
            if isinstance(data[2], dict):  # Mode 0/1 with text + raw prompts
                image, sal_box, text_features, prompt_texts = data
                intent_box = None
            else:  # Legacy Mode 2 with text
                image, sal_box, intent_box, text_features = data
        elif len(data) == 5:  # Mode 2 with text + raw prompts
            image, sal_box, intent_box, text_features, prompt_texts = data
        else:
            raise ValueError(f"Unexpected number of return values: {len(data)}")

        image, sal_box = image.to(device), sal_box.to(device)
        if intent_box is not None:
            intent_box = intent_box.to(device)
            if getattr(cfg, 'disable_spatial_boxes', False):
                intent_box = None
        if text_features is not None:
            text_features = {
                'input_ids': text_features['input_ids'].to(device),
                'attention_mask': text_features['attention_mask'].to(device)
            }

        text_spatial_boxes = text_spatial_mask = use_text_spatial = None
        if prompt_texts is not None and not getattr(cfg, 'disable_text_spatial_parser', False):
            text_spatial_boxes, text_spatial_mask, use_text_spatial = (
                create_text_spatial_boxes_batch(
                    list(prompt_texts),
                    cfg.max_elem,
                    cfg.num_class,
                    device=device,
                )
            )

        bbox, cls, _ = diffusion_model.reverse_ddim(
            image,
            sal_box,
            cfg,
            save_inter=False,
            intent_box=intent_box,
            text_features=text_features,
            text_spatial_boxes=text_spatial_boxes,
            text_spatial_mask=text_spatial_mask,
            use_text_spatial=use_text_spatial,
        )
        samples = torch.cat([cls, bbox], dim=2)
        sample_output.append(samples.cpu())
        cnt = cnt + image.shape[0]
        logger.log(f"created {cnt} samples")

    sample_output = torch.concat(sample_output, dim=0)
    return sample_output
def sample_cond(diffusion_model, testing_dl, cfg, cond='c'):
    sample_output = []
    device = diffusion_model.device
    cnt = 0
    for idx, data in enumerate(testing_dl):
        # Handle different return values based on spatial guidance and text control
        text_features = None
        if len(data) == 3:  # Mode 0 or 1: single box type
            image, layout, sal_box = data
            intent_box = None
        elif len(data) == 4:  # Mode 2: both saliency and intent, or Mode 0/1 with text
            if isinstance(data[3], dict):  # text_features is a dict
                image, layout, sal_box, text_features = data
                intent_box = None
            else:  # intent_box
                image, layout, sal_box, intent_box = data
        elif len(data) == 5:  # Mode 2 with text
            image, layout, sal_box, intent_box, text_features = data
        else:
            raise ValueError(f"Unexpected number of return values: {len(data)}")

        image, layout, sal_box = image.to(device), layout.to(device), sal_box.to(device)
        if intent_box is not None:
            intent_box = intent_box.to(device)
            if getattr(cfg, 'disable_spatial_boxes', False):
                intent_box = None
        if text_features is not None:
            text_features = {
                'input_ids': text_features['input_ids'].to(device),
                'attention_mask': text_features['attention_mask'].to(device)
            }

        box, cls, mask = diffusion_model.conditional_reverse_ddim(layout, image, sal_box, cfg, cond=cond, intent_box=intent_box, text_features=text_features)
        samples = torch.cat([cls, box], dim=2)
        sample_output.append(samples.cpu())

        cnt = cnt + image.shape[0]
        logger.log(f"created {cnt} samples")

    sample_output = torch.concat(sample_output, dim=0)
    return sample_output

def sample_refine(diffusion_model, testing_dl, cfg):
    samples = {'output': [], 'noise': [], 'gt': []}
    num_class = cfg.num_class
    cnt = 0
    device = diffusion_model.device

    for idx, (image, layout, sal_box) in enumerate(testing_dl):
        image, layout, sal_box = image.to(device), layout.to(device), sal_box.to(device)
        real_label = layout[:,:,:num_class]
        box_gt, cls_gt, mask_gt = finalize(layout, num_class)
        cls_gt[:,1:,:] = 0

        noise = torch.normal(0, 0.01, size=box_gt.size()).to(device)
        box_noise = torch.clamp(box_gt + noise, min=0, max=1)
        noise_layout = torch.cat((real_label, 2 * (box_noise - 0.5)), dim=2).to(device)

        box, cls, _ = diffusion_model.refinement_reverse_ddim(noise_layout, image, sal_box)

        for key, value in zip(samples.keys(), [
            torch.cat([cls, box], dim=2),
            torch.cat([cls_gt, box_noise], dim=2),
            torch.cat([cls_gt, box_gt], dim=2)
        ]):
            samples[key].append(value.cpu())

        cnt = cnt + image.shape[0]
        logger.log(f"created {cnt} samples")

    return [torch.cat(samples[key], dim=0) for key in samples.keys()]

def main(opt):
    seed = opt.seed
    set_seed(seed)

    device = torch.device(f"cuda:{opt.gpuid}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpuid)

    cfg = load_config(
        f'configs/{opt.dataset}_{opt.anno}_test.yaml',
        path_profile=getattr(opt, 'path_profile', None),
    )
    cfg.task = opt.task
    cfg.imgname_order_dir = os.path.join(cfg.imgname_order_dir, f'seed_{seed}_{opt.dataset}_{opt.anno}_test.pt')
    cfg.v_encoder = opt.v_encoder
    cfg.spatial_guidance = opt.spatial_guidance
    cfg.text_control = opt.text_control
    cfg.spatial_metrics = opt.spatial_metrics
    cfg.protocol = opt.protocol
    cfg.disable_spatial_boxes = opt.disable_spatial_boxes
    cfg.disable_text_spatial_parser = opt.disable_text_spatial_parser
    cfg.oracle_intent_map = opt.oracle_intent_map
    if opt.oracle_intent_map and opt.anno != 'anno':
        raise ValueError("--oracle-intent-map requires --anno anno")
    if opt.protocol == 'image_only' and opt.text_control:
        raise ValueError("image_only protocol cannot use --text-control")
    if opt.protocol in {'oracle_prompt', 'freeform_prompt', 'text_baseline'} and not opt.text_control:
        raise ValueError(f"{opt.protocol} protocol requires --text-control")
    if opt.prompts_csv:
        prompt_path = rebase_project_path(
            opt.prompts_csv, getattr(opt, 'path_profile', None)
        )
        if not os.path.isfile(prompt_path):
            raise FileNotFoundError(f"Prompt CSV not found: {prompt_path}")
        cfg.paths.test.all_prompts = prompt_path
        print(f"Prompt override: {prompt_path}")
        if opt.prompt_subset_only:
            with open(prompt_path, newline='') as prompt_file:
                rows = list(csv.DictReader(prompt_file))
            cfg.test_image_names = {
                os.path.basename(row['poster_path'])
                for row in rows
                if row.get('poster_path')
            }
            if not cfg.test_image_names:
                raise ValueError(
                    "--prompt-subset-only requires a non-empty poster_path column"
                )
            print(f"Prompt subset: {len(cfg.test_image_names)} images")

    print(f"Using {cfg.v_encoder} encoder")
    print(f"Using spatial guidance: {cfg.spatial_guidance}")
    print(f"Text control enabled: {cfg.text_control}")
    print(f"Path profile: {cfg.path_profile} ({cfg.project_root})")

    if cfg.task == 'uncond':
        testing_set = test_uncond_dataset(cfg)
    else:
        testing_set = test_cond_dataset(cfg)
    testing_dl = DataLoader(testing_set, num_workers=cfg.num_workers, batch_size=cfg.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    logger.log(f"Testing set size: {len(testing_set)}")

    # test_output_pt_dir = ""
    # test_output = torch.load(test_output_pt_dir)

    ddim_steps = getattr(opt, 'ddim_num_steps', 100)
    ddim_schedule = getattr(opt, 'ddim_schedule', 'cosine')
    model_cfg = cfg
    if opt.model_dataset and opt.model_dataset != opt.dataset:
        model_cfg = load_config(
            f'configs/{opt.model_dataset}_{opt.anno}_test.yaml',
            path_profile=getattr(opt, 'path_profile', None),
        )
        logger.log(
            f"Cross-dataset model: {opt.model_dataset} checkpoint evaluated on {opt.dataset}"
        )

    diffusion_model = Diffusion(num_timesteps=1000,
                                ddim_num_steps=ddim_steps,
                                n_head=model_cfg.n_head,
                                dim_model=model_cfg.d_model,
                                feature_dim=model_cfg.feature_dim,
                                seq_dim=model_cfg.num_class + 4,
                                num_layers=model_cfg.n_layers,
                                device=device,
                                max_elem=model_cfg.max_elem,
                                v_encoder=cfg.v_encoder,
                                spatial_guidance=cfg.spatial_guidance,
                                text_control=getattr(cfg, 'text_control', False),
                                text_conditioning_mode=opt.text_conditioning_mode,
                                text_guidance_scale=opt.text_guidance_scale,
                                ddim_schedule=ddim_schedule)
    checkpoint_path = rebase_project_path(
        opt.check_path, getattr(opt, 'path_profile', None)
    )
    model_weights = torch.load(checkpoint_path, map_location=device)
    diffusion_model.model.load_state_dict(model_weights)
    diffusion_model.model.eval()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    sample_start = time.perf_counter()

    if cfg.task == 'uncond':
        test_output = sample_uncond(diffusion_model, testing_dl, cfg)
    elif cfg.task == 'refine':
        test_output, test_output_noise, test_output_gt = sample_refine(diffusion_model, testing_dl, cfg)
    else:
        test_output = sample_cond(diffusion_model, testing_dl, cfg, cond=cfg.task)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    sample_seconds = time.perf_counter() - sample_start

    img_names = torch.load(cfg.imgname_order_dir)
    img_names = img_names[:test_output.shape[0]]
    # occ_matrix = torch.load("")
    # rea_matrix = torch.load("")

    ground_truth = None
    if opt.anno == 'anno':
        ground_truth = load_ground_truth_layouts(img_names, cfg)
    metrics, per_image_records = metric(
        img_names,
        test_output,
        cfg,
        ground_truth=ground_truth,
        return_records=True,
    )

    if opt.save_test_output:
        if opt.save_test_output == 'auto':
            if not opt.experiment_name:
                raise ValueError("--save-test-output auto requires --experiment_name")
            output_path = os.path.join(
                cfg.project_root,
                'experiments',
                'paper_figures',
                f'{opt.experiment_name}_test_output.pt',
            )
        else:
            output_path = rebase_project_path(
                opt.save_test_output, getattr(opt, 'path_profile', None)
            )
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.save(
            {
                'img_names': img_names,
                'test_output': test_output.cpu(),
                'protocol': opt.protocol,
                'dataset': opt.dataset,
                'model_dataset': opt.model_dataset or opt.dataset,
                'training_checkpoint': checkpoint_path,
                'inference_seed': seed,
                'ddim_steps': ddim_steps,
                'ddim_schedule': ddim_schedule,
            },
            output_path,
        )
        logger.log(f"Saved test tensors to {output_path}")

    # visualize: save to experiment-specific folder for paper figures when --experiment_name is set
    if opt.experiment_name:
        project_root = cfg.project_root
        paper_figures_dir = os.path.join(project_root, 'experiments', 'paper_figures')
        os.makedirs(paper_figures_dir, exist_ok=True)
        metrics_path = os.path.join(paper_figures_dir, f'{opt.experiment_name}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
        per_image_path = os.path.join(
            paper_figures_dir, f'{opt.experiment_name}_per_image.csv'
        )
        pd.DataFrame(per_image_records).to_csv(per_image_path, index=False)
        total_params = sum(parameter.numel() for parameter in diffusion_model.model.parameters())
        trainable_params = sum(
            parameter.numel()
            for parameter in diffusion_model.model.parameters()
            if parameter.requires_grad
        )
        evidence = {
            'experiment': opt.experiment_name,
            'protocol': opt.protocol,
            'dataset': opt.dataset,
            'model_dataset': opt.model_dataset or opt.dataset,
            'annotation_split': opt.anno,
            'checkpoint': checkpoint_path,
            'inference_seed': seed,
            'ddim_steps': ddim_steps,
            'ddim_schedule': ddim_schedule,
            'num_samples': int(test_output.shape[0]),
            'sampling_seconds': sample_seconds,
            'milliseconds_per_sample': 1000.0 * sample_seconds / max(1, test_output.shape[0]),
            'samples_per_second': test_output.shape[0] / max(sample_seconds, 1e-12),
            'peak_cuda_memory_bytes': int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else 0,
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'text_conditioning_mode': opt.text_conditioning_mode,
            'text_guidance_scale': opt.text_guidance_scale,
            'disable_text_spatial_parser': bool(opt.disable_text_spatial_parser),
            'oracle_intent_map': bool(opt.oracle_intent_map),
            'metrics_file': metrics_path,
            'per_image_file': per_image_path,
        }
        with open(
            os.path.join(paper_figures_dir, f'{opt.experiment_name}_evidence.json'),
            'w',
        ) as f:
            json.dump(evidence, f, indent=2)
        logger.log(f"Saved metrics to {metrics_path}")
        cfg.save_imgs_dir = os.path.join(paper_figures_dir, opt.experiment_name)
    else:
        cfg.save_imgs_dir = os.path.join(cfg.save_imgs_dir, f'{opt.dataset}_{opt.anno}_test')
    if not opt.no_render:
        visualize_images(img_names, test_output, cfg)

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
        '--dataset',
        type=str,
        default='pku',
        help='choose dataset to test (pku, cgl)')
    parser.add_argument(
        '--anno',
        type=str,
        default='unanno',
        help='choose dataset to test (anno, unanno)')
    parser.add_argument(
        '--task',
        type=str,
        default='uncond',
        help='choose task to test (uncond, c, cwh, complete, refinement)'
    )
    parser.add_argument(
        '--check_path',
        type=str,
        default='',
        help='choose checkpoint'
    )
    parser.add_argument(
        '--v_encoder',
        type=str,
        default='vit',
        help='choose visual encoder(vit, swin, convnext)')
    parser.add_argument(
        '--spatial_guidance',
        type=int,
        default=0,
        help='Spatial guidance: 0=saliency, 1=intent map+boxes, 2=both maps+boxes, 3=saliency map+intent boxes')
    parser.add_argument(
        '--text_control',
        action='store_true',
        help='Enable text control for layout generation')
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='',
        help='Experiment name for saving inference images to experiments/paper_figures/<name>/ (for paper figures)')
    parser.add_argument(
        '--ddim_num_steps',
        type=int,
        default=100,
        help='Number of DDIM sampling steps (use same value for fair comparison; default 100)')
    parser.add_argument(
        '--ddim-schedule', '--ddim_schedule',
        dest='ddim_schedule',
        choices=('training', 'cosine', 'linear'),
        default='cosine',
        help="Cumulative alpha schedule used by DDIM sampling. Default 'cosine' matches the trained DDPM schedule; 'linear' is retained only for legacy/sensitivity checks.")
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed for reproducibility (default 1)')
    parser.add_argument(
        '--save-test-output', '--save_test_output',
        dest='save_test_output',
        default='',
        help="Save {'img_names', 'test_output'} tensors; use 'auto' with --experiment_name or provide a .pt path")
    parser.add_argument(
        '--prompts-csv', '--prompts_csv',
        dest='prompts_csv',
        default='',
        help='Override the test prompt CSV for robustness/free-form evaluation')
    parser.add_argument(
        '--spatial-metrics', '--spatial_metrics',
        dest='spatial_metrics',
        action='store_true',
        help='Compute 3x3-grid spatial prompt adherence from the active prompt CSV')
    parser.add_argument(
        '--prompt-subset-only', '--prompt_subset_only',
        dest='prompt_subset_only',
        action='store_true',
        help='Evaluate only images named in the prompt CSV poster_path column')
    parser.add_argument(
        '--path-profile', '--path_profile',
        dest='path_profile',
        choices=('local', 'server'),
        default='local',
        help='Path profile: local=current checkout; server=/home/viplab/Aagha/intent_aware_layout_generation')
    parser.add_argument(
        '--protocol',
        choices=('image_only', 'oracle_prompt', 'freeform_prompt', 'text_baseline', 'cross_dataset'),
        default='image_only',
        help='Evaluation protocol recorded with every result')
    parser.add_argument(
        '--model-dataset', '--model_dataset',
        dest='model_dataset',
        choices=('pku', 'cgl'),
        default='',
        help='Dataset architecture used by the checkpoint; enables cross-dataset evaluation')
    parser.add_argument(
        '--text-conditioning-mode', '--text_conditioning_mode',
        dest='text_conditioning_mode',
        choices=('token', 'pooled'),
        default='token',
        help='Token-level cross-attention (main) or pooled-text ablation')
    parser.add_argument(
        '--text-guidance-scale', '--text_guidance_scale',
        dest='text_guidance_scale', type=float, default=1.0,
        help='Inference-only scale applied to projected text tokens')
    parser.add_argument(
        '--no-render', '--no_render',
        dest='no_render',
        action='store_true',
        help='Skip PNG rendering for large metric sweeps')
    parser.add_argument(
        '--disable-spatial-boxes', '--disable_spatial_boxes',
        dest='disable_spatial_boxes',
        action='store_true',
        help='Ablation: retain the spatial pixel map but omit intent-box tokens')
    parser.add_argument(
        '--disable-text-spatial-parser', '--disable_text_spatial_parser',
        dest='disable_text_spatial_parser',
        action='store_true',
        help='Diagnostic: keep BERT text conditioning but disable parsed text-spatial boxes')
    parser.add_argument(
        '--oracle-intent-map', '--oracle_intent_map',
        dest='oracle_intent_map', action='store_true',
        help='Diagnostic upper bound: rasterize GT layout density as the intent map')

    opt = parser.parse_args()
    main(opt)


# for idx in range(model_output_gt.shape[0]):
#     image_path = os.path.join(test_inp_dir, names[idx])
#     img = Image.open(image_path).convert("RGB")
#     res = model_output_gt[idx]
#     cls = res[:,:1]
#     box = res[:,1:]
#     draw_single(box, cls, img, idx, save_dir_1, width, height, num_class)
#
#     print(idx)
