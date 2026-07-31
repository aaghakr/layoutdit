import os
import os.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import argparse
from transformers import set_seed

import torch
from torch.utils.data import DataLoader
from data_process.dataloader import train_dataset, test_uncond_dataset, test_cond_dataset, custom_collate_fn
from scripts.train_util import TrainLoop
from cgbdm.diffusion import Diffusion
from utils import logger
from utils.util import get_parameter_number, load_config

CUDA_LAUNCH_BLOCKING=1

def main(opt):
    seed = opt.seed
    set_seed(seed)

    device = torch.device(f"cuda:{opt.gpuid}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpuid)

    # config_module = importlib.import_module(f'configs/config_{opt.dataset}')
    # cfg = config_module.config

    if getattr(opt, 'config', None):
        cfg = load_config(opt.config, path_profile=getattr(opt, 'path_profile', None))
    else:
        cfg = load_config(
            f'configs/{opt.dataset}.yaml',
            path_profile=getattr(opt, 'path_profile', None),
        )
    cfg.experiment_name = opt.experiment_name
    cfg.task = opt.task
    cfg.v_encoder = opt.v_encoder
    cfg.spatial_guidance = opt.spatial_guidance
    cfg.text_control = opt.text_control
    cfg.text_conditioning_mode = opt.text_conditioning_mode
    cfg.disable_spatial_boxes = opt.disable_spatial_boxes
    cfg.disable_text_spatial_parser = opt.disable_text_spatial_parser
    cfg.precision = opt.precision
    cfg.allow_tf32 = opt.allow_tf32
    cfg.skip_training_validation = opt.skip_training_validation
    if opt.epochs is not None:
        cfg.epochs = opt.epochs
    if opt.train_batch_size is not None:
        cfg.train_batch_size = opt.train_batch_size
    if opt.test_batch_size is not None:
        cfg.test_batch_size = opt.test_batch_size
    if opt.num_workers is not None:
        cfg.num_workers = opt.num_workers
    if opt.lambda1 is not None:
        cfg.lambda1 = opt.lambda1
    if opt.lambda2 is not None:
        cfg.lambda2 = opt.lambda2
    cfg.deterministic = not opt.allow_nondeterministic
    if cfg.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
        torch.backends.cudnn.allow_tf32 = cfg.allow_tf32
    cfg.seed = seed
    print(f"Using {cfg.v_encoder} encoder")
    print(f"Using spatial guidance: {cfg.spatial_guidance}")
    print(f"Text control enabled: {cfg.text_control}")
    print(f"Path profile: {cfg.path_profile} ({cfg.project_root})")


    training_set = train_dataset(cfg)
    data_generator = torch.Generator()
    data_generator.manual_seed(seed)
    training_dl = DataLoader(
        training_set,
        num_workers=cfg.num_workers,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        generator=data_generator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    if cfg.task == 'uncond':
        cfg.imgname_order_dir = os.path.join(
            cfg.imgname_order_dir,
            f'{opt.experiment_name}_seed_{seed}_{opt.dataset}_unanno_test.pt',
        )
        evaling_set = test_uncond_dataset(cfg)
    else:
        cfg.imgname_order_dir = os.path.join(
            cfg.imgname_order_dir,
            f'{opt.experiment_name}_seed_{seed}_{opt.dataset}_anno_test.pt',
        )
        evaling_set = test_cond_dataset(cfg)
    evaling_dl = DataLoader(
        evaling_set,
        num_workers=cfg.num_workers,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )

    logger.info(f"Training set size: {len(training_set)}, Evaling set size:{len(evaling_set)}")

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
                                text_conditioning_mode=opt.text_conditioning_mode)
    total_num, trainable_num = get_parameter_number(diffusion_model.model)
    logger.info(f"trainable_num/total_num: %.2fM/%.2fM" % (trainable_num / 1e6, total_num / 1e6))

    # weights = torch.load('')
    # model_ddpm.model.load_state_dict(weights)
    TrainLoop(
        cfg,
        diffusion_model=diffusion_model,
        training_dl=training_dl,
        testing_dl=None,
        evaling_dl=evaling_dl,
        device=device,
    ).run_loop()

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
        help='choose dataset to train')
    parser.add_argument(
        '--task',
        type=str,
        default='uncond',
        help='choose task to train(uncond,c,cwh,complete)'
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
        default='default_experiment',
        help='name of the experiment'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Independent training seed (default: 1)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='',
        help='Optional YAML config under code/ (e.g. configs/pku_lambda_high_place.yaml). If set, overrides configs/{dataset}.yaml.'
    )
    parser.add_argument(
        '--path-profile', '--path_profile',
        dest='path_profile',
        choices=('local', 'server'),
        default='local',
        help='Path profile: local=current checkout; server=/home/viplab/Aagha/intent_aware_layout_generation'
    )
    parser.add_argument(
        '--text-conditioning-mode', '--text_conditioning_mode',
        dest='text_conditioning_mode',
        choices=('token', 'pooled'),
        default='token',
        help='Token-level cross-attention (main) or pooled-text ablation')
    parser.add_argument(
        '--disable-spatial-boxes', '--disable_spatial_boxes',
        dest='disable_spatial_boxes',
        action='store_true',
        help='Ablation: retain spatial pixel maps but omit intent-box tokens')
    parser.add_argument(
        '--disable-text-spatial-parser', '--disable_text_spatial_parser',
        dest='disable_text_spatial_parser',
        action='store_true',
        help='Ablation: train/evaluate with BERT text conditioning but without parsed text-spatial boxes')
    parser.add_argument('--lambda1', type=float, default=None, help='Override prompt-count loss weight')
    parser.add_argument('--lambda2', type=float, default=None, help='Override placement loss weight')
    parser.add_argument(
        '--precision', choices=('fp32', 'bf16'), default='fp32',
        help='Training compute precision; bf16 is recommended on A100 GPUs')
    parser.add_argument(
        '--allow-tf32', action='store_true',
        help='Allow deterministic TensorFloat-32 kernels for remaining FP32 matrix operations')
    parser.add_argument(
        '--skip-training-validation', action='store_true',
        help='Skip costly DDIM validation during training; the final checkpoint is still saved')
    parser.add_argument('--epochs', type=int, default=None, help='Override configured epoch count')
    parser.add_argument('--train-batch-size', type=int, default=None, help='Override training batch size')
    parser.add_argument('--test-batch-size', type=int, default=None, help='Override validation batch size')
    parser.add_argument('--num-workers', type=int, default=None, help='Override DataLoader worker count')
    parser.add_argument(
        '--allow-nondeterministic', action='store_true',
        help='Allow nondeterministic cuDNN kernels (deterministic is the paper default)')
    opt = parser.parse_args()
    main(opt)
