import os
import sys
import json
import platform
from pathlib import Path
import torch
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:  # minimal no-op fallback; TensorBoard is optional
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass
from tqdm import tqdm
from utils import logger
import torch.nn as nn
from utils.metric import metric
from utils.aux_losses import loss_count, loss_place
from scripts.test import sample_uncond, sample_cond
from cgbdm.text_spatial import create_text_spatial_boxes_batch


class TrainLoop:
    def __init__(
            self,
            cfg,
            diffusion_model,
            training_dl,
            testing_dl,
            evaling_dl,
            device,
    ):
        self.datetime=cfg.datetime
        self.diffusion_model = diffusion_model

        self.train_data = training_dl
        self.val_data = evaling_dl
        self.test_data = testing_dl
        self.cfg = cfg

        self.initial_lr = cfg.lr
        self.gradient_clipping = cfg.gradient_clipping
        self.epochs = cfg.epochs
        self.num_class = cfg.num_class
        self.device = device
        self.precision = getattr(cfg, 'precision', 'fp32')
        self.autocast_enabled = self.device.type == 'cuda' and self.precision == 'bf16'

        self.master_params = list(self.diffusion_model.model.parameters())

        self.opt = optim.Adam(self.master_params, lr=self.initial_lr, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.epochs)

        log_dir = f"runs/{self.cfg.dataset_cls}/{self.cfg.experiment_name}/{self.datetime}"
        self.writer = SummaryWriter(log_dir=log_dir)

    @staticmethod
    def _config_dict(value):
        if hasattr(value, '__dict__'):
            return {
                key: TrainLoop._config_dict(item)
                for key, item in vars(value).items()
            }
        if isinstance(value, (list, tuple)):
            return [TrainLoop._config_dict(item) for item in value]
        return value

    def requires_grad(self, model, flag=False):
        """
        Set requires_grad flag for all parameters in a model.
        """
        for p in model.parameters():
            p.requires_grad = flag

    def get_description(self, epoch, epochs, lr, loss):
        return (f'Epoch {epoch} / Epochs {epochs}, '
                f'LR: {lr:.2e}, '
                f'Loss: {loss:.4f}')

    def optimize_normal(self):
        if self.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(self.diffusion_model.model.parameters(), self.gradient_clipping)
        self.opt.step()

    def log_metrics(self, metric_res, epoch):
        metrics = {
            'Val': 'val',
            'Ove': 'ove',
            'Und_l': 'undl',
            'Und_s': 'unds',
            'Rea': 'rea',
            'Occ': 'occ'
        }
        for display_name, metric_key in metrics.items():
            self.writer.add_scalar(display_name, metric_res[metric_key], epoch)

    def test_uncond(self):
        test_output = sample_uncond(self.diffusion_model, self.val_data, self.cfg)
        img_names = torch.load(self.cfg.imgname_order_dir)

        # load matrix infomation
        # occ_matrix = torch.load("")
        # rea_matrix = torch.load("")

        metrics = metric(img_names, test_output, self.cfg)

        # store sample output
        # base_test_output_dir = Path('')
        # test_output_dir = base_test_output_dir / self.datetime
        # test_output_dir.mkdir(parents=True, exist_ok=True)
        # test_output_dir = test_output_dir + 'test_output.pt'
        # torch.save(test_output, test_output_dir)
        return metrics

    def test_constraint(self,):
        cond = self.cfg.task
        # occ_matrix = torch.load("")
        # rea_matrix = torch.load("")

        test_output = sample_cond(self.diffusion_model, self.val_data, self.cfg, cond=cond)
        img_names = torch.load(self.cfg.imgname_order_dir)
        metrics = metric(img_names, test_output, self.cfg)

        # store sample output
        # output_dir = Path('') / self.cfg.task / self.datetime
        # output_dir.mkdir(parents=True, exist_ok=True)
        # output_path = output_dir / 'test_output.pt'
        # torch.save(test_output, output_path)
        return metrics


    def run_loop(self):
        logger.info(f"Training for {self.epochs} epochs...")
        base_check_dir = Path(self.cfg.base_check_dir)
        experiment_name =  self.cfg.experiment_name
        check_dir = base_check_dir / experiment_name / self.datetime
        check_dir.mkdir(parents=True, exist_ok=True)
        provenance = {
            'config': self._config_dict(self.cfg),
            'python': platform.python_version(),
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'device': str(self.device),
            'gpu': torch.cuda.get_device_name(self.device) if torch.cuda.is_available() else None,
            'total_parameters': sum(p.numel() for p in self.diffusion_model.model.parameters()),
            'trainable_parameters': sum(
                p.numel() for p in self.diffusion_model.model.parameters() if p.requires_grad
            ),
            'precision': self.precision,
            'autocast_enabled': self.autocast_enabled,
            'allow_tf32': getattr(self.cfg, 'allow_tf32', False),
            'training_validation_skipped': getattr(
                self.cfg, 'skip_training_validation', False
            ),
        }
        (check_dir / 'run_config.json').write_text(json.dumps(provenance, indent=2) + '\n')

        for epoch in range(self.epochs):
            epoch += 1
            self.run_train_step(self.train_data, epoch)
            logger.info("train finish!")
            # Modify log_test_epochs, observe the validation set results on tensorboard, and select the optimal weight
            should_validate = (
                not getattr(self.cfg, 'skip_training_validation', False)
                and epoch >= 400
                and epoch % self.cfg.log_test_epochs == 0
            )
            if should_validate:
                if self.cfg.task == 'uncond':
                    metrics = self.test_uncond()
                else:
                    metrics = self.test_constraint()
                self.log_metrics(metrics, epoch)
                logger.log(f"Sample {self.cfg.task} {epoch} epoch done!")

            if should_validate or epoch == self.epochs:
                file_name = f'Epoch{epoch}_cgbdm_weights.pth'
                check_epoch_dir = os.path.join(check_dir, file_name)
                torch.save(self.diffusion_model.model.state_dict(), check_epoch_dir)

            self.scheduler.step()
        logger.info("Done!")
        # torch.save(self.diffusion_model.model.state_dict(), check_dir)
        self.writer.close()

    def run_train_step(self, data, epoch):
        self.diffusion_model.model.train()
        steps = 0
        total_loss = 0.0
        mse_loss = nn.MSELoss()
        pbar = tqdm(data, desc=f'Epoch {epoch}')

        for idx, data in enumerate(pbar):
            self.opt.zero_grad(set_to_none=True)

            # Handle different return values based on spatial guidance and text control
            text_features = None
            prompt_texts = None
            if len(data) == 3:
                image, layout, sal_box = data
                intent_box = None
            elif len(data) == 4:
                if isinstance(data[3], dict):
                    image, layout, sal_box, text_features = data
                    intent_box = None
                else:
                    image, layout, sal_box, intent_box = data
            elif len(data) == 5:
                if isinstance(data[4], list):
                    image, layout, sal_box, text_features, prompt_texts = data
                    intent_box = None
                else:
                    image, layout, sal_box, intent_box, text_features = data
            elif len(data) == 6:
                image, layout, sal_box, intent_box, text_features, prompt_texts = data
            else:
                image, layout, sal_box = data
                intent_box = None

            non_blocking = self.device.type == 'cuda'
            image = image.to(self.device, non_blocking=non_blocking)
            layout = layout.to(self.device, non_blocking=non_blocking)
            sal_box = sal_box.to(self.device, non_blocking=non_blocking)
            if intent_box is not None:
                intent_box = intent_box.to(self.device, non_blocking=non_blocking)
                if getattr(self.cfg, 'disable_spatial_boxes', False):
                    intent_box = None
            if text_features is not None:
                text_features = {
                    'input_ids': text_features['input_ids'].to(
                        self.device, non_blocking=non_blocking
                    ),
                    'attention_mask': text_features['attention_mask'].to(
                        self.device, non_blocking=non_blocking
                    )
                }

            # Build text-spatial guidance from prompts that contain positions
            text_spatial_boxes = None
            text_spatial_mask = None
            use_text_spatial = None
            if (
                prompt_texts
                and getattr(self.cfg, 'text_control', False)
                and not getattr(self.cfg, 'disable_text_spatial_parser', False)
            ):
                gt_classes = torch.argmax(layout[:, :, :self.num_class], dim=2)
                text_spatial_boxes, text_spatial_mask, use_text_spatial = \
                    create_text_spatial_boxes_batch(
                        prompt_texts,
                        getattr(self.cfg, 'max_elem', 16),
                        self.num_class,
                        gt_classes_batch=gt_classes,
                        device=self.device,
                    )

            t = self.diffusion_model.sample_t([layout.shape[0]], t_max=self.diffusion_model.num_timesteps - 1)
            lambda1 = getattr(self.cfg, 'lambda1', 0.0)
            lambda2 = getattr(self.cfg, 'lambda2', 0.0)
            need_reparam = (lambda1 and prompt_texts) or (lambda2 and intent_box is not None)

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.autocast_enabled,
            ):
                if need_reparam:
                    eps_theta, e, l_0_pred = self.diffusion_model.forward_t(
                        layout, image, sal_box, t=t, cond=self.cfg.task, reparam=True,
                        intent_box=intent_box, text_features=text_features,
                        text_spatial_boxes=text_spatial_boxes,
                        text_spatial_mask=text_spatial_mask,
                        use_text_spatial=use_text_spatial,
                    )
                else:
                    eps_theta, e = self.diffusion_model.forward_t(
                        layout, image, sal_box, t=t, cond=self.cfg.task,
                        intent_box=intent_box, text_features=text_features,
                        text_spatial_boxes=text_spatial_boxes,
                        text_spatial_mask=text_spatial_mask,
                        use_text_spatial=use_text_spatial,
                    )
                    l_0_pred = None

                loss_denoise = mse_loss(e, eps_theta)

            # Keep metric-like geometry/count reductions in FP32 while the
            # expensive neural forward pass uses A100-native BF16.
            loss = loss_denoise
            l_count = l_place = None
            if lambda1 and prompt_texts and l_0_pred is not None:
                l_count = loss_count(
                    l_0_pred.float(), prompt_texts, self.num_class, self.device
                )
                loss = loss + lambda1 * l_count
            if lambda2 and intent_box is not None and l_0_pred is not None:
                l_place = loss_place(
                    l_0_pred.float(), intent_box.float(), self.num_class, self.device
                )
                loss = loss + lambda2 * l_place
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={epoch}, batch={idx}; "
                    f"denoise={float(loss_denoise.detach())}, "
                    f"count={float(l_count.detach()) if l_count is not None else None}, "
                    f"place={float(l_place.detach()) if l_place is not None else None}"
                )
            loss_value = float(loss.detach())
            total_loss += loss_value
            steps += 1

            description = self.get_description(epoch, self.epochs, self.opt.param_groups[0]["lr"], total_loss / steps)
            pbar.set_description(description)
            loss.backward()
            self.optimize_normal()

        logger.log(description)
        self.writer.add_scalar('Loss/train', total_loss / steps, epoch)
