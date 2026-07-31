import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pandas import read_csv
from PIL import Image, ImageDraw, ImageFilter
from utils.util import box_xyxy_to_cxcywh, natural_sort_cmp
from transformers import BertTokenizer
import pandas as pd


def save_image_order(image_names, output_path):
    """Persist evaluation ordering after creating its runtime directory."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(image_names, output)

def custom_collate_fn(batch):
    """
    Custom collate function to handle mixed return types (tensors + dictionaries)
    """
    # Separate tensors, dictionaries, and optional prompt texts
    tensors = []
    text_features_list = []
    prompt_texts_list = []

    for item in batch:
        if isinstance(item[-1], str) and len(item) >= 2 and isinstance(item[-2], dict):
            prompt_texts_list.append(item[-1])
            text_features_list.append(item[-2])
            tensors.append(item[:-2])
        elif isinstance(item[-1], dict):
            text_features_list.append(item[-1])
            tensors.append(item[:-1])
            prompt_texts_list.append(None)
        else:
            text_features_list.append(None)
            prompt_texts_list.append(None)
            tensors.append(item)

    # Collate tensors - handle variable sizes by padding
    collated_tensors = []
    for i in range(len(tensors[0])):
        tensor_list = [tensor[i] for tensor in tensors]

        # Check if all tensors have the same shape
        if all(t.shape == tensor_list[0].shape for t in tensor_list):
            collated_tensors.append(torch.stack(tensor_list))
        else:
            # Handle variable sizes by padding to the maximum size
            max_size = max(t.shape[0] for t in tensor_list)
            padded_tensors = []
            for t in tensor_list:
                if t.shape[0] < max_size:
                    # Pad with zeros
                    padding_size = max_size - t.shape[0]
                    if len(t.shape) == 1:
                        padding = torch.zeros(padding_size, dtype=t.dtype, device=t.device)
                    else:
                        padding = torch.zeros((padding_size,) + t.shape[1:], dtype=t.dtype, device=t.device)
                    padded_t = torch.cat([t, padding], dim=0)
                else:
                    padded_t = t
                padded_tensors.append(padded_t)
            collated_tensors.append(torch.stack(padded_tensors))

    # Handle text_features
    if any(tf is not None for tf in text_features_list):
        # Find a non-None text_features to get the structure
        non_none_tf = next(tf for tf in text_features_list if tf is not None)
        collated_text_features = {}
        for key in non_none_tf.keys():
            collated_text_features[key] = torch.stack([
                tf[key] if tf is not None else torch.zeros_like(non_none_tf[key])
                for tf in text_features_list
            ])
        if any(p is not None for p in prompt_texts_list):
            return tuple(collated_tensors) + (collated_text_features, prompt_texts_list)
        return tuple(collated_tensors) + (collated_text_features,)
    else:
        return tuple(collated_tensors)

class train_dataset(Dataset):
    def __init__(self, cfg):
        img = sorted(os.listdir(cfg.paths.train.inp_dir))
        # torch.save(img, "imgname_train_save.pt")
        self.inp = list(map(lambda x: os.path.join(cfg.paths.train.inp_dir, x), img))

        # Handle spatial guidance modes
        if cfg.spatial_guidance == 0:  # Saliency only
            self.sal = list(map(lambda x: os.path.join(cfg.paths.train.sal_dir, x), img))
            self.sal_sub = list(map(lambda x: os.path.join(cfg.paths.train.sal_sub_dir, x), img))
            self.box_path = cfg.paths.train.salbox_dir
            self.use_saliency = True
            self.use_intent = False

        elif cfg.spatial_guidance == 1:  # Intent only - NO SALIENCY
            self.intent = list(map(lambda x: os.path.join(cfg.paths.train.intent_map_dir, x), img))
            self.box_path = cfg.paths.train.intentbox_dir  # Use intent boxes
            self.use_saliency = False
            self.use_intent = True

        elif cfg.spatial_guidance == 2:  # Both saliency and intent
            self.sal = list(map(lambda x: os.path.join(cfg.paths.train.sal_dir, x), img))
            self.sal_sub = list(map(lambda x: os.path.join(cfg.paths.train.sal_sub_dir, x), img))
            self.intent = list(map(lambda x: os.path.join(cfg.paths.train.intent_map_dir, x), img))
            self.box_path = cfg.paths.train.salbox_dir  # Use salbox as primary
            self.use_saliency = True
            self.use_intent = True
            self.box_only = False
        elif cfg.spatial_guidance == 3:  # Saliency pixels + intent boxes (no intent pixels)
            self.sal = list(map(lambda x: os.path.join(cfg.paths.train.sal_dir, x), img))
            self.sal_sub = list(map(lambda x: os.path.join(cfg.paths.train.sal_sub_dir, x), img))
            self.box_path = cfg.paths.train.salbox_dir
            self.use_saliency = True
            self.use_intent = False
            self.box_only = True
        else:
            raise ValueError(f"Invalid spatial guidance: {cfg.spatial_guidance}")

        self.num_class = cfg.num_class
        self.max_elem = cfg.max_elem

        df_anno = read_csv(cfg.paths.train.annotated_dir)
        df_box = read_csv(self.box_path)  # This will be salbox or intentbox based on mode

        self.poster_name = list(img)
        self.groups_annotated = df_anno.groupby(df_anno.poster_path)
        self.groups_box = df_box.groupby(df_box.poster_path)

        # Load intent boxes when needed
        if cfg.spatial_guidance in (1, 2, 3):
            df_intent_box = read_csv(cfg.paths.train.intentbox_dir)
            self.groups_intent_box = df_intent_box.groupby(df_intent_box.poster_path)
        else:
            self.groups_intent_box = None

        # Text control support
        self.text_control = getattr(cfg, 'text_control', False)
        if self.text_control:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.prompts_df = pd.read_csv(cfg.paths.train.all_prompts)
            self.groups_prompts = self.prompts_df.groupby(self.prompts_df.poster_path)

        self.transform_rgb = transforms.Compose([
            transforms.Resize([384,256]), # (360,240)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.transform_l = transforms.Compose([
            transforms.Resize([384,256]),  # (360,240)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
        ])

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        img_inp = Image.open(self.inp[idx]).convert("RGB")
        width, height = img_inp.size

        # Transform RGB image first
        img_inp = self.transform_rgb(img_inp)

        # Handle different spatial guidance modes
        if self.use_saliency and not self.use_intent:  # Mode 0: Saliency only
            img_sal = Image.open(self.sal[idx]).convert("L")
            img_sal_sub = Image.open(self.sal_sub[idx]).convert("L")
            img_sal_map = Image.fromarray(np.maximum(np.array(img_sal), np.array(img_sal_sub)))
            img_sal_map = self.transform_l(img_sal_map)
            imgs = torch.concat([img_inp, img_sal_map])

        elif self.use_intent and not self.use_saliency:  # Mode 1: Intent only - NO SALIENCY AT ALL
            img_intent = Image.open(self.intent[idx]).convert("L")
            img_intent = self.transform_l(img_intent)
            imgs = torch.concat([img_inp, img_intent])  # Only RGB + Intent

        elif self.use_saliency and self.use_intent:  # Mode 2: Both saliency and intent
            # Load both maps separately
            img_sal = Image.open(self.sal[idx]).convert("L")
            img_sal_sub = Image.open(self.sal_sub[idx]).convert("L")
            img_sal_map = Image.fromarray(np.maximum(np.array(img_sal), np.array(img_sal_sub)))

            img_intent = Image.open(self.intent[idx]).convert("L")

            # Transform both maps separately
            img_sal_map = self.transform_l(img_sal_map)
            img_intent = self.transform_l(img_intent)

            # CONCATENATE: RGB + Saliency + Intent (5 channels total)
            imgs = torch.concat([img_inp, img_sal_map, img_intent])

        # Rest of the method for label processing...
        sliced_df_annotated = self.groups_annotated.get_group(self.poster_name[idx])
        sliced_df_box = self.groups_box.get_group(self.poster_name[idx])  # This is salbox OR intentbox

        gt_box = torch.tensor(list(map(eval, sliced_df_annotated["box_elem"])))
        gt_cls = list(sliced_df_annotated["cls_elem"])

        # This will be saliency boxes OR intent boxes based on spatial_guidance
        sal_box = torch.tensor(list(map(eval, sliced_df_box["box_elem"])))
        sal_box = box_xyxy_to_cxcywh(sal_box)
        sal_box[:, ::2] /= width
        sal_box[:, 1::2] /= height

        label_cls = np.zeros((self.max_elem, self.num_class))
        label_box = np.zeros((self.max_elem, 4))

        for i in range(len(gt_cls)):
            if i>=self.max_elem:
                break
            label_cls[i][int(gt_cls[i])] = 1
            label_box[i] = gt_box[i]
            if label_box[i][0] > label_box[i][2] or label_box[i][1] > label_box[i][3]:
                label_box[i][:2], label_box[i][2:] = label_box[i][2:], label_box[i][:2]
            label_box[i] = box_xyxy_to_cxcywh(torch.tensor(label_box[i]))
            label_box[i][::2] /= width
            label_box[i][1::2] /= height

        for i in range(len(gt_cls), self.max_elem):
            label_cls[i][0] = 1

        label = np.concatenate((label_cls, label_box), axis=1)
        label[:, self.num_class:] = 2 * (label[:, self.num_class:] - 0.5)
        sal_box = 2 * (sal_box - 0.5)
        # Handle text control
        text_features = None
        if self.text_control:
            try:
                sliced_df_prompts = self.groups_prompts.get_group(self.poster_name[idx])
                # Randomly sample a prompt (mixes basic, spatial, enhanced, etc.)
                row = sliced_df_prompts.iloc[random.randint(0, len(sliced_df_prompts) - 1)]
                prompt_text = row.get('prompt') or row.get('text_prompt') or ''
                text_encoding = self.tokenizer(
                    prompt_text,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                text_features = {
                    'input_ids': text_encoding['input_ids'].squeeze(0),
                    'attention_mask': text_encoding['attention_mask'].squeeze(0)
                }
            except (KeyError, IndexError):
                prompt_text = ''
                text_encoding = self.tokenizer(
                    "",
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                text_features = {
                    'input_ids': text_encoding['input_ids'].squeeze(0),
                    'attention_mask': text_encoding['attention_mask'].squeeze(0)
                }

        # Return intent_box when needed (when text_control, also return prompt_text for L_text loss)
        if getattr(self, 'box_only', False):  # Mode 3: saliency pixels + intent boxes
            sliced_df_intentbox = self.groups_intent_box.get_group(self.poster_name[idx])
            intent_box = torch.tensor(list(map(eval, sliced_df_intentbox["box_elem"])))
            intent_box = box_xyxy_to_cxcywh(intent_box)
            intent_box[:, ::2] /= width
            intent_box[:, 1::2] /= height
            intent_box = 2 * (intent_box - 0.5)
            intent_box = intent_box[: self.max_elem]
            if intent_box.shape[0] < self.max_elem:
                padded = torch.full((self.max_elem, 4), -1.0, dtype=intent_box.dtype)
                padded[: intent_box.shape[0]] = intent_box
                intent_box = padded
            if self.text_control:
                return imgs, torch.tensor(label).float(), sal_box.float(), intent_box.float(), text_features, prompt_text
            return imgs, torch.tensor(label).float(), sal_box.float(), intent_box.float()
        elif self.use_intent and not self.use_saliency:  # Mode 1: Intent only
            # For intent only, we need to return intent boxes instead of saliency boxes
            if hasattr(self, 'groups_intent_box'):
                sliced_df_intentbox = self.groups_intent_box.get_group(self.poster_name[idx])
                intent_box = torch.tensor(list(map(eval, sliced_df_intentbox["box_elem"])))
                intent_box = box_xyxy_to_cxcywh(intent_box)
                intent_box[:, ::2] /= width
                intent_box[:, 1::2] /= height
                intent_box = 2 * (intent_box - 0.5)
                # Pad/truncate to max_elem so downstream (model + losses) have fixed shape [max_elem, 4]
                intent_box = intent_box[: self.max_elem]
                if intent_box.shape[0] < self.max_elem:
                    # Boxes are already in [-1, 1]; -1 maps to a zero-area box
                    # at the origin, whereas zero would map to a large 0.5 box.
                    padded = torch.full((self.max_elem, 4), -1.0, dtype=intent_box.dtype)
                    padded[: intent_box.shape[0]] = intent_box
                    intent_box = padded
                if self.text_control:
                    return imgs, torch.tensor(label).float(), intent_box.float(), text_features, prompt_text
                else:
                    return imgs, torch.tensor(label).float(), intent_box.float()
            else:
                if self.text_control:
                    return imgs, torch.tensor(label).float(), sal_box.float(), text_features, prompt_text
                else:
                    return imgs, torch.tensor(label).float(), sal_box.float()
        elif self.use_saliency and self.use_intent:  # Mode 2: Both
            # For both modes, we need both saliency and intent boxes
            if hasattr(self, 'groups_intent_box'):
                sliced_df_intentbox = self.groups_intent_box.get_group(self.poster_name[idx])
                intent_box = torch.tensor(list(map(eval, sliced_df_intentbox["box_elem"])))
                intent_box = box_xyxy_to_cxcywh(intent_box)
                intent_box[:, ::2] /= width
                intent_box[:, 1::2] /= height
                intent_box = 2 * (intent_box - 0.5)
                # Pad/truncate to max_elem so downstream (model + losses) have fixed shape [max_elem, 4]
                intent_box = intent_box[: self.max_elem]
                if intent_box.shape[0] < self.max_elem:
                    padded = torch.full((self.max_elem, 4), -1.0, dtype=intent_box.dtype)
                    padded[: intent_box.shape[0]] = intent_box
                    intent_box = padded
                if self.text_control:
                    return imgs, torch.tensor(label).float(), sal_box.float(), intent_box.float(), text_features, prompt_text
                else:
                    return imgs, torch.tensor(label).float(), sal_box.float(), intent_box.float()
            else:
                if self.text_control:
                    return imgs, torch.tensor(label).float(), sal_box.float(), text_features, prompt_text
                else:
                    return imgs, torch.tensor(label).float(), sal_box.float()
        else:  # Mode 0: Saliency only
            if self.text_control:
                return imgs, torch.tensor(label).float(), sal_box.float(), text_features, prompt_text
            else:
                return imgs, torch.tensor(label).float(), sal_box.float()


class test_uncond_dataset(Dataset):
    def __init__(self, cfg):
        img = sorted(os.listdir(cfg.paths.test.inp_dir))
        allowed = getattr(cfg, 'test_image_names', None)
        if allowed is not None:
            img = [name for name in img if name in allowed]
        # img.sort(key=cmp_to_key(natural_sort_cmp))
        save_image_order(img, cfg.imgname_order_dir)

        self.bg = list(map(lambda x: os.path.join(cfg.paths.test.inp_dir, x), img))

        # Handle spatial guidance modes
        if cfg.spatial_guidance == 0:  # Saliency only
            self.sal = list(map(lambda x: os.path.join(cfg.paths.test.sal_dir, x), img))
            self.sal_sub = list(map(lambda x: os.path.join(cfg.paths.test.sal_sub_dir, x), img))
            self.box_path = cfg.paths.test.salbox_dir
            self.use_saliency = True
            self.use_intent = False

        elif cfg.spatial_guidance == 1:  # Intent only - NO SALIENCY
            self.intent = list(map(lambda x: os.path.join(cfg.paths.test.intent_map_dir, x), img))
            self.box_path = cfg.paths.test.intentbox_dir  # Use intent boxes
            self.use_saliency = False
            self.use_intent = True

        elif cfg.spatial_guidance == 2:  # Both
            self.sal = list(map(lambda x: os.path.join(cfg.paths.test.sal_dir, x), img))
            self.sal_sub = list(map(lambda x: os.path.join(cfg.paths.test.sal_sub_dir, x), img))
            self.intent = list(map(lambda x: os.path.join(cfg.paths.test.intent_map_dir, x), img))
            self.box_path = cfg.paths.test.salbox_dir
            self.use_saliency = True
            self.use_intent = True
            self.box_only = False
        elif cfg.spatial_guidance == 3:  # Saliency pixels + intent boxes
            self.sal = list(map(lambda x: os.path.join(cfg.paths.test.sal_dir, x), img))
            self.sal_sub = list(map(lambda x: os.path.join(cfg.paths.test.sal_sub_dir, x), img))
            self.box_path = cfg.paths.test.salbox_dir
            self.use_saliency = True
            self.use_intent = False
            self.box_only = True

        self.max_elem = cfg.max_elem

        # Load saliency boxes whenever the image stream contains saliency
        # pixels. Mode 3 uses saliency pixels plus intent-box tokens, so it
        # still needs saliency boxes for the standard salbox encoder.
        if cfg.spatial_guidance in (0, 2, 3):
            df_salbox = read_csv(cfg.paths.test.salbox_dir)
            self.groups_salbox = df_salbox.groupby(df_salbox.poster_path)
        else:
            self.groups_salbox = None

        if cfg.spatial_guidance in (1, 2, 3):
            df_intentbox = read_csv(cfg.paths.test.intentbox_dir)
            self.groups_intentbox = df_intentbox.groupby(df_intentbox.poster_path)
        else:
            self.groups_intentbox = None

        self.poster_name = list(img)
        self.oracle_intent_map = getattr(cfg, 'oracle_intent_map', False)
        if self.oracle_intent_map:
            oracle_frame = read_csv(cfg.paths.test.annotated_dir)
            self.groups_oracle_layout = oracle_frame.groupby(oracle_frame.poster_path)

        # Text control support
        self.text_control = getattr(cfg, 'text_control', False)
        if self.text_control:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.prompts_df = pd.read_csv(cfg.paths.test.all_prompts)
            self.groups_prompts = self.prompts_df.groupby(self.prompts_df.poster_path)

        self.transform_rgb = transforms.Compose([
            transforms.Resize([384, 256]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.transform_l = transforms.Compose([
            transforms.Resize([384, 256]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
        ])

    def __len__(self):
        return len(self.bg)

    def _oracle_intent(self, idx, width, height):
        """Rasterize GT layout density as an explicitly oracle diagnostic map."""
        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        rows = self.groups_oracle_layout.get_group(self.poster_name[idx])
        for value in rows["box_elem"].iloc[: self.max_elem]:
            x1, y1, x2, y2 = list(map(float, eval(value)))
            draw.rectangle((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), fill=255)
        radius = max(1.0, min(width, height) * 0.02)
        return canvas.filter(ImageFilter.GaussianBlur(radius=radius))

    def __getitem__(self, idx):
        img_bg = Image.open(self.bg[idx]).convert("RGB")
        width, height = img_bg.size

        # Transform RGB image first
        img_bg = self.transform_rgb(img_bg)

        # Handle different spatial guidance modes
        if self.use_saliency and not self.use_intent:  # Mode 0: Saliency only
            img_sal = Image.open(self.sal[idx]).convert("L")
            img_sal_sub = Image.open(self.sal_sub[idx]).convert("L")
            img_sal_map = Image.fromarray(np.maximum(np.array(img_sal), np.array(img_sal_sub)))
            img_sal_map = self.transform_l(img_sal_map)
            imgs = torch.concat([img_bg, img_sal_map])

        elif self.use_intent and not self.use_saliency:  # Mode 1: Intent only - NO SALIENCY
            img_intent = (
                self._oracle_intent(idx, width, height)
                if self.oracle_intent_map
                else Image.open(self.intent[idx]).convert("L")
            )
            img_intent = self.transform_l(img_intent)
            imgs = torch.concat([img_bg, img_intent])  # Only RGB + Intent

        elif self.use_saliency and self.use_intent:  # Mode 2: Both
            img_sal = Image.open(self.sal[idx]).convert("L")
            img_sal_sub = Image.open(self.sal_sub[idx]).convert("L")
            img_sal_map = Image.fromarray(np.maximum(np.array(img_sal), np.array(img_sal_sub)))

            img_intent = (
                self._oracle_intent(idx, width, height)
                if self.oracle_intent_map
                else Image.open(self.intent[idx]).convert("L")
            )

            # Transform both maps separately
            img_sal_map = self.transform_l(img_sal_map)
            img_intent = self.transform_l(img_intent)

            # CONCATENATE: RGB + Saliency + Intent (5 channels total)
            imgs = torch.concat([img_bg, img_sal_map, img_intent])

        # Handle box loading based on spatial guidance
        if self.use_saliency:  # Mode 0 or 2: Use saliency boxes
            sliced_df_salbox = self.groups_salbox.get_group(self.poster_name[idx])
            sal_box = torch.tensor(list(map(eval, sliced_df_salbox["box_elem"])))
            sal_box = box_xyxy_to_cxcywh(sal_box)
            sal_box[:, ::2] /= width
            sal_box[:, 1::2] /= height
            sal_box = 2 * (sal_box - 0.5)
        else:  # Mode 1: Use intent boxes
            sliced_df_intentbox = self.groups_intentbox.get_group(self.poster_name[idx])
            sal_box = torch.tensor(list(map(eval, sliced_df_intentbox["box_elem"])))
            sal_box = box_xyxy_to_cxcywh(sal_box)
            sal_box[:, ::2] /= width
            sal_box[:, 1::2] /= height
            sal_box = 2 * (sal_box - 0.5)

        # Handle text control
        text_features = None
        if self.text_control:
            try:
                sliced_df_prompts = self.groups_prompts.get_group(self.poster_name[idx])
                # Get the first prompt for this image (support both 'prompt' and 'text_prompt' columns)
                row = sliced_df_prompts.iloc[0]
                prompt_text = row.get('prompt') or row.get('text_prompt') or ''
                # Tokenize the text
                text_encoding = self.tokenizer(
                    prompt_text,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                text_features = {
                    'input_ids': text_encoding['input_ids'].squeeze(0),
                    'attention_mask': text_encoding['attention_mask'].squeeze(0)
                }
            except (KeyError, IndexError):
                # If no prompt found, use empty text
                prompt_text = ''
                text_encoding = self.tokenizer(
                    "",
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                text_features = {
                    'input_ids': text_encoding['input_ids'].squeeze(0),
                    'attention_mask': text_encoding['attention_mask'].squeeze(0)
                }

        # Return intent_box when needed for test_uncond_dataset
        if getattr(self, 'box_only', False):  # Mode 3: saliency pixels + intent boxes
            sliced_df_intentbox = self.groups_intentbox.get_group(self.poster_name[idx])
            intent_box = torch.tensor(list(map(eval, sliced_df_intentbox["box_elem"])))
            intent_box = box_xyxy_to_cxcywh(intent_box)
            intent_box[:, ::2] /= width
            intent_box[:, 1::2] /= height
            intent_box = 2 * (intent_box - 0.5)
            intent_box = intent_box[: self.max_elem]
            if intent_box.shape[0] < self.max_elem:
                padded = torch.full((self.max_elem, 4), -1.0, dtype=intent_box.dtype)
                padded[: intent_box.shape[0]] = intent_box
                intent_box = padded
            if self.text_control:
                return imgs, sal_box.float(), intent_box.float(), text_features, prompt_text
            return imgs, sal_box.float(), intent_box.float()
        elif self.use_intent and not self.use_saliency:  # Mode 1: Intent only
            if hasattr(self, 'groups_intentbox'):
                sliced_df_intentbox = self.groups_intentbox.get_group(self.poster_name[idx])
                intent_box = torch.tensor(list(map(eval, sliced_df_intentbox["box_elem"])))
                intent_box = box_xyxy_to_cxcywh(intent_box)
                intent_box[:, ::2] /= width
                intent_box[:, 1::2] /= height
                intent_box = 2 * (intent_box - 0.5)
                # Pad/truncate to max_elem so downstream (model + metrics) have fixed shape [max_elem, 4]
                intent_box = intent_box[: self.max_elem]
                if intent_box.shape[0] < self.max_elem:
                    padded = torch.full((self.max_elem, 4), -1.0, dtype=intent_box.dtype)
                    padded[: intent_box.shape[0]] = intent_box
                    intent_box = padded
                if self.text_control:
                    return imgs, intent_box.float(), text_features, prompt_text
                else:
                    return imgs, intent_box.float()
            else:
                if self.text_control:
                    return imgs, sal_box.float(), text_features, prompt_text
                else:
                    return imgs, sal_box.float()
        elif self.use_saliency and self.use_intent:  # Mode 2: Both
            if hasattr(self, 'groups_intentbox'):
                sliced_df_intentbox = self.groups_intentbox.get_group(self.poster_name[idx])
                intent_box = torch.tensor(list(map(eval, sliced_df_intentbox["box_elem"])))
                intent_box = box_xyxy_to_cxcywh(intent_box)
                intent_box[:, ::2] /= width
                intent_box[:, 1::2] /= height
                intent_box = 2 * (intent_box - 0.5)
                # Pad/truncate to max_elem so downstream (model + metrics) have fixed shape [max_elem, 4]
                intent_box = intent_box[: self.max_elem]
                if intent_box.shape[0] < self.max_elem:
                    padded = torch.full((self.max_elem, 4), -1.0, dtype=intent_box.dtype)
                    padded[: intent_box.shape[0]] = intent_box
                    intent_box = padded
                if self.text_control:
                    return imgs, sal_box.float(), intent_box.float(), text_features, prompt_text
                else:
                    return imgs, sal_box.float(), intent_box.float()
            else:
                if self.text_control:
                    return imgs, sal_box.float(), text_features, prompt_text
                else:
                    return imgs, sal_box.float()
        else:  # Mode 0: Saliency only
            if self.text_control:
                return imgs, sal_box.float(), text_features, prompt_text
            else:
                return imgs, sal_box.float()

class test_cond_dataset(Dataset):
    def __init__(self, cfg):
        img = sorted(os.listdir(cfg.paths.test.inp_dir))
        allowed = getattr(cfg, 'test_image_names', None)
        if allowed is not None:
            img = [name for name in img if name in allowed]
        save_image_order(img, cfg.imgname_order_dir)

        self.inp = list(map(lambda x: os.path.join(cfg.paths.test.inp_dir, x), img))

        # Handle spatial guidance modes
        if cfg.spatial_guidance == 0:  # Saliency only
            self.sal = list(map(lambda x: os.path.join(cfg.paths.test.sal_dir, x), img))
            self.sal_sub = list(map(lambda x: os.path.join(cfg.paths.test.sal_sub_dir, x), img))
            self.box_path = cfg.paths.test.salbox_dir
            self.use_saliency = True
            self.use_intent = False

        elif cfg.spatial_guidance == 1:  # Intent only - NO SALIENCY
            self.intent = list(map(lambda x: os.path.join(cfg.paths.test.intent_map_dir, x), img))
            self.box_path = cfg.paths.test.intentbox_dir  # Use intent boxes
            self.use_saliency = False
            self.use_intent = True

        elif cfg.spatial_guidance == 2:  # Both
            self.sal = list(map(lambda x: os.path.join(cfg.paths.test.sal_dir, x), img))
            self.sal_sub = list(map(lambda x: os.path.join(cfg.paths.test.sal_sub_dir, x), img))
            self.intent = list(map(lambda x: os.path.join(cfg.paths.test.intent_map_dir, x), img))
            self.box_path = cfg.paths.test.salbox_dir
            self.use_saliency = True
            self.use_intent = True

        self.max_elem = cfg.max_elem

        # Only load boxes for the mode we're using
        if cfg.spatial_guidance == 0 or cfg.spatial_guidance == 2:
            df_salbox = read_csv(cfg.paths.test.salbox_dir)
            self.groups_salbox = df_salbox.groupby(df_salbox.poster_path)
        else:
            self.groups_salbox = None

        if cfg.spatial_guidance == 1 or cfg.spatial_guidance == 2:
            df_intentbox = read_csv(cfg.paths.test.intentbox_dir)
            self.groups_intentbox = df_intentbox.groupby(df_intentbox.poster_path)
        else:
            self.groups_intentbox = None

        self.poster_name = list(img)

        # Text control support
        self.text_control = getattr(cfg, 'text_control', False)
        if self.text_control:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.prompts_df = pd.read_csv(cfg.paths.test.all_prompts)
            self.groups_prompts = self.prompts_df.groupby(self.prompts_df.poster_path)

        self.transform_rgb = transforms.Compose([
            transforms.Resize([384, 256]),  # (360,240)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.transform_l = transforms.Compose([
            transforms.Resize([384, 256]),  # (360,240)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
        ])

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        img_inp = Image.open(self.inp[idx]).convert("RGB")
        width, height = img_inp.size

        # Transform RGB image first
        img_inp = self.transform_rgb(img_inp)

        # Handle different spatial guidance modes
        if self.use_saliency and not self.use_intent:  # Mode 0: Saliency only
            img_sal = Image.open(self.sal[idx]).convert("L")
            img_sal_sub = Image.open(self.sal_sub[idx]).convert("L")
            img_sal_map = Image.fromarray(np.maximum(np.array(img_sal), np.array(img_sal_sub)))
            img_sal_map = self.transform_l(img_sal_map)
            imgs = torch.concat([img_inp, img_sal_map])

        elif self.use_intent and not self.use_saliency:  # Mode 1: Intent only - NO SALIENCY
            img_intent = Image.open(self.intent[idx]).convert("L")
            img_intent = self.transform_l(img_intent)
            imgs = torch.concat([img_inp, img_intent])  # Only RGB + Intent

        elif self.use_saliency and self.use_intent:  # Mode 2: Both
            img_sal = Image.open(self.sal[idx]).convert("L")
            img_sal_sub = Image.open(self.sal_sub[idx]).convert("L")
            img_sal_map = Image.fromarray(np.maximum(np.array(img_sal), np.array(img_sal_sub)))

            img_intent = Image.open(self.intent[idx]).convert("L")

            # Transform both maps separately
            img_sal_map = self.transform_l(img_sal_map)
            img_intent = self.transform_l(img_intent)

            # CONCATENATE: RGB + Saliency + Intent (5 channels total)
            imgs = torch.concat([img_inp, img_sal_map, img_intent])

        # Handle box loading based on spatial guidance
        if self.use_saliency:  # Mode 0 or 2: Use saliency boxes
            sliced_df_salbox = self.groups_salbox.get_group(self.poster_name[idx])
            sal_box = torch.tensor(list(map(eval, sliced_df_salbox["box_elem"])))
            sal_box = box_xyxy_to_cxcywh(sal_box)
            sal_box[:, ::2] /= width
            sal_box[:, 1::2] /= height
            sal_box = 2 * (sal_box - 0.5)
        else:  # Mode 1: Use intent boxes
            sliced_df_intentbox = self.groups_intentbox.get_group(self.poster_name[idx])
            sal_box = torch.tensor(list(map(eval, sliced_df_intentbox["box_elem"])))
            sal_box = box_xyxy_to_cxcywh(sal_box)
            sal_box[:, ::2] /= width
            sal_box[:, 1::2] /= height
            sal_box = 2 * (sal_box - 0.5)

        # Handle text control
        text_features = None
        if self.text_control:
            try:
                sliced_df_prompts = self.groups_prompts.get_group(self.poster_name[idx])
                # Get the first prompt for this image (support both 'prompt' and 'text_prompt' columns)
                row = sliced_df_prompts.iloc[0]
                prompt_text = row.get('prompt') or row.get('text_prompt') or ''
                # Tokenize the text
                text_encoding = self.tokenizer(
                    prompt_text,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                text_features = {
                    'input_ids': text_encoding['input_ids'].squeeze(0),
                    'attention_mask': text_encoding['attention_mask'].squeeze(0)
                }
            except (KeyError, IndexError):
                # If no prompt found, use empty text
                text_encoding = self.tokenizer(
                    "",
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                text_features = {
                    'input_ids': text_encoding['input_ids'].squeeze(0),
                    'attention_mask': text_encoding['attention_mask'].squeeze(0)
                }

        # Return intent_box when needed for test_cond_dataset
        if getattr(self, 'box_only', False):  # Mode 3: saliency pixels + intent boxes
            sliced_df_intentbox = self.groups_intentbox.get_group(self.poster_name[idx])
            intent_box = torch.tensor(list(map(eval, sliced_df_intentbox["box_elem"])))
            intent_box = box_xyxy_to_cxcywh(intent_box)
            intent_box[:, ::2] /= width
            intent_box[:, 1::2] /= height
            intent_box = 2 * (intent_box - 0.5)
            # Pad/truncate to max_elem so downstream (model + metrics) have fixed shape [max_elem, 4]
            intent_box = intent_box[: self.max_elem]
            if intent_box.shape[0] < self.max_elem:
                padded = torch.full((self.max_elem, 4), -1.0, dtype=intent_box.dtype)
                padded[: intent_box.shape[0]] = intent_box
                intent_box = padded
            if self.text_control:
                return imgs, sal_box.float(), intent_box.float(), text_features, prompt_text
            else:
                return imgs, sal_box.float(), intent_box.float()
        elif self.use_intent and not self.use_saliency:  # Mode 1: Intent only
            if hasattr(self, 'groups_intentbox'):
                sliced_df_intentbox = self.groups_intentbox.get_group(self.poster_name[idx])
                intent_box = torch.tensor(list(map(eval, sliced_df_intentbox["box_elem"])))
                intent_box = box_xyxy_to_cxcywh(intent_box)
                intent_box[:, ::2] /= width
                intent_box[:, 1::2] /= height
                intent_box = 2 * (intent_box - 0.5)
                if self.text_control:
                    return imgs, intent_box.float(), text_features
                else:
                    return imgs, intent_box.float()
            else:
                if self.text_control:
                    return imgs, sal_box.float(), text_features
                else:
                    return imgs, sal_box.float()
        elif self.use_saliency and self.use_intent:  # Mode 2: Both
            if hasattr(self, 'groups_intentbox'):
                sliced_df_intentbox = self.groups_intentbox.get_group(self.poster_name[idx])
                intent_box = torch.tensor(list(map(eval, sliced_df_intentbox["box_elem"])))
                intent_box = box_xyxy_to_cxcywh(intent_box)
                intent_box[:, ::2] /= width
                intent_box[:, 1::2] /= height
                intent_box = 2 * (intent_box - 0.5)
                if self.text_control:
                    return imgs, sal_box.float(), intent_box.float(), text_features
                else:
                    return imgs, sal_box.float(), intent_box.float()
            else:
                if self.text_control:
                    return imgs, sal_box.float(), text_features
                else:
                    return imgs, sal_box.float()
        else:  # Mode 0: Saliency only
            if self.text_control:
                return imgs, sal_box.float(), text_features
            else:
                return imgs, sal_box.float()
