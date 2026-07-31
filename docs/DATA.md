# Data layout

Datasets and model weights are not distributed in this repository. Expected
paths are rooted at `data/`:

## Public benchmark sources

IntentDiT is evaluated on two public content-aware poster-layout benchmarks:

- **PKU PosterLayout:** official project page
  [https://mipl.pku.edu.cn/PosterLayout/](https://mipl.pku.edu.cn/PosterLayout/)
  and official code/data repository
  [https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023](https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023).
  A convenience Hugging Face mirror is also available at
  [https://huggingface.co/datasets/creative-graphic-design/PKU-PosterLayout](https://huggingface.co/datasets/creative-graphic-design/PKU-PosterLayout).
- **CGL Dataset:** original CGL-GAN release
  [https://github.com/minzhouGithub/CGL-GAN](https://github.com/minzhouGithub/CGL-GAN),
  which points to the Tianchi dataset page
  [https://tianchi.aliyun.com/dataset/142692](https://tianchi.aliyun.com/dataset/142692).
  A Hugging Face page for CGL-Dataset-v2 is available at
  [https://huggingface.co/datasets/creative-graphic-design/CGL-Dataset-v2](https://huggingface.co/datasets/creative-graphic-design/CGL-Dataset-v2).

Please follow the original dataset terms and access requirements. The artifact
bundle below contains the processed files used by this repository, but the
official benchmark pages remain the authoritative sources for dataset access and
licensing.

## Processed artifacts

Download the external artifact bundle for datasets, checkpoints, model weights,
and generated evidence files:

```text
https://drive.google.com/drive/folders/1TrXbdr_ItKoHe7GZIe4XWne9uy1S8-E2?usp=drive_link
```

Place the downloaded files under the paths shown below.

```text
data/
  dataset/
    pku/split/
      train/{inpaint,saliency,saliency_sub,intent_map}/
      val/{inpaint,saliency,saliency_sub,intent_map}/
      test_anno/{inpaint,saliency,saliency_sub,intent_map}/
      test_unanno/{inpaint,saliency,saliency_sub,intent_map}/
      csv/
    cgl/split/
      train/{inpaint,saliency,saliency_sub,intent_map}/
      val/{inpaint,saliency,saliency_sub,intent_map}/
      test_anno/{inpaint,saliency,saliency_sub,intent_map}/
      test_unanno/{inpaint,saliency,saliency_sub,intent_map}/
      csv/
  model_weights/
    intent_map/
    saliency_detection/
  checkpoints/{pku,cgl}/
  output/
```

CSV filenames are specified in `code/configs/*.yaml`. Keep official benchmark
splits unchanged and record checksums for any generated prompt CSVs.

## Custom prompt data

The original PKU PosterLayout and CGL annotations do not provide natural-language
instructions for prompt controllability. We therefore provide custom prompt
artifacts for evaluating text-guided layout generation:

- **Template prompt families:** deterministic prompts generated from the
  annotation CSVs, covering class/count descriptions and coarse spatial
  language. These are used for controlled prompt-adherence evaluation.
- **Stress-test prompts:** automatically constructed relation, out-of-domain,
  and conflict prompts used to diagnose parser and instruction-following
  behavior.
- **Free-form prompts:** independently authored image-grounded prompts for a
  held-out subset of PKU and CGL images. Human prompt authors wrote these
  prompts using `prompt_app/` while viewing the background image but not the
  ground-truth layout boxes. Each row contains:

```text
poster_path,text_prompt,author_id,independent_of_ground_truth
```

The public seed files are:

```text
prompt_app/free_form_pku.csv
prompt_app/free_form_cgl.csv
```

These free-form prompts are intended to represent natural user requests, not
oracle descriptions of the original annotations. They include requests for
element classes, counts, and coarse spatial placement, and are used for
free-form prompt robustness, manual reference evaluation, and instruction
following analysis.

Reconstruct missing derived inputs with:

```bash
PATH_PROFILE=server bash scripts/paper/prepare_missing_inputs.sh
```

Oracle prompt files are deterministically derived from annotations. The generated
`free_form_*_template.csv` files contain blank assignments only: their prompts must
be written independently without access to ground-truth boxes, then saved without
the `_template` suffix. Do not substitute oracle prompts for this evaluation.
