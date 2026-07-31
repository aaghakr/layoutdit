# Free-form Prompt Authoring App

This small web app lets a designer fill the independent free-form prompt CSVs
used by stages `07`, `09`, `11`, and the user study preparation.

It has no third-party dependencies; it uses Python's standard library and writes
directly to:

- `free_form_pku.csv`
- `free_form_cgl.csv`

## Configure image folders

Edit `config.json` if your images are elsewhere:

```json
{
  "datasets": {
    "pku": {
      "csv": "free_form_pku.csv",
      "image_dir": "../data/dataset/pku/split/test_anno/inpaint"
    },
    "cgl": {
      "csv": "free_form_cgl.csv",
      "image_dir": "../data/dataset/cgl/split/test_anno/inpaint"
    }
  }
}
```

The app displays `image_dir/poster_path` for each row in the CSV.

## Run

From the project root:

```bash
cd prompt_app
python app.py --host 0.0.0.0 --port 7860
```

Then open:

```text
http://SERVER_IP:7860
```

If the server firewall blocks the port, use an SSH tunnel from your laptop:

```bash
ssh -L 7860:localhost:7860 USER@SERVER_IP
```

Then open:

```text
http://localhost:7860
```

## Prompt writing rules

- Write one natural user instruction per image, usually 12–35 words.
- Look only at the image/background, not ground-truth boxes or annotation CSVs.
- Use only the classes supported by the selected dataset/model:
  - PKU: `text` / `text box`, `logo` / `icon`, `underlay` / `panel`.
  - CGL: `text` / `text box`, `logo` / `icon`, `underlay` / `panel`,
    `embellishment` / `decoration` / `graphic element`.
- For PKU, avoid `decoration`, `embellishment`, and `graphic element`; PKU has
  no embellishment class in the model.
- For CGL, `embellishment`, `decoration`, and `graphic element` are allowed for
  decorative visual objects.
- If describing a title or caption, include the class word too, e.g.
  `one text box for the title` or `caption text at bottom-center`; do not rely
  on only `title`/`caption`.
- Keep prompts natural, but keep class words and spatial keywords clear enough
  for the model to read. Mild natural variation is fine; avoid spelling
  mistakes in key words such as `text`, `logo`, `underlay`, and `top-center`.
- Use exact 3×3 spatial phrases when asking for position: `top-left`,
  `top-center`, `top-right`, `middle-left`, `middle-center`, `middle-right`,
  `bottom-left`, `bottom-center`, `bottom-right`.
- Keep `independent_of_ground_truth` as `yes` if the prompt was written without
  viewing ground-truth layout boxes.

Example:

```text
Create a clean poster with one text box for the title at top-center, two short
text boxes at middle-center, and a small logo at bottom-right.
```

## Validate after authoring

```bash
cd code

python scripts/validate_freeform_prompts.py \
  --prompts ../prompt_app/free_form_pku.csv \
  --annotations ../data/dataset/pku/split/csv/test.csv

python scripts/validate_freeform_prompts.py \
  --prompts ../prompt_app/free_form_cgl.csv \
  --annotations ../data/dataset/cgl/split/csv/test.csv
```

If you keep the official copies under `data/prompts/`, copy the completed CSVs:

```bash
cp ../prompt_app/free_form_pku.csv ../data/prompts/free_form_pku.csv
cp ../prompt_app/free_form_cgl.csv ../data/prompts/free_form_cgl.csv
```
