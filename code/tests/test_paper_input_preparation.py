import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.prepare_paper_inputs import (
    create_freeform_template,
    prepare_derived_prompts,
    sha256,
)


class PaperInputPreparationTests(unittest.TestCase):
    def test_prompt_artifacts_are_complete_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_dir = root / "data/dataset/pku/split/csv"
            csv_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "poster_path": ["a.png", "a.png", "b.png"],
                    "cls_elem": [1, 2, 3],
                    "box_elem": [
                        "[10, 10, 110, 110]",
                        "[300, 600, 400, 700]",
                        "[100, 300, 300, 500]",
                    ],
                }
            ).to_csv(csv_dir / "test.csv", index=False)

            first = prepare_derived_prompts(
                root, "pku", ("test",), True, rich_variations=2, seed=2026
            )
            first_hashes = {path.name: sha256(path) for path in first}
            second = prepare_derived_prompts(
                root, "pku", ("test",), True, rich_variations=2, seed=2026
            )
            second_hashes = {path.name: sha256(path) for path in second}

            self.assertEqual(first_hashes, second_hashes)
            self.assertTrue((csv_dir / "test_with_prompts_spatial.csv").is_file())
            self.assertEqual(
                len(pd.read_csv(csv_dir / "test_with_rich_prompts.csv")), 4
            )
            self.assertEqual(
                len(pd.read_csv(csv_dir / "test_with_all_prompts.csv")), 12
            )

            template = create_freeform_template(root, "pku", 2, 2026, False)
            frame = pd.read_csv(template, keep_default_na=False)
            self.assertEqual(len(frame), 2)
            self.assertTrue((frame.text_prompt == "").all())


if __name__ == "__main__":
    unittest.main()
