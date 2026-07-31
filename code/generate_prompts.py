#!/usr/bin/env python3
"""
Comprehensive script for generating all types of prompts and fixing CSV formatting
Combines: generate_prompt.py, generate_rich_prompt.py, and fix_csv_quotes.py
"""

import pandas as pd
from collections import Counter
import os
import ast
import random
import csv
from pathlib import Path

try:
    import inflect
except ImportError:
    class _SimpleInflector:
        _WORDS = {
            0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
            5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
            10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
            14: "fourteen", 15: "fifteen", 16: "sixteen",
        }

        def number_to_words(self, number):
            return self._WORDS.get(int(number), str(number))

        @staticmethod
        def plural(term):
            if term.endswith(("s", "x", "z", "ch", "sh")):
                return term + "es"
            if term.endswith("y") and len(term) > 1 and term[-2].lower() not in "aeiou":
                return term[:-1] + "ies"
            return term + "s"

    class _InflectFallback:
        @staticmethod
        def engine():
            return _SimpleInflector()

    inflect = _InflectFallback()

# Try to import augly, but don't fail if it's not available
try:
    import augly.text as textaugs
    AUGLY_AVAILABLE = True
except ImportError:
    AUGLY_AVAILABLE = False
    print("⚠️  AugLy not available. Rich prompts will be generated without text augmentation.")

def number_to_words(num):
    """Convert numbers to words using inflect library."""
    p = inflect.engine()
    return p.number_to_words(num)

def get_position(box, canvas_width=513, canvas_height=750):
    """Determines the position of an element on the canvas."""
    if not box or len(box) != 4:
        return "an unknown position"
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2

    position = ""
    if y_center < canvas_height / 3:
        position += "top"
    elif y_center > 2 * canvas_height / 3:
        position += "bottom"
    else:
        position += "middle"

    if x_center < canvas_width / 3:
        position += "-left"
    elif x_center > 2 * canvas_width / 3:
        position += "-right"
    else:
        position += "-center"
    return position

def get_relative_position(box1, box2):
    """Determines the relative position between two elements."""
    if box1[3] < box2[1]:  # y2 of box1 is above y1 of box2
        return "above"
    if box1[1] > box2[3]:  # y1 of box1 is below y2 of box2
        return "below"
    if box1[2] < box2[0]:  # x2 of box1 is to the left of x1 of box2
        return "to the left of"
    if box1[0] > box2[2]:  # x1 of box1 is to the right of x2 of box2
        return "to the right of"
    return "overlapping with"

def create_text_prompts_from_csv(input_csv_path: str, output_csv_path: str, dataset_name: str = "pku", prompt_style: str = "enhanced"):
    """
    Reads a layout CSV, groups elements by image, and generates
    natural language prompts for each layout.

    Args:
        input_csv_path: Path to the input CSV file (e.g., 'train.csv').
        output_csv_path: Path to save the new CSV with prompts.
        dataset_name: Dataset name ('pku' or 'cgl') to determine class mapping.
        prompt_style: Style of prompts ('basic', 'enhanced', 'advanced').
    """
    # Class mapping based on your dataset analysis
    if dataset_name.lower() == "pku":
        CLASS_MAP = {
            1: 'Text',           # Most common element
            2: 'Logo',            # Second most common
            3: 'Underlay'        # Background/underlay elements
        }
    elif dataset_name.lower() == "cgl":
        CLASS_MAP = {
            1: 'Text',           # Most common element
            2: 'Logo',            # Second most common
            3: 'Underlay',       # Background/underlay elements
            4: 'Embellishment'   # Decorative elements
        }
    else:
        # Default mapping for unknown datasets
        CLASS_MAP = {
            1: 'Text',
            2: 'Logo',
            3: 'Underlay',
            4: 'Embellishment'
        }

    try:
        df = pd.read_csv(input_csv_path)
        print(f"  📊 Loaded CSV with {len(df)} rows")
    except FileNotFoundError:
        print(f"  ❌ Error: The file {input_csv_path} was not found.")
        return False
    except Exception as e:
        print(f"  ❌ Error reading CSV: {e}")
        return False

    # Check if required columns exist
    required_columns = ['poster_path', 'cls_elem']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"  ❌ Error: Missing required columns: {missing_columns}")
        return False

    # Group all elements by the poster_path
    grouped = df.groupby('poster_path')
    print(f"  📁 Found {len(grouped)} unique images")

    prompt_data = []
    processed_count = 0

    for poster_path, group in grouped:
        # Get all class IDs for the current image
        class_ids = group['cls_elem'].tolist()
        boxes = group['box_elem'].tolist()

        # Map IDs to names
        class_names = [CLASS_MAP.get(cid, f'Unknown_{cid}') for cid in class_ids]

        # Count the occurrences of each element type
        element_counts = Counter(class_names)

        # --- Generate text prompt based on style ---
        if prompt_style == "basic":
            # Simple count-based prompts with mixed numeric/natural language
            prompt_parts = []
            p = inflect.engine()
            for class_name, count in sorted(element_counts.items()):
                plural = 's' if count > 1 else ''
                if random.random() < 0.5:
                    count_str = str(count)
                else:
                    count_str = p.number_to_words(count)
                prompt_parts.append(f"{count_str} {class_name}{plural}")
            prompt = "A layout with " + ", ".join(prompt_parts) + "."

        elif prompt_style == "enhanced":
            # Enhanced prompts with positional information
            prompt_parts = []

            # Group elements by type and position
            element_positions = {}
            for i, (class_name, box_str) in enumerate(zip(class_names, boxes)):
                try:
                    # Parse the box coordinates
                    box = ast.literal_eval(box_str)
                    position = get_position(box)

                    if class_name not in element_positions:
                        element_positions[class_name] = []
                    element_positions[class_name].append(position)
                except:
                    # Fallback if box parsing fails
                    if class_name not in element_positions:
                        element_positions[class_name] = []
                    element_positions[class_name].append("position-unknown")

            # Generate descriptive prompts with mixed numeric/natural language
            p = inflect.engine()
            for class_name, count in sorted(element_counts.items()):
                plural = 's' if count > 1 else ''
                if random.random() < 0.5:
                    count_str = str(count)
                else:
                    count_str = p.number_to_words(count)

                if class_name in element_positions and len(element_positions[class_name]) > 0:
                    # Get unique positions for this element type
                    positions = sorted(set(element_positions[class_name]))
                    if len(positions) == 1:
                        prompt_parts.append(f"{count_str} {class_name}{plural} in the {positions[0]}")
                    else:
                        # Multiple positions - describe the distribution
                        if count <= 3:
                            position_desc = ", ".join(positions)
                            prompt_parts.append(f"{count_str} {class_name}{plural} in the {position_desc}")
                        else:
                            prompt_parts.append(f"{count_str} {class_name}{plural} distributed across the layout")
                else:
                    prompt_parts.append(f"{count_str} {class_name}{plural}")

            prompt = "A layout with " + ", ".join(prompt_parts) + "."

        elif prompt_style == "advanced":
            # Advanced prompts with relative positioning
            prompt_parts = []

            # Parse all boxes
            parsed_boxes = []
            for box_str in boxes:
                try:
                    box = ast.literal_eval(box_str)
                    parsed_boxes.append(box)
                except:
                    parsed_boxes.append(None)

            # Generate advanced prompts with relationships and mixed numeric/natural language
            p = inflect.engine()
            for class_name, count in sorted(element_counts.items()):
                plural = 's' if count > 1 else ''
                if random.random() < 0.5:
                    count_str = str(count)
                else:
                    count_str = p.number_to_words(count)

                if count == 1:
                    # Single element - describe its position
                    element_indices = [i for i, name in enumerate(class_names) if name == class_name]
                    if element_indices and parsed_boxes[element_indices[0]]:
                        position = get_position(parsed_boxes[element_indices[0]])
                        prompt_parts.append(f"{count_str} {class_name} in the {position}")
                    else:
                        prompt_parts.append(f"{count_str} {class_name}")
                else:
                    # Multiple elements - describe distribution
                    element_indices = [i for i, name in enumerate(class_names) if name == class_name]
                    valid_boxes = [parsed_boxes[i] for i in element_indices if parsed_boxes[i] is not None]

                    if valid_boxes:
                        positions = [get_position(box) for box in valid_boxes]
                        unique_positions = sorted(set(positions))

                        if len(unique_positions) == 1:
                            prompt_parts.append(f"{count_str} {class_name}{plural} in the {unique_positions[0]}")
                        else:
                            prompt_parts.append(f"{count_str} {class_name}{plural} distributed across the layout")
                    else:
                        prompt_parts.append(f"{count_str} {class_name}{plural}")

            prompt = "A layout with " + ", ".join(prompt_parts) + "."

        elif prompt_style == "spatial":
            # Spatial prompts: EVERY element gets an explicit position keyword.
            # Critical for training the text-spatial grounding module.
            prompt_parts = []
            p = inflect.engine()

            # Parse all boxes
            parsed_boxes = []
            for box_str in boxes:
                try:
                    box = ast.literal_eval(box_str)
                    parsed_boxes.append(box)
                except:
                    parsed_boxes.append(None)

            # Group elements with their per-element positions
            element_position_lists = {}
            for i, (class_name, box) in enumerate(zip(class_names, parsed_boxes)):
                if box is not None:
                    position = get_position(box)
                else:
                    position = "middle-center"
                element_position_lists.setdefault(class_name, []).append(position)

            for class_name in sorted(element_position_lists.keys()):
                positions = element_position_lists[class_name]
                count = len(positions)
                plural = 's' if count > 1 else ''
                if random.random() < 0.5:
                    count_str = str(count)
                else:
                    count_str = p.number_to_words(count)

                # Always include all positions (group same positions)
                pos_counter = Counter(positions)
                if len(pos_counter) == 1:
                    pos_name = list(pos_counter.keys())[0]
                    prompt_parts.append(f"{count_str} {class_name}{plural} at {pos_name}")
                else:
                    pos_desc = " and ".join(
                        f"{c} {class_name}{'s' if c > 1 else ''} at {p}"
                        for p, c in pos_counter.items()
                    )
                    prompt_parts.append(pos_desc)

            template = random.choice([
                "A layout with {}.",
                "Place {}.",
                "Design with {}.",
            ])
            prompt = template.format(", ".join(prompt_parts))

        else:
            # Default to basic style
            prompt_parts = []
            for class_name, count in sorted(element_counts.items()):
                plural = 's' if count > 1 else ''
                prompt_parts.append(f"{count} {class_name}{plural}")
            prompt = "A layout with " + ", ".join(prompt_parts) + "."

        if not prompt_parts:
            continue

        prompt_data.append({'poster_path': poster_path, 'text_prompt': prompt})
        processed_count += 1

    # Save the generated prompts to a new CSV file with proper quoting
    prompt_df = pd.DataFrame(prompt_data)
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        # Write header
        writer.writerow(['poster_path', 'text_prompt'])
        # Write data
        for _, row in prompt_df.iterrows():
            writer.writerow([row['poster_path'], row['text_prompt']])

    print(f"  ✅ Successfully generated {processed_count} prompts and saved to {output_csv_path}")

    # Show sample of generated prompts
    if len(prompt_df) > 0:
        print(f"  📝 Sample prompts:")
        for i, row in prompt_df.head(2).iterrows():
            print(f"    {row['poster_path']}: {row['text_prompt']}")

    return True

def create_rich_text_prompts(
    input_csv_path: str,
    output_csv_path: str,
    dataset_name: str = "pku",
    num_variations: int = 3,
    use_augmentation: bool = True,
):
    """
    Reads a layout CSV and generates rich, augmented natural language prompts.

    Args:
        input_csv_path: Path to the input CSV file.
        output_csv_path: Path to save the new CSV with prompts.
        dataset_name: 'pku' or 'cgl' to determine class mapping.
        num_variations: Number of different prompts to generate for each image.
    """
    p = inflect.engine()

    # Define class maps and synonym maps for augmentation
    if dataset_name.lower() == "pku":
        CLASS_MAP = {1: 'Text', 2: 'Logo', 3: 'Underlay'}
        SYNONYM_MAP = {
            'Text': ['text', 'text box', 'text field'],
            'Logo': ['logo', 'icon', 'brand mark'],
            'Underlay': ['underlay', 'background panel', 'panel']
        }
    else:  # cgl
        CLASS_MAP = {1: 'Text', 2: 'Logo', 3: 'Underlay', 4: 'Embellishment'}
        SYNONYM_MAP = {
            'Text': ['text', 'text box'],
            'Logo': ['logo', 'icon'],
            'Underlay': ['underlay', 'panel'],
            'Embellishment': ['embellishment', 'decoration', 'graphic element']
        }

    try:
        df = pd.read_csv(input_csv_path)
        print(f"  📊 Loaded CSV with {len(df)} rows")
    except FileNotFoundError:
        print(f"  ❌ Error: File not found at {input_csv_path}")
        return False

    grouped = df.groupby('poster_path')
    prompt_data = []

    for poster_path, group in grouped:
        elements = []
        for _, row in group.iterrows():
            try:
                box = ast.literal_eval(row['box_elem'])
                class_name = CLASS_MAP.get(row['cls_elem'], 'element')
                position = get_position(box)
                elements.append({'class_name': class_name, 'position': position})
            except (ValueError, SyntaxError):
                continue

        if not elements:
            continue

        # Generate multiple prompt variations for each image
        for _ in range(num_variations):
            random.shuffle(elements)
            element_counts = Counter(el['class_name'] for el in elements)

            # --- 1. Choose a Sentence Template ---
            template = random.choice([
                "A layout with {}.",
                "Generate a design that includes {}.",
                "Create a poster containing {}.",
                "This layout has {}."
            ])

            # --- 2. Build the Description Parts ---
            prompt_parts = []
            for class_name, count in sorted(element_counts.items()):
                term = random.choice(SYNONYM_MAP.get(class_name, [class_name]))
                if count > 1:
                    term = p.plural(term)

                # Add positional context for single elements
                if count == 1:
                    pos = [el['position'] for el in elements if el['class_name'] == class_name][0]
                    prompt_parts.append(f"a single {term} in the {pos}")
                else:
                    # Use mixed numeric/natural language numbers
                    if random.random() < 0.5:
                        count_str = str(count)
                    else:
                        count_str = p.number_to_words(count)
                    prompt_parts.append(f"{count_str} {term}")

            description = ", ".join(prompt_parts)
            base_prompt = template.format(description)

            # --- 3. Apply AugLy for Robustness (if available) ---
            if use_augmentation and AUGLY_AVAILABLE and random.random() < 0.5: # Apply augmentation 50% of the time
                aug_function_class = random.choice([
                    textaugs.ReplaceSimilarChars,
                    textaugs.SimulateTypos,
                ])
                aug_function = aug_function_class()
                final_prompt = aug_function(base_prompt)
                if isinstance(final_prompt, list):
                    final_prompt = final_prompt[0]
            else:
                final_prompt = base_prompt

            prompt_data.append({'poster_path': poster_path, 'text_prompt': final_prompt})

    # Save to CSV with proper quoting
    prompt_df = pd.DataFrame(prompt_data)
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(['poster_path', 'text_prompt'])
        for _, row in prompt_df.iterrows():
            writer.writerow([row['poster_path'], row['text_prompt']])

    print(f"  ✅ Successfully generated {len(prompt_df)} prompts and saved to {output_csv_path}")

    if not prompt_df.empty:
        print(f"  📝 Sample prompts:")
        sample_df = prompt_df.sample(min(3, len(prompt_df)))
        for _, row in sample_df.iterrows():
            print(f"    {row['poster_path']}: {row['text_prompt']}")

    return True

def fix_csv_quotes(file_path):
    """
    Fix missing quotes in CSV file by re-reading and re-writing with proper quoting

    Args:
        file_path: Path to CSV file to fix
    """
    print(f"  🔧 Fixing quotes in: {file_path}")

    try:
        df = pd.read_csv(file_path)
        print(f"    📊 Loaded {len(df)} rows")

        # Create backup
        backup_path = str(file_path).replace('.csv', '_backup_quotes.csv')
        df.to_csv(backup_path, index=False)
        print(f"    💾 Created backup: {backup_path}")

        # Re-write with proper quoting
        df.to_csv(file_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"    ✅ Fixed quotes and saved to: {file_path}")

        return True

    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False

def merge_prompts_for_split(dataset: str, split: str) -> bool:
    """Merge rich/basic/enhanced/advanced into one CSV per split.

    Output: dataset/{dataset}/split/csv/{split}_with_all_prompts.csv
    Order: rich (if exists) → basic → enhanced → advanced
    Columns: poster_path,text_prompt (quoted)
    """
    base = f"dataset/{dataset}/split/csv/{split}_with_"
    files_in_order = [
        base + "rich_prompts.csv",           # may not exist (e.g., test)
        base + "prompts_basic.csv",
        base + "prompts_enhanced.csv",
        base + "prompts_advanced.csv",
        base + "prompts_spatial.csv",
    ]
    dfs = []
    for f in files_in_order:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                # Normalize columns if needed
                if 'poster_path' in df.columns and 'text_prompt' in df.columns:
                    dfs.append(df[['poster_path', 'text_prompt']])
            except Exception as e:
                print(f"  ❌ Failed reading {f}: {e}")
        else:
            print(f"  ⚠️  Skipping missing: {f}")

    if not dfs:
        print(f"  ❌ No prompt files found to merge for {dataset} {split}")
        return False

    merged = pd.concat(dfs, ignore_index=True)
    # Optional: sort by poster_path to group prompts together
    merged.sort_values(by=['poster_path'], inplace=True, kind='stable')

    out_path = f"dataset/{dataset}/split/csv/{split}_with_all_prompts.csv"
    # Write with full quoting
    with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(['poster_path', 'text_prompt'])
        for _, row in merged.iterrows():
            writer.writerow([row['poster_path'], row['text_prompt']])

    print(f"  ✅ Merged prompts saved to {out_path} ({len(merged)} rows)")
    return True

def main():
    """Main function to generate all prompts and fix CSV formatting"""

    print("🚀 COMPREHENSIVE PROMPT GENERATION SCRIPT")
    print("=" * 60)
    print("This script will:")
    print("1. Generate basic, enhanced, and advanced prompts for all splits")
    print("2. Generate rich prompts with augmentation")
    print("3. Fix CSV quote formatting")
    print("=" * 60)

    # Define all required combinations
    datasets = ['pku', 'cgl']
    splits = ['train', 'val', 'test']
    styles = ['basic', 'enhanced', 'advanced', 'spatial']

    # Step 1: Generate all standard prompts
    print("\n📝 STEP 1: Generating standard prompts (basic, enhanced, advanced)")
    print("-" * 60)

    total_tasks = len(datasets) * len(splits) * len(styles)
    completed_tasks = 0

    for dataset in datasets:
        print(f"\n📂 Processing {dataset.upper()} dataset...")

        for split in splits:
            print(f"\n🔄 Processing {split} split...")

            for style in styles:
                input_file = f"dataset/{dataset}/split/csv/{split}.csv"
                output_file = f"dataset/{dataset}/split/csv/{split}_with_prompts_{style}.csv"

                print(f"  📝 Generating {style} prompts...")
                try:
                    if create_text_prompts_from_csv(input_file, output_file, dataset_name=dataset, prompt_style=style):
                        completed_tasks += 1
                        print(f"  ✅ {style} prompts for {dataset} {split} completed!")
                    else:
                        print(f"  ❌ Failed to generate {style} prompts for {dataset} {split}")
                except Exception as e:
                    print(f"  ❌ Error generating {style} prompts for {dataset} {split}: {e}")

    print(f"\n📊 Standard prompts: {completed_tasks}/{total_tasks} completed")

    # Step 2: Generate rich prompts
    print("\n🎨 STEP 2: Generating rich prompts with augmentation")
    print("-" * 60)

    rich_tasks = 0
    rich_completed = 0

    for dataset in datasets:
        for split in ['train', 'val']:  # Only train and val for rich prompts
            input_file = f"dataset/{dataset}/split/csv/{split}.csv"
            output_file = f"dataset/{dataset}/split/csv/{split}_with_rich_prompts.csv"

            print(f"\n🎨 Generating rich prompts for {dataset} {split}...")
            try:
                if create_rich_text_prompts(input_file, output_file, dataset_name=dataset):
                    rich_completed += 1
                    print(f"  ✅ Rich prompts for {dataset} {split} completed!")
                else:
                    print(f"  ❌ Failed to generate rich prompts for {dataset} {split}")
            except Exception as e:
                print(f"  ❌ Error generating rich prompts for {dataset} {split}: {e}")
            rich_tasks += 1

    print(f"\n📊 Rich prompts: {rich_completed}/{rich_tasks} completed")

    # Step 3: Fix CSV quotes
    print("\n🔧 STEP 3: Fixing CSV quote formatting")
    print("-" * 60)

    # Files to fix
    files_to_fix = []
    for dataset in datasets:
        for split in ['train', 'val']:
            files_to_fix.extend([
                f"dataset/{dataset}/split/csv/{split}_with_rich_prompts.csv",
                f"dataset/{dataset}/split/csv/{split}_with_prompts_basic.csv",
                f"dataset/{dataset}/split/csv/{split}_with_prompts_enhanced.csv",
                f"dataset/{dataset}/split/csv/{split}_with_prompts_advanced.csv"
            ])

    quote_fixed = 0
    quote_total = len(files_to_fix)

    for file_path in files_to_fix:
        if os.path.exists(file_path):
            print(f"\n🔧 Fixing quotes in: {file_path}")
            if fix_csv_quotes(file_path):
                quote_fixed += 1
                print(f"  ✅ Successfully fixed!")
            else:
                print(f"  ❌ Failed to fix!")
        else:
            print(f"  ⚠️  File not found: {file_path}")

    print(f"\n📊 CSV quotes: {quote_fixed}/{quote_total} files fixed")

    # Step 4: Merge prompts per split
    print("\n🔗 STEP 4: Merging prompts per split into *_with_all_prompts.csv")
    print("-" * 60)
    merge_total = 0
    merge_ok = 0
    for dataset in datasets:
        for split in ['train', 'val', 'test']:
            merge_total += 1
            print(f"\n🔗 Merging {dataset} {split}...")
            if merge_prompts_for_split(dataset, split):
                merge_ok += 1
            else:
                print(f"  ❌ Merge failed for {dataset} {split}")

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 COMPREHENSIVE PROMPT GENERATION COMPLETE!")
    print("=" * 60)
    print(f"📊 Summary:")
    print(f"  - Standard prompts: {completed_tasks}/{total_tasks}")
    print(f"  - Rich prompts: {rich_completed}/{rich_tasks}")
    print(f"  - CSV quotes fixed: {quote_fixed}/{quote_total}")
    print(f"  - Merges: {merge_ok}/{merge_total}")

    if completed_tasks == total_tasks and rich_completed == rich_tasks and quote_fixed == quote_total:
        print("\n🎉 ALL TASKS COMPLETED SUCCESSFULLY!")
        print("\nGenerated files:")
        for dataset in datasets:
            print(f"\n{dataset.upper()} dataset:")
            for split in splits:
                print(f"  {split}:")
                for style in styles:
                    print(f"    - {split}_with_prompts_{style}.csv")
                if split in ['train', 'val']:
                    print(f"    - {split}_with_rich_prompts.csv")
                print(f"    - {split}_with_all_prompts.csv")
    else:
        print("\n⚠️  Some tasks failed. Check the error messages above.")

    print("\nPrompt Style Examples:")
    print("Basic: 'A layout with 2 Texts, 1 Logo.'")
    print("Enhanced: 'A layout with 2 Texts in the middle-center, 1 Logo in the top-left.'")
    print("Advanced: 'A layout with 1 Logo in the top-center, 2 Texts distributed across the layout.'")
    print("Spatial: 'A layout with 2 Texts at top-center, 1 Logo at bottom-right.'")
    print("Rich: 'Generate a design that includes 2 text boxes in the top-center, a single logo in the bottom-left.'")

if __name__ == "__main__":
    main()
