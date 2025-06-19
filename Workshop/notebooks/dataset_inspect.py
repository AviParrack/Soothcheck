# %%
import sys
import os
from pathlib import Path
from typing import List

if not "notebooks" in os.listdir():
    # Get the current directory of the notebook
    notebook_dir = Path(os.getcwd())
    # Assume the project root is one level up from the notebook directory
    project_root = notebook_dir.parent
    # Change the current working directory to the project root
    if Path(os.getcwd()).resolve() != project_root.resolve():
        os.chdir(project_root)
        print(f"Changed CWD to project root: {project_root}")

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

from src.config_models.data_config import DatasetConfig
from src.utils.utils import long_string_display, full_string_display

# Update these variables for the config and build you want to inspect
DATA_CONFIG_NAME = "general"  # Config name (e.g., "general")
RAW_DATASET_NAME = "alexk"  # Raw dataset name (e.g., "adam")
BUILD_NAME = "sft"  # Build name (e.g., "sft", "dpo")

# %%

dataset_config = DatasetConfig.load(DATA_CONFIG_NAME)
processed_path = dataset_config.get_processed_data_path(RAW_DATASET_NAME, BUILD_NAME)

dataset = load_from_disk(processed_path)
print(f"Dataset loaded from: {processed_path}")
print(f"Splits: {list(dataset.keys())}")

# Load tokenizer for this build
tokenizer_name = dataset_config.get_effective_tokenizer_name_or_path(BUILD_NAME)
print(f"Loading tokenizer for build '{BUILD_NAME}': {tokenizer_name}")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Get build-specific settings
prompt_format = dataset_config.get_effective_prompt_format(BUILD_NAME)
print(f"Prompt format for build '{BUILD_NAME}': {prompt_format}")

if "train" in dataset:
    trainset = dataset["train"]
    print(f"Train split found. Number of items: {len(trainset)}")

    # --- Summary Statistics ---
    print("\n--- Dataset Summary Statistics ---")
    print(f"Total samples in train set: {len(trainset)}")
    if "pipeline_name" in trainset.column_names:
        unique_pipeline_names_summary = np.unique(trainset["pipeline_name"])
        print("Samples per pipeline type:")
        for p_name in unique_pipeline_names_summary:
            count = len(
                trainset.filter(lambda example: example["pipeline_name"] == p_name)
            )
            print(f"  - {p_name}: {count} samples")
    else:
        print("'pipeline_name' column not found, cannot provide per-pipeline counts.")
    print("--- End Summary Statistics ---")
    # ---------------------------

    # Display examples from each pipeline type
    if len(trainset) > 0 and "pipeline_name" in trainset.column_names:
        unique_pipelines = np.unique(trainset["pipeline_name"])
        print(f"\nFound pipeline types: {unique_pipelines}")

        for pipeline_name_to_show in unique_pipelines:
            print(f"\n--- Examples from pipeline: '{pipeline_name_to_show}' ---")
            pipeline_subset = trainset.filter(
                lambda example: example["pipeline_name"] == pipeline_name_to_show
            )

            num_items_to_show_per_pipeline = 2
            actual_num_to_show = min(
                num_items_to_show_per_pipeline, len(pipeline_subset)
            )

            if len(pipeline_subset) > 0:
                print(
                    f"Displaying {actual_num_to_show} random example(s) from this pipeline (total: {len(pipeline_subset)}):"
                )

                indices_to_show = []
                if len(pipeline_subset) <= actual_num_to_show:
                    indices_to_show = list(range(len(pipeline_subset)))
                else:
                    indices_to_show = np.random.choice(
                        len(pipeline_subset), actual_num_to_show, replace=False
                    )

                for i, item_idx in enumerate(indices_to_show):
                    item = pipeline_subset[
                        int(item_idx)
                    ]  # Ensure item_idx is int for indexing
                    print(
                        f"\nExample {i+1} (Original Index in subset: {item_idx}, Pipeline: {item['pipeline_name']}):"
                    )

                    if prompt_format == "text" and "text" in item:
                        # For text, display directly using full_string_display
                        # No additional content indent needed here as full_string_display handles its own structure
                        print(
                            full_string_display(
                                item["text"], col_width=100, content_indent_str="  "
                            )
                        )  # Indent content by 2 spaces
                    elif prompt_format == "chat" and "messages" in item:
                        print("  ====CHAT MESSAGES BEGIN====")
                        for msg_idx, msg in enumerate(item["messages"]):
                            role = msg.get("role", "N/A")
                            content = msg.get("content", "")
                            print(f"    Message {msg_idx + 1} - Role: {role}")
                            # Use full_string_display for the content of each message, with a deeper indent
                            print(
                                full_string_display(
                                    content, col_width=90, content_indent_str="      "
                                )
                            )  # Indent message content by 6 spaces
                        print("  ====CHAT MESSAGES END====")
                    else:
                        print(
                            f"    Cannot display content. Item keys: {list(item.keys())}, Expected format: {prompt_format}"
                        )
                        # Optionally print raw item for debugging if format is unexpected
                        # print(f"    Raw item: {item}")

            else:
                print(
                    f"No items found for pipeline '{pipeline_name_to_show}' in the train set (this is unexpected if it was listed as unique)."
                )
    elif "pipeline_name" not in trainset.column_names:
        print(
            "ERROR: 'pipeline_name' column not found in the train set. Cannot display examples by pipeline."
        )
    else:  # trainset is empty
        print("Train set is empty.")

    # Tokenize and plot histogram of token lengths
    # Determine the correct column for token length calculation
    token_length_column = None
    if prompt_format == "text" and "text" in trainset.column_names:
        token_length_column = "text"
    elif prompt_format == "chat" and "messages" in trainset.column_names:
        # For chat, we might want to tokenize concatenated content or just the assistant's part
        # For simplicity in histogram, let's concatenate all message contents for token length.
        print(
            "\nTokenizing chat messages by concatenating content for length analysis."
        )
        # No direct column, need a custom function
    else:
        print(
            "\nCould not determine column for token length analysis based on prompt_format and available columns."
        )

    if len(trainset) > 0:
        if token_length_column:  # Text format
            print(
                f"\nTokenizing text from column: '{token_length_column}' for histogram."
            )

            def get_token_length(example):
                return {
                    "token_length": len(tokenizer.encode(example[token_length_column]))
                }

            tokenized_trainset = trainset.map(get_token_length, num_proc=os.cpu_count())
            token_lengths = tokenized_trainset["token_length"]

        elif (
            prompt_format == "chat" and "messages" in trainset.column_names
        ):  # Chat format

            def get_chat_token_length(example):
                full_chat_text = ""
                for msg in example["messages"]:
                    full_chat_text += (
                        msg.get("content", "") + tokenizer.eos_token
                    )  # Add eos as separator
                return {"token_length": len(tokenizer.encode(full_chat_text))}

            tokenized_trainset = trainset.map(
                get_chat_token_length, num_proc=os.cpu_count()
            )
            token_lengths = tokenized_trainset["token_length"]
        else:
            token_lengths = None  # Cannot compute

        if token_lengths is not None and len(token_lengths) > 0:
            plt.figure(figsize=(10, 6))

            # For stacked histogram, prepare data per pipeline
            if "pipeline_name" in tokenized_trainset.column_names:
                unique_pipelines_hist = np.unique(tokenized_trainset["pipeline_name"])
                data_for_stacked_hist = []
                colors_for_stacked_hist = plt.cm.get_cmap(
                    "viridis", len(unique_pipelines_hist)
                ).colors
                labels_for_stacked_hist = []

                for i, p_name in enumerate(unique_pipelines_hist):
                    pipeline_token_lengths = tokenized_trainset.filter(
                        lambda ex: ex["pipeline_name"] == p_name
                    )["token_length"]
                    if len(pipeline_token_lengths) > 0:
                        data_for_stacked_hist.append(pipeline_token_lengths)
                        labels_for_stacked_hist.append(p_name)

                if data_for_stacked_hist:  # Ensure we have data to plot
                    plt.hist(
                        data_for_stacked_hist,
                        bins=50,
                        color=colors_for_stacked_hist[: len(data_for_stacked_hist)],
                        alpha=0.7,
                        stacked=True,
                        label=labels_for_stacked_hist,
                    )
                    plt.legend(title="Pipeline Types")
                else:
                    # Fallback to simple histogram if something went wrong with per-pipeline data prep
                    plt.hist(token_lengths, bins=50, color="blue", alpha=0.7)
            else:
                # Fallback if pipeline_name column is somehow missing after tokenization
                plt.hist(token_lengths, bins=50, color="blue", alpha=0.7)

            plt.title(
                f"Histogram of Token Lengths in Train Set ({DATA_CONFIG_NAME}:{RAW_DATASET_NAME}:{BUILD_NAME})"
            )
            plt.xlabel("Token Length")
            plt.ylabel("Number of Items")
            plt.grid(axis="y", alpha=0.75)
            # Add some descriptive statistics to the plot
            mean_len = np.mean(token_lengths)
            median_len = np.median(token_lengths)
            std_len = np.std(token_lengths)
            plt.axvline(
                mean_len,
                color="red",
                linestyle="dashed",
                linewidth=1,
                label=f"Mean: {mean_len:.2f}",
            )
            plt.axvline(
                median_len,
                color="green",
                linestyle="dashed",
                linewidth=1,
                label=f"Median: {median_len:.2f}",
            )
            plt.legend()

            print(f"\nToken length statistics (train set):")
            print(f"  Mean: {mean_len:.2f}")
            print(f"  Median: {median_len:.2f}")
            print(f"  Std Dev: {std_len:.2f}")
            print(f"  Min: {np.min(token_lengths)}")
            print(f"  Max: {np.max(token_lengths)}")

            # Determine a suitable filename for the plot
            plot_filename = f"token_lengths_histogram_{DATA_CONFIG_NAME}_{RAW_DATASET_NAME}_{BUILD_NAME}.png"
            # Ensure notebooks/outputs directory exists
            output_dir = Path("notebooks/outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = output_dir / plot_filename

            try:
                plt.savefig(plot_path)
                print(f"Histogram saved to: {plot_path}")
            except Exception as e:
                print(f"Error saving plot: {e}")
            plt.show()

        elif (
            token_length_column and token_length_column not in trainset.column_names
        ):  # This condition might be redundant due to earlier checks
            print(
                f"ERROR: Column '{token_length_column}' for token length calculation not found in the train set."
            )
        # else: trainset is empty, already handled for example display

    # Create pie charts for augmentation distribution
    if len(trainset) > 0 and token_lengths is not None:
        print("\n--- Augmentation Distribution Analysis ---")

        # Determine augmentation source for each sample
        augmentation_names = []
        augmentation_token_counts = {}
        augmentation_sample_counts = {}

        for i, item in enumerate(trainset):
            # Get augmentation name - default to "Raw Corpus" if not present or empty
            aug_name = item.get("augmentation_name", None)
            if not aug_name or aug_name == "unknown" or str(aug_name).strip() == "":
                aug_name = "Raw Corpus"

            augmentation_names.append(aug_name)

            # Count samples
            if aug_name not in augmentation_sample_counts:
                augmentation_sample_counts[aug_name] = 0
            augmentation_sample_counts[aug_name] += 1

            # Count tokens (if we have token lengths)
            if i < len(token_lengths):
                if aug_name not in augmentation_token_counts:
                    augmentation_token_counts[aug_name] = 0
                augmentation_token_counts[aug_name] += token_lengths[i]

        # Create pie charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Sample count pie chart
        sample_labels = list(augmentation_sample_counts.keys())
        sample_counts = list(augmentation_sample_counts.values())

        # Use a colormap for consistent colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(sample_labels)))

        wedges1, texts1, autotexts1 = ax1.pie(
            sample_counts,
            labels=sample_labels,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        ax1.set_title(
            f"Training Samples by Source\n({DATA_CONFIG_NAME}:{RAW_DATASET_NAME}:{BUILD_NAME})",
            fontsize=12,
            fontweight="bold",
        )

        # Make percentage text more readable
        for autotext in autotexts1:
            autotext.set_color("black")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(10)

        # Token count pie chart
        if augmentation_token_counts:
            token_labels = list(augmentation_token_counts.keys())
            token_counts = list(augmentation_token_counts.values())

            wedges2, texts2, autotexts2 = ax2.pie(
                token_counts,
                labels=token_labels,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
            )
            ax2.set_title(
                f"Total Tokens by Source\n({DATA_CONFIG_NAME}:{RAW_DATASET_NAME}:{BUILD_NAME})",
                fontsize=12,
                fontweight="bold",
            )

            # Make percentage text more readable
            for autotext in autotexts2:
                autotext.set_color("black")
                autotext.set_fontweight("bold")
                autotext.set_fontsize(10)
        else:
            ax2.text(
                0.5,
                0.5,
                "Token data not available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax2.transAxes,
                fontsize=14,
            )
            ax2.set_title("Token Distribution (No Data)")

        plt.tight_layout()

        # Save the pie charts
        pie_filename = f"augmentation_distribution_{DATA_CONFIG_NAME}_{RAW_DATASET_NAME}_{BUILD_NAME}.png"
        pie_path = output_dir / pie_filename

        try:
            plt.savefig(pie_path, dpi=300, bbox_inches="tight")
            print(f"Augmentation distribution charts saved to: {pie_path}")
        except Exception as e:
            print(f"Error saving pie charts: {e}")

        plt.show()

        # Print detailed statistics
        print("\nDetailed augmentation statistics:")
        print("Sample counts:")
        total_samples = sum(sample_counts)
        for label, count in zip(sample_labels, sample_counts):
            percentage = (count / total_samples) * 100
            print(f"  - {label}: {count:,} samples ({percentage:.1f}%)")

        if augmentation_token_counts:
            print("\nToken counts:")
            total_tokens = sum(token_counts)
            for label, count in zip(token_labels, token_counts):
                percentage = (count / total_tokens) * 100
                avg_tokens_per_sample = count / augmentation_sample_counts.get(label, 1)
                print(
                    f"  - {label}: {count:,} tokens ({percentage:.1f}%, {avg_tokens_per_sample:.0f} avg/sample)"
                )

        print("--- End Augmentation Distribution Analysis ---")

else:
    print("ERROR: 'train' split not found in the loaded dataset.")

# %%
