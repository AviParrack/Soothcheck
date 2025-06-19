# Example usage: python -m scripts.datalook
# This script provides a Gradio web interface for browsing processed datasets.
# It allows you to:
# - Select from available dataset configurations
# - Browse through examples with pagination
# - Filter by pipeline type
# - View both text and chat format data in a nice table format
# - View DPO datasets with prompt/accepted/rejected responses
# - See token length statistics and histograms

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json

# Ensure we're in the project root
if not "notebooks" in os.listdir():
    notebook_dir = Path(os.getcwd())
    project_root = notebook_dir.parent
    if Path(os.getcwd()).resolve() != project_root.resolve():
        os.chdir(project_root)
        print(f"Changed CWD to project root: {project_root}")

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import AutoTokenizer
import plotly.express as px
import plotly.graph_objects as go

from src.config_models.data_config import DatasetConfig
from src.utils.utils import full_string_display


class DatasetBrowser:
    def __init__(self):
        self.current_config = None
        self.current_dataset = None
        self.current_split_data = None
        self.current_tokenizer = None
        self.current_build_name = None  # Store current build name
        self.current_dataset_type = None  # Store dataset type: 'sft', 'dpo'
        self.available_configs = self._get_available_configs()

    def _get_available_configs(self) -> List[str]:
        """Get list of available dataset configurations."""
        config_dir = Path("configs/data")
        if not config_dir.exists():
            return []

        configs = []
        for config_file in config_dir.glob("*.json"):
            configs.append(config_file.stem)
        return sorted(configs)

    def _detect_dataset_type(self, dataset) -> str:
        """Detect if this is an SFT or DPO dataset based on columns."""
        if not dataset:
            return "unknown"

        # Get column names from the first split
        first_split = next(iter(dataset.values()))
        columns = set(first_split.column_names)

        # DPO datasets have these specific columns
        dpo_columns = {"prompt", "response_accepted", "response_rejected"}
        if dpo_columns.issubset(columns):
            return "dpo"

        # SFT datasets have either text or messages
        if "text" in columns or "messages" in columns:
            return "sft"

        return "unknown"

    def load_dataset(
        self, config_name: str, raw_dataset_name: str, build_name: str
    ) -> Tuple[str, Dict, Dict]:
        """Load dataset with the specified config, raw dataset name, and build."""
        try:
            # Load configuration
            self.current_config = DatasetConfig.load(config_name)
            self.current_build_name = build_name  # Store for later use

            # Get processed dataset path for this build
            processed_path = self.current_config.get_processed_data_path(
                raw_dataset_name, build_name
            )

            # Check if path exists
            if not os.path.exists(processed_path):
                return (
                    f"Dataset not found at {processed_path}. Make sure you've run dataset preparation first.",
                    {},
                )

            # Load dataset
            self.current_dataset = load_from_disk(processed_path)

            # Detect dataset type
            self.current_dataset_type = self._detect_dataset_type(self.current_dataset)

            # Load tokenizer for this build
            tokenizer_name = self.current_config.get_effective_tokenizer_name_or_path(
                build_name
            )
            self.current_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            # Get dataset info - ensure all keys are strings
            dataset_info = {
                "config_name": str(config_name),
                "raw_dataset_name": str(raw_dataset_name),
                "build_name": str(build_name),
                "dataset_type": str(self.current_dataset_type),
                "tokenizer": str(tokenizer_name),
            }

            # Add format info based on dataset type
            if self.current_dataset_type == "dpo":
                dataset_info["format"] = "DPO (prompt/accepted/rejected)"
            else:
                dataset_info["prompt_format"] = str(
                    self.current_config.get_effective_prompt_format(build_name)
                )
                dataset_info["max_length"] = int(
                    self.current_config.get_effective_max_length(build_name)
                )

            dataset_info.update(
                {
                    "splits": [str(s) for s in self.current_dataset.keys()],
                    "total_samples": int(
                        sum(len(split) for split in self.current_dataset.values())
                    ),
                }
            )

            # Get split info - ensure all keys are strings
            split_info = {}
            for split_name, split_data in self.current_dataset.items():
                pipeline_counts = {}
                if "pipeline_name" in split_data.column_names:
                    unique_pipelines = np.unique(split_data["pipeline_name"])
                    for pipeline in unique_pipelines:
                        count = len(
                            split_data.filter(lambda x: x["pipeline_name"] == pipeline)
                        )
                        pipeline_counts[str(pipeline)] = int(count)

                split_info[str(split_name)] = {
                    "samples": int(len(split_data)),
                    "columns": [str(col) for col in split_data.column_names],
                    "pipelines": pipeline_counts,
                }

            return (
                f"Successfully loaded dataset: {config_name} | {raw_dataset_name} | {build_name} | {self.current_dataset_type.upper()}",
                dataset_info,
                split_info,
            )

        except Exception as e:
            return f"Error loading dataset: {str(e)}", {}, {}

    def get_split_data(
        self, split_name: str
    ) -> Tuple[List[str], List[str], List[str], int]:
        """Get data for a specific split and return available pipelines, source files, augmentation names, and total count."""
        if not self.current_dataset or split_name not in self.current_dataset:
            return [], [], [], 0

        self.current_split_data = self.current_dataset[split_name]

        pipelines = ["All"]
        if "pipeline_name" in self.current_split_data.column_names:
            unique_pipelines = np.unique(self.current_split_data["pipeline_name"])
            pipelines.extend(sorted(unique_pipelines))

        source_files = ["All"]
        if "source" in self.current_split_data.column_names:
            # Extract filenames from source paths
            unique_sources = set()
            for source in self.current_split_data["source"]:
                if " (line " in source:
                    # Remove " (line X)" suffix
                    file_path = source.split(" (line ")[0]
                else:
                    file_path = source
                filename = file_path.split("/")[-1]  # Get just the filename
                unique_sources.add(filename)
            source_files.extend(sorted(unique_sources))

        augmentations = ["All"]
        if "augmentation_name" in self.current_split_data.column_names:
            # Extract unique augmentation names, handling missing values
            unique_augmentations = set()
            for aug in self.current_split_data["augmentation_name"]:
                if aug and aug != "unknown":  # Filter out empty and "unknown" values
                    unique_augmentations.add(aug)
            if unique_augmentations:  # Only add if we found any
                augmentations.extend(sorted(unique_augmentations))

        return pipelines, source_files, augmentations, len(self.current_split_data)

    def get_examples(
        self,
        split_name: str,
        pipeline_filter: str,
        augmentation_filter: str,
        page: int,
        page_size: int = 10,
    ) -> Tuple[str, str]:
        """Get examples for display in full format."""
        if not self.current_split_data:
            return "No data loaded", "No data loaded"

        # Filter by pipeline if specified
        filtered_data = self.current_split_data
        if (
            pipeline_filter != "All"
            and "pipeline_name" in self.current_split_data.column_names
        ):
            filtered_data = filtered_data.filter(
                lambda x: x["pipeline_name"] == pipeline_filter
            )

        # Filter by augmentation if specified
        if (
            augmentation_filter != "All"
            and "augmentation_name" in self.current_split_data.column_names
        ):
            filtered_data = filtered_data.filter(
                lambda x: x.get("augmentation_name", "unknown") == augmentation_filter
            )

        total_items = len(filtered_data)
        if total_items == 0:
            return "No examples found for the selected filters", "No examples found"

        # Calculate pagination
        start_idx = page * page_size
        end_idx = min(start_idx + page_size, total_items)

        if start_idx >= total_items:
            return (
                f"Page {page + 1} is beyond available data",
                f"Page {page + 1} is beyond available data",
            )

        # Build the display content
        content_parts = []

        for i in range(start_idx, end_idx):
            item = filtered_data[i]
            actual_index = start_idx + (i - start_idx)
            pipeline = item.get("pipeline_name", "N/A")
            augmentation = item.get("augmentation_name", "N/A")

            # Add separator and header
            content_parts.append("=" * 80)
            content_parts.append(
                f"**Example {actual_index + 1} | Pipeline: {pipeline} | Augmentation: {augmentation}**"
            )

            # Calculate and show token length
            if self.current_tokenizer:
                token_length = self._calculate_token_length(item)
                content_parts.append(f"**Token Length: {token_length}**")

            content_parts.append("")  # Empty line

            # Format content based on dataset type
            if self.current_dataset_type == "dpo":
                self._format_dpo_content(item, content_parts)
            else:
                self._format_sft_content(item, content_parts)

            content_parts.append("")  # Empty line after each example

        full_content = "\n".join(content_parts)
        status = f"Showing examples {start_idx + 1}-{end_idx} of {total_items} (Page {page + 1})"

        return full_content, status

    def _calculate_token_length(self, item: Dict[str, Any]) -> str:
        """Calculate token length for an item based on dataset type."""
        if not self.current_tokenizer:
            return "N/A"

        try:
            if self.current_dataset_type == "dpo":
                # For DPO, calculate tokens for prompt + both responses
                prompt_tokens = len(
                    self.current_tokenizer.encode(item.get("prompt", ""))
                )
                accepted_tokens = len(
                    self.current_tokenizer.encode(item.get("response_accepted", ""))
                )
                rejected_tokens = len(
                    self.current_tokenizer.encode(item.get("response_rejected", ""))
                )
                return f"Prompt: {prompt_tokens}, Accepted: {accepted_tokens}, Rejected: {rejected_tokens}"
            else:
                # For SFT datasets
                prompt_format = self.current_config.get_effective_prompt_format(
                    self.current_build_name
                )
                if prompt_format == "text" and "text" in item:
                    token_length = len(self.current_tokenizer.encode(item["text"]))
                elif prompt_format == "chat" and "messages" in item:
                    total_content = ""
                    for msg in item["messages"]:
                        total_content += (
                            msg.get("content", "") + self.current_tokenizer.eos_token
                        )
                    token_length = len(self.current_tokenizer.encode(total_content))
                else:
                    return "N/A"
                return str(token_length)
        except Exception as e:
            return f"Error: {str(e)}"

    def _format_dpo_content(self, item: Dict[str, Any], content_parts: List[str]):
        """Format DPO dataset content for display."""
        prompt = item.get("prompt", "N/A")

        content_parts.append("**🎯 PROMPT:**")
        content_parts.append("")
        if prompt == "" or prompt.strip() == "":
            content_parts.append("*(empty prompt)*")
        else:
            content_parts.append(prompt)
        content_parts.append("")

        content_parts.append("**✅ ACCEPTED RESPONSE:**")
        content_parts.append("")
        content_parts.append(item.get("response_accepted", "N/A"))
        content_parts.append("")

        content_parts.append("**❌ REJECTED RESPONSE:**")
        content_parts.append("")
        content_parts.append(item.get("response_rejected", "N/A"))
        content_parts.append("")

    def _format_sft_content(self, item: Dict[str, Any], content_parts: List[str]):
        """Format SFT dataset content for display."""
        prompt_format = self.current_config.get_effective_prompt_format(
            self.current_build_name
        )
        if prompt_format == "text" and "text" in item:
            content_parts.append("**Content:**")
            content_parts.append("")
            content_parts.append(item["text"])

        elif prompt_format == "chat" and "messages" in item:
            content_parts.append("**Conversation:**")
            content_parts.append("")
            for msg_idx, msg in enumerate(item["messages"]):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                # Use different styling for different roles
                if role == "user":
                    content_parts.append(f"🧑 **USER:**")
                elif role == "assistant":
                    content_parts.append(f"🤖 **ASSISTANT:**")
                elif role == "system":
                    content_parts.append(f"⚙️ **SYSTEM:**")
                else:
                    content_parts.append(f"❓ **{role.upper()}:**")

                content_parts.append("")
                content_parts.append(content)
                content_parts.append("")  # Empty line between messages

    def get_token_length_plot(
        self, split_name: str, pipeline_filter: str, augmentation_filter: str
    ):
        """Generate token length histogram."""
        if not self.current_split_data or not self.current_tokenizer:
            return None

        # Filter by pipeline if specified
        filtered_data = self.current_split_data
        if (
            pipeline_filter != "All"
            and "pipeline_name" in self.current_split_data.column_names
        ):
            filtered_data = filtered_data.filter(
                lambda x: x["pipeline_name"] == pipeline_filter
            )

        # Filter by augmentation if specified
        if (
            augmentation_filter != "All"
            and "augmentation_name" in self.current_split_data.column_names
        ):
            filtered_data = filtered_data.filter(
                lambda x: x.get("augmentation_name", "unknown") == augmentation_filter
            )

        if len(filtered_data) == 0:
            return None

        # Calculate token lengths based on dataset type
        if self.current_dataset_type == "dpo":
            return self._generate_dpo_token_plot(
                filtered_data, split_name, pipeline_filter, augmentation_filter
            )
        else:
            return self._generate_sft_token_plot(
                filtered_data, split_name, pipeline_filter, augmentation_filter
            )

    def _generate_dpo_token_plot(
        self,
        filtered_data,
        split_name: str,
        pipeline_filter: str,
        augmentation_filter: str,
    ):
        """Generate token length plot for DPO datasets."""
        prompt_lengths = []
        accepted_lengths = []
        rejected_lengths = []

        for item in filtered_data:
            try:
                prompt_len = len(self.current_tokenizer.encode(item.get("prompt", "")))
                accepted_len = len(
                    self.current_tokenizer.encode(item.get("response_accepted", ""))
                )
                rejected_len = len(
                    self.current_tokenizer.encode(item.get("response_rejected", ""))
                )

                prompt_lengths.append(prompt_len)
                accepted_lengths.append(accepted_len)
                rejected_lengths.append(rejected_len)
            except Exception:
                continue

        if not prompt_lengths:
            return None

        # Create subplots for DPO components
        fig = go.Figure()

        fig.add_trace(
            go.Histogram(x=prompt_lengths, name="Prompt", opacity=0.7, nbinsx=30)
        )

        fig.add_trace(
            go.Histogram(
                x=accepted_lengths, name="Accepted Response", opacity=0.7, nbinsx=30
            )
        )

        fig.add_trace(
            go.Histogram(
                x=rejected_lengths, name="Rejected Response", opacity=0.7, nbinsx=30
            )
        )

        fig.update_layout(
            title=f"DPO Token Length Distribution - {split_name} ({pipeline_filter}, {augmentation_filter})",
            xaxis_title="Token Length",
            yaxis_title="Count",
            barmode="overlay",
        )

        return fig

    def _generate_sft_token_plot(
        self,
        filtered_data,
        split_name: str,
        pipeline_filter: str,
        augmentation_filter: str,
    ):
        """Generate token length plot for SFT datasets."""
        token_lengths = []
        prompt_format = self.current_config.get_effective_prompt_format(
            self.current_build_name
        )
        for item in filtered_data:
            try:
                if prompt_format == "text" and "text" in item:
                    content = item["text"]
                elif prompt_format == "chat" and "messages" in item:
                    content = ""
                    for msg in item["messages"]:
                        content += (
                            msg.get("content", "") + self.current_tokenizer.eos_token
                        )
                else:
                    continue

                token_length = len(self.current_tokenizer.encode(content))
                token_lengths.append(token_length)
            except Exception:
                continue

        if not token_lengths:
            return None

        # Create plotly histogram
        fig = px.histogram(
            x=token_lengths,
            nbins=50,
            title=f"Token Length Distribution - {split_name} ({pipeline_filter}, {augmentation_filter})",
            labels={"x": "Token Length", "y": "Count"},
        )

        # Add statistics
        mean_len = np.mean(token_lengths)
        median_len = np.median(token_lengths)

        fig.add_vline(
            x=mean_len,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_len:.1f}",
        )
        fig.add_vline(
            x=median_len,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Median: {median_len:.1f}",
        )

        return fig


def create_interface():
    browser = DatasetBrowser()

    with gr.Blocks(title="Dataset Browser", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Dataset Browser")
        gr.Markdown("Browse through processed datasets with pagination and filtering.")

        with gr.Row():
            with gr.Column(scale=1):
                config_dropdown = gr.Dropdown(
                    choices=browser.available_configs,
                    label="Select Dataset Configuration",
                    value=(
                        browser.available_configs[0]
                        if browser.available_configs
                        else None
                    ),
                )
                raw_dataset_input = gr.Textbox(
                    label="Raw Dataset Name",
                    placeholder="e.g., adam, alice, bob",
                    value="adam",
                )
                build_dropdown = gr.Dropdown(
                    choices=["sft", "dpo"], label="Build Name", value="sft"
                )
                load_btn = gr.Button("Load Dataset", variant="primary")

                status_text = gr.Textbox(label="Status", interactive=False, lines=2)

            with gr.Column(scale=2):
                dataset_info = gr.JSON(label="Dataset Information")
                split_info = gr.JSON(label="Split Information")

        gr.Markdown("## Browse Examples")

        with gr.Row():
            split_dropdown = gr.Dropdown(label="Split", choices=[], interactive=True)
            pipeline_dropdown = gr.Dropdown(
                label="Pipeline Filter", choices=["All"], value="All"
            )
            augmentation_dropdown = gr.Dropdown(
                label="Augmentation Filter", choices=["All"], value="All"
            )
            page_size_slider = gr.Slider(
                minimum=5, maximum=50, value=10, step=5, label="Page Size"
            )

        with gr.Row():
            prev_btn = gr.Button("Previous Page", size="sm")
            page_info = gr.Textbox(value="Page 1", interactive=False, container=False)
            next_btn = gr.Button("Next Page", size="sm")

        examples_display = gr.Markdown(
            label="Examples", value="Select a dataset and split to view examples"
        )

        table_status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("## Token Length Analysis")

        with gr.Row():
            plot_btn = gr.Button("Generate Token Length Plot")
            token_plot = gr.Plot(label="Token Length Distribution")

        # State variables
        current_page = gr.State(0)
        total_examples = gr.State(0)

        # Event handlers
        def load_dataset_handler(config_name, raw_dataset_name, build_name):
            status, dataset_info_dict, split_info_dict = browser.load_dataset(
                config_name, raw_dataset_name, build_name
            )
            splits = list(dataset_info_dict.get("splits", []))
            return (
                status,
                dataset_info_dict,
                split_info_dict,
                gr.update(choices=splits, value=splits[0] if splits else None),
            )

        def split_changed_handler(split_name):
            if split_name:
                pipelines, source_files, augmentations, total = browser.get_split_data(
                    split_name
                )
                return (
                    gr.update(choices=pipelines, value="All"),
                    gr.update(choices=augmentations, value="All"),
                    total,
                    0,
                )
            return (
                gr.update(choices=["All"], value="All"),
                gr.update(choices=["All"], value="All"),
                0,
                0,
            )

        def update_examples_handler(
            split_name, pipeline_filter, augmentation_filter, page, page_size
        ):
            if split_name:
                full_content, status = browser.get_examples(
                    split_name, pipeline_filter, augmentation_filter, page, page_size
                )
                return full_content, status, f"Page {page + 1}"
            return "No split selected", "No split selected", "Page 1"

        def prev_page_handler(current_page_val):
            return max(0, current_page_val - 1)

        def next_page_handler(current_page_val, total_examples_val, page_size):
            max_page = (total_examples_val - 1) // page_size
            return min(max_page, current_page_val + 1)

        def generate_plot_handler(split_name, pipeline_filter, augmentation_filter):
            if split_name:
                return browser.get_token_length_plot(
                    split_name, pipeline_filter, augmentation_filter
                )
            return None

        # Wire up events
        load_btn.click(
            load_dataset_handler,
            inputs=[config_dropdown, raw_dataset_input, build_dropdown],
            outputs=[status_text, dataset_info, split_info, split_dropdown],
        )

        split_dropdown.change(
            split_changed_handler,
            inputs=[split_dropdown],
            outputs=[
                pipeline_dropdown,
                augmentation_dropdown,
                total_examples,
                current_page,
            ],
        )

        # Update examples when any relevant input changes
        for component in [
            split_dropdown,
            pipeline_dropdown,
            augmentation_dropdown,
            current_page,
            page_size_slider,
        ]:
            component.change(
                update_examples_handler,
                inputs=[
                    split_dropdown,
                    pipeline_dropdown,
                    augmentation_dropdown,
                    current_page,
                    page_size_slider,
                ],
                outputs=[examples_display, table_status, page_info],
            )

        prev_btn.click(prev_page_handler, inputs=[current_page], outputs=[current_page])

        next_btn.click(
            next_page_handler,
            inputs=[current_page, total_examples, page_size_slider],
            outputs=[current_page],
        )

        plot_btn.click(
            generate_plot_handler,
            inputs=[split_dropdown, pipeline_dropdown, augmentation_dropdown],
            outputs=[token_plot],
        )

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True, server_name="0.0.0.0", server_port=7860)
