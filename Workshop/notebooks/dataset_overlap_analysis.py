# %%
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Counter
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np

if not "notebooks" in os.listdir():
    # Get the current directory of the notebook
    notebook_dir = Path(os.getcwd())
    # Assume the project root is one level up from the notebook directory
    project_root = notebook_dir.parent
    # Change the current working directory to the project root
    if Path(os.getcwd()).resolve() != project_root.resolve():
        os.chdir(project_root)
        print(f"Changed CWD to project root: {project_root}")

from transformers import AutoTokenizer
from src.config_models.data_config import DatasetConfig
from src.data.data_utils import (
    get_corpus_text_strings,
    get_all_augmented_text_strings,
    get_corpus_text_strings_with_paths,
    get_all_augmented_text_strings_with_paths,
)

# Update these variables for the config and build you want to analyze
DATA_CONFIG_NAME = "general"  # Config name (e.g., "general")
RAW_DATASET_NAME = "rudolf"  # Raw dataset name (e.g., "adam")
BUILD_NAME = "sft"  # Build name (e.g., "sft", "dpo")

# Load dataset configuration
dataset_config = DatasetConfig.load(DATA_CONFIG_NAME)

# Load tokenizer for this build
tokenizer_name = dataset_config.get_effective_tokenizer_name_or_path(BUILD_NAME)
print(f"Loading tokenizer for build '{BUILD_NAME}': {tokenizer_name}")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

print(f"Analyzing dataset: {DATA_CONFIG_NAME}:{RAW_DATASET_NAME}:{BUILD_NAME}")
print(f"Using tokenizer: {tokenizer_name}")

# %%


def generate_ngrams(text: str, n: int, tokenizer) -> List[str]:
    """Generate n-grams from text using the tokenizer."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram_tokens = tokens[i : i + n]
        ngram_text = tokenizer.decode(ngram_tokens, skip_special_tokens=True)
        ngrams.append(ngram_text)
    return ngrams


def count_ngrams_in_texts(
    texts: List[str], n: int, tokenizer, file_names: List[str] = None
) -> Tuple[Counter, Dict[str, Dict[str, int]]]:
    """Count n-grams in a list of texts and track which files they appear in with counts."""
    ngram_counts = Counter()
    ngram_files = defaultdict(lambda: defaultdict(int))

    for i, text in enumerate(texts):
        file_name = file_names[i] if file_names and i < len(file_names) else f"text_{i}"
        ngrams = generate_ngrams(text, n, tokenizer)
        for ngram in ngrams:
            ngram_counts[ngram] += 1
            ngram_files[ngram][file_name] += 1

    return ngram_counts, dict(ngram_files)


def analyze_overlap(corpus_ngrams: Counter, augmented_ngrams: Counter, n: int) -> Dict:
    """Analyze overlap between corpus and augmented data n-grams."""
    results = {}

    # Find overlapping n-grams
    overlapping_ngrams = set(corpus_ngrams.keys()) & set(augmented_ngrams.keys())

    # Calculate duplication statistics
    duplication_counts = defaultdict(
        int
    )  # How many times each corpus n-gram appears in augmented data

    for ngram in overlapping_ngrams:
        corpus_count = corpus_ngrams[ngram]
        augmented_count = augmented_ngrams[ngram]

        # Count how many times this corpus n-gram is duplicated in augmented data
        for _ in range(augmented_count):
            duplication_counts[corpus_count] += 1

    # Calculate percentages
    total_corpus_ngrams = sum(corpus_ngrams.values())
    total_augmented_ngrams = sum(augmented_ngrams.values())

    results["n"] = n
    results["total_corpus_ngrams"] = total_corpus_ngrams
    results["total_augmented_ngrams"] = total_augmented_ngrams
    results["overlapping_ngrams"] = len(overlapping_ngrams)
    results["overlap_in_corpus"] = sum(
        corpus_ngrams[ngram] for ngram in overlapping_ngrams
    )
    results["overlap_in_augmented"] = sum(
        augmented_ngrams[ngram] for ngram in overlapping_ngrams
    )

    # Percentage of corpus n-grams that appear in augmented data
    results["corpus_overlap_percentage"] = (
        (results["overlap_in_corpus"] / total_corpus_ngrams * 100)
        if total_corpus_ngrams > 0
        else 0
    )

    # Percentage of augmented n-grams that come from corpus
    results["augmented_from_corpus_percentage"] = (
        (results["overlap_in_augmented"] / total_augmented_ngrams * 100)
        if total_augmented_ngrams > 0
        else 0
    )

    # Duplication statistics
    results["duplication_counts"] = dict(duplication_counts)

    return results


def looks_like_url_or_link(text: str) -> bool:
    """Check if text looks like a URL, file path, or other link-like content."""
    # Convert to lowercase for some checks
    text_lower = text.lower()

    # Common URL patterns
    if "//" in text:
        return True
    if "%" in text and any(c in text for c in "0123456789abcdefABCDEF"):
        return True  # URL encoding
    if text_lower.startswith(("http:", "https:", "ftp:", "www.")):
        return True
    if any(
        ext in text_lower
        for ext in [".com", ".org", ".net", ".edu", ".gov", ".co.uk", ".io"]
    ):
        return True

    # File path patterns
    if text.count("/") >= 3:  # Multiple slashes suggest file paths
        return True
    if text.count("\\") >= 2:  # Windows paths
        return True

    # URL query/fragment patterns
    if "?" in text and "&" in text:
        return True
    if "#" in text and any(c in text for c in "=/&?"):
        return True

    # URL parameter patterns (like CDN parameters: q_auto:good,fl_progressive)
    # Look for parameter-like patterns with underscores, colons, and commas
    if ":" in text and "," in text:
        # Count parameter-like segments (things separated by commas that contain colons or underscores)
        segments = text.split(",")
        param_like_segments = 0
        for segment in segments:
            # Remove leading/trailing whitespace and punctuation for this check
            clean_segment = segment.strip(",.:/;")
            if (":" in segment or "_" in clean_segment) and len(clean_segment) > 2:
                # Check if it looks like a parameter (contains technical-looking patterns)
                if any(
                    pattern in clean_segment
                    for pattern in ["_auto", "fl_", "q_", "c_", "w_", "h_"]
                ):
                    param_like_segments += 1
                elif ":" in segment and not any(c.isspace() for c in segment):
                    # Colon with no spaces suggests key:value parameter
                    param_like_segments += 1

        if param_like_segments >= 2:  # Multiple parameter-like segments
            return True

    # Multiple underscores suggest technical/URL content
    if text.count("_") >= 3:
        return True

    # Patterns that look like image/CDN parameters
    cdnlike_patterns = ["_auto", "fl_", "q_auto", "c_", "w_", "h_", "f_auto", "dpr_"]
    if any(pattern in text_lower for pattern in cdnlike_patterns):
        return True

    # Dot patterns that suggest URLs/domains (but not normal sentences)
    # Look for dots between non-whitespace characters, but exclude common sentence patterns
    dot_indices = [i for i, c in enumerate(text) if c == "."]
    suspicious_dots = 0
    for dot_idx in dot_indices:
        # Check characters around the dot
        before_char = text[dot_idx - 1] if dot_idx > 0 else " "
        after_char = text[dot_idx + 1] if dot_idx < len(text) - 1 else " "

        # Skip if it looks like end of sentence (dot followed by space/end)
        if after_char in " \n\t" or dot_idx == len(text) - 1:
            continue

        # Skip if it looks like decimal number
        if before_char.isdigit() and after_char.isdigit():
            continue

        # Count suspicious dots (non-whitespace on both sides, not decimal)
        if before_char not in " \n\t" and after_char not in " \n\t":
            suspicious_dots += 1

    # If we have multiple suspicious dots, it's likely a URL/path
    if suspicious_dots >= 2:
        return True

    return False


def print_top_ngrams(
    ngram_counts: Counter,
    ngram_files: Dict[str, Dict[str, int]],
    title: str,
    top_k: int = 20,
):
    """Print top k most common n-grams with their counts and source files, filtering out URL-like content."""
    print(f"\n{title}")
    print("=" * len(title))

    printed_count = 0
    original_rank = 0

    for ngram, count in ngram_counts.most_common():
        original_rank += 1

        # Skip URL-like n-grams
        if looks_like_url_or_link(ngram):
            continue

        printed_count += 1
        if printed_count > top_k:
            break

        file_counts = ngram_files.get(ngram, {})
        # Sort files by count (descending), then by name
        sorted_files = sorted(file_counts.items(), key=lambda x: (-x[1], x[0]))

        files_with_counts = [f"{file}({count})" for file, count in sorted_files[:5]]
        files_str = ", ".join(files_with_counts)
        if len(sorted_files) > 5:
            remaining_count = sum(count for _, count in sorted_files[5:])
            files_str += f" ... (+{len(sorted_files) - 5} more files, {remaining_count} more occurrences)"

        print(f"{original_rank:2d}. Count: {count:4d} | Files: [{files_str}]")
        print(f"    Text: \"{ngram[:100]}{'...' if len(ngram) > 100 else ''}\"")
        print()


# Load corpus and augmented texts
print("Loading corpus texts...")
corpus_texts_with_paths = get_corpus_text_strings_with_paths(
    dataset_config, RAW_DATASET_NAME, BUILD_NAME
)
corpus_texts = [text for text, path in corpus_texts_with_paths]
corpus_file_names = [path for text, path in corpus_texts_with_paths]
print(f"Loaded {len(corpus_texts)} corpus texts")

print("Loading augmented texts...")
augmented_texts_with_paths = get_all_augmented_text_strings_with_paths(
    dataset_config, RAW_DATASET_NAME, BUILD_NAME
)
augmented_texts = [text for text, path in augmented_texts_with_paths]
augmented_file_names = [path for text, path in augmented_texts_with_paths]
print(f"Loaded {len(augmented_texts)} augmented texts")

if not corpus_texts:
    print("ERROR: No corpus texts found. Check your dataset configuration.")
    sys.exit(1)

if not augmented_texts:
    print("ERROR: No augmented texts found. Check your dataset configuration.")
    sys.exit(1)

# N-gram lengths to analyze
ngram_lengths = [4, 8, 12, 16, 20]

# Perform analysis for each n-gram length
print("\n" + "=" * 80)
print("DATASET OVERLAP ANALYSIS")
print("=" * 80)

all_results = {}

for n in ngram_lengths:
    print(f"\n--- Analyzing {n}-grams ---")

    # Count n-grams in corpus
    print(f"Counting {n}-grams in corpus...")
    corpus_ngrams, corpus_files = count_ngrams_in_texts(
        corpus_texts, n, tokenizer, corpus_file_names
    )

    # Count n-grams in augmented data
    print(f"Counting {n}-grams in augmented data...")
    augmented_ngrams, augmented_files = count_ngrams_in_texts(
        augmented_texts, n, tokenizer, augmented_file_names
    )

    # Analyze overlap
    overlap_results = analyze_overlap(corpus_ngrams, augmented_ngrams, n)
    all_results[n] = overlap_results

    # Print statistics
    print(f"\n{n}-gram Statistics:")
    print(f"  Total corpus {n}-grams: {overlap_results['total_corpus_ngrams']:,}")
    print(f"  Total augmented {n}-grams: {overlap_results['total_augmented_ngrams']:,}")
    print(f"  Unique overlapping {n}-grams: {overlap_results['overlapping_ngrams']:,}")
    print(
        f"  Corpus {n}-grams appearing in augmented: {overlap_results['overlap_in_corpus']:,} ({overlap_results['corpus_overlap_percentage']:.2f}%)"
    )
    print(
        f"  Augmented {n}-grams from corpus: {overlap_results['overlap_in_augmented']:,} ({overlap_results['augmented_from_corpus_percentage']:.2f}%)"
    )

    # Print top corpus n-grams (self-comparison)
    print_top_ngrams(
        corpus_ngrams,
        corpus_files,
        f"Top 20 Most Common {n}-grams in Corpus (Self-Comparison)",
    )

    # Find overlapping n-grams and their counts
    overlapping_ngrams = set(corpus_ngrams.keys()) & set(augmented_ngrams.keys())
    overlapping_counts = Counter()
    overlapping_files = {}

    for ngram in overlapping_ngrams:
        overlapping_counts[ngram] = augmented_ngrams[ngram]
        overlapping_files[ngram] = augmented_files[ngram]

    # Print top overlapping n-grams
    print_top_ngrams(
        overlapping_counts,
        overlapping_files,
        f"Top 20 Most Common {n}-grams in Augmented Data (Found in Corpus)",
    )

# Create summary visualizations
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

# Create plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Corpus overlap percentage
ngram_lens = list(all_results.keys())
corpus_overlap_pcts = [all_results[n]["corpus_overlap_percentage"] for n in ngram_lens]

ax1.bar(range(len(ngram_lens)), corpus_overlap_pcts)
ax1.set_xlabel("N-gram Length")
ax1.set_ylabel("Percentage")
ax1.set_title("Percentage of Corpus N-grams Found in Augmented Data")
ax1.set_xticks(range(len(ngram_lens)))
ax1.set_xticklabels([f"{n}-gram" for n in ngram_lens])
for i, pct in enumerate(corpus_overlap_pcts):
    ax1.text(i, pct + 0.5, f"{pct:.1f}%", ha="center")

# Plot 2: Augmented data from corpus percentage
augmented_from_corpus_pcts = [
    all_results[n]["augmented_from_corpus_percentage"] for n in ngram_lens
]

ax2.bar(range(len(ngram_lens)), augmented_from_corpus_pcts, color="orange")
ax2.set_xlabel("N-gram Length")
ax2.set_ylabel("Percentage")
ax2.set_title("Percentage of Augmented N-grams Coming from Corpus")
ax2.set_xticks(range(len(ngram_lens)))
ax2.set_xticklabels([f"{n}-gram" for n in ngram_lens])
for i, pct in enumerate(augmented_from_corpus_pcts):
    ax2.text(i, pct + 0.5, f"{pct:.1f}%", ha="center")

# Plot 3: Total n-gram counts
corpus_counts = [all_results[n]["total_corpus_ngrams"] for n in ngram_lens]
augmented_counts = [all_results[n]["total_augmented_ngrams"] for n in ngram_lens]

x = np.arange(len(ngram_lens))
width = 0.35

ax3.bar(x - width / 2, corpus_counts, width, label="Corpus", alpha=0.8)
ax3.bar(x + width / 2, augmented_counts, width, label="Augmented", alpha=0.8)
ax3.set_xlabel("N-gram Length")
ax3.set_ylabel("Total N-gram Count")
ax3.set_title("Total N-gram Counts by Length")
ax3.set_xticks(x)
ax3.set_xticklabels([f"{n}-gram" for n in ngram_lens])
ax3.legend()
ax3.set_yscale("log")

# Plot 4: Overlap counts
overlap_counts = [all_results[n]["overlapping_ngrams"] for n in ngram_lens]

ax4.bar(range(len(ngram_lens)), overlap_counts, color="green")
ax4.set_xlabel("N-gram Length")
ax4.set_ylabel("Count")
ax4.set_title("Number of Unique Overlapping N-grams")
ax4.set_xticks(range(len(ngram_lens)))
ax4.set_xticklabels([f"{n}-gram" for n in ngram_lens])
for i, count in enumerate(overlap_counts):
    ax4.text(i, count + max(overlap_counts) * 0.01, f"{count:,}", ha="center")

plt.tight_layout()

# Save the plot
output_dir = Path("notebooks/outputs")
output_dir.mkdir(parents=True, exist_ok=True)
plot_filename = (
    f"dataset_overlap_analysis_{DATA_CONFIG_NAME}_{RAW_DATASET_NAME}_{BUILD_NAME}.png"
)
plot_path = output_dir / plot_filename

try:
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nAnalysis plots saved to: {plot_path}")
except Exception as e:
    print(f"Error saving plots: {e}")

plt.show()

# Print final summary
print(f"\nFinal Summary for {DATA_CONFIG_NAME}:{RAW_DATASET_NAME}:{BUILD_NAME}")
print("-" * 60)
for n in ngram_lens:
    results = all_results[n]
    print(
        f"{n}-grams: {results['corpus_overlap_percentage']:.1f}% of corpus found in augmented, "
        f"{results['augmented_from_corpus_percentage']:.1f}% of augmented comes from corpus"
    )

print(
    f"\nAnalysis complete! Found {len(corpus_texts)} corpus texts and {len(augmented_texts)} augmented texts."
)

# %%
