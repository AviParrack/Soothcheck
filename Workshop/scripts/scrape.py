#!/usr/bin/env python
# This script scrapes content from various sources and saves it to dataset directories.
# It supports different scraping types and can be extended for new sources.
# The scraped content is saved as raw files in datasets/raw/{dataset_name}/
# which can then be processed using the prepare_dataset script.
#
# Example usage:
# python -m scripts.scrape my_dataset substack noahpinion.blog
# python -m scripts.scrape my_dataset substack platformer.substack.com
# (scrapes the specified substack URL and saves to datasets/raw/my_dataset/)
#
# Arguments:
#   dataset_name (str): Name of the dataset to save scraped content to
#   scraping_type (str): Type of scraping to perform (currently supports: "substack")
#   source_identifier (str): URL for the source (e.g., "noahpinion.blog" or "platformer.substack.com")

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# When running `python -m scripts.scrape` from the project root,
# the project root is added to sys.path, allowing absolute imports from src.
from src.data.scrape.substack import scrape_substack_to_dataset


def scrape_substack(dataset_name: str, substack_url: str) -> None:
    """
    Scrape a Substack publication and save posts to the dataset directory.

    Args:
        dataset_name: Name of the dataset to save posts to
        substack_url: Full URL for the substack (e.g., "noahpinion.blog" or "platformer.substack.com")
    """
    print(f"Starting Substack scraping for {substack_url}")
    try:
        scrape_substack_to_dataset(substack_url, dataset_name)
        print(f"Successfully completed Substack scraping for {substack_url}")
    except Exception as e:
        print(f"Error during Substack scraping: {e}")
        import traceback

        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Scrape content from various sources and save to dataset directories. "
        "The scraped content is saved as raw files which can then be processed using prepare_dataset.",
        epilog="Example usage:\n"
        "  python -m scripts.scrape my_dataset substack noahpinion.blog\n"
        "  python -m scripts.scrape my_dataset substack platformer.substack.com\n"
        "  (scrapes the specified substack and saves to datasets/raw/my_dataset/)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of the dataset to save scraped content to. This will create or add to datasets/raw/[dataset_name]/",
    )

    parser.add_argument(
        "scraping_type",
        type=str,
        choices=["substack"],
        help="Type of scraping to perform. Currently supported: 'substack'",
    )

    parser.add_argument(
        "source_identifier",
        type=str,
        help="URL for the source to scrape. For 'substack': the full domain (e.g., 'noahpinion.blog' or 'platformer.substack.com')",
    )

    args = parser.parse_args()

    print(f"Starting scraping operation:")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Type: {args.scraping_type}")
    print(f"  Source: {args.source_identifier}")
    print()

    # Create the datasets/raw directory structure if it doesn't exist
    raw_datasets_dir = Path("datasets") / "raw"
    raw_datasets_dir.mkdir(parents=True, exist_ok=True)

    # Route to the appropriate scraping function
    if args.scraping_type == "substack":
        scrape_substack(args.dataset_name, args.source_identifier)
    else:
        print(f"Error: Unsupported scraping type '{args.scraping_type}'")
        print(f"Supported types: substack")
        return

    print(f"\nScraping operation completed!")
    print(f"Raw files saved to: datasets/raw/{args.dataset_name}/")
    print(f"Next steps:")
    print(f"  1. Create a config file at configs/data/{args.dataset_name}.json")
    print(f"  2. Run: python -m scripts.prepare_dataset {args.dataset_name}")


if __name__ == "__main__":
    # Ensure we can import from src
    if "src" not in sys.path[0] and "src" not in "".join(sys.path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            print(f"Adjusted sys.path to include project root: {project_root}")

    main()
