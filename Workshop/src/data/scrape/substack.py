# This file contains methods for scraping Substack publications and saving posts as markdown files
# to the appropriate dataset directory. Main functions include:
# - scrape_substack_to_dataset: scrapes all posts from a substack and saves to dataset directory
# - get_substack_posts: fetches post data from substack API
# - convert_post_to_markdown: converts HTML post content to markdown format using html2text

import os
import re
import html2text
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import html
from substack_api import Newsletter
import requests


def normalize_substack_url(url: str) -> str:
    """
    Normalize a substack URL to ensure it has the proper format for API access.

    Args:
        url: The URL (e.g., "nosetgauge.com", "https://platformer.substack.com", "www.example.com")

    Returns:
        Normalized URL suitable for Newsletter API access
    """
    # Remove any existing protocol and www
    url = url.lower().strip()
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^www\.", "", url)

    # Ensure we have a clean domain
    url = url.rstrip("/")

    # Return as full URL for Newsletter API
    return f"https://{url}"


def scrape_substack_to_dataset(substack_url: str, dataset_name: str) -> None:
    """
    Scrape all posts from a Substack publication and save them as markdown files
    to the dataset's raw data directory.

    Args:
        substack_url: The full substack URL (e.g., "nosetgauge.com" or "platformer.substack.com")
        dataset_name: Name of the dataset to save posts to (will create datasets/raw/{dataset_name}/)
    """
    normalized_url = normalize_substack_url(substack_url)
    print(f"Starting scrape of {normalized_url} for dataset '{dataset_name}'")

    # Get all posts from the substack
    posts = get_substack_posts(normalized_url)
    if not posts:
        print(f"No posts found for {normalized_url}")
        return

    print(f"Found {len(posts)} posts to scrape")

    # Create the dataset directory
    dataset_dir = Path("datasets") / "raw" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Save each post as a markdown file
    for i, post in enumerate(posts):
        try:
            markdown_content = convert_post_to_markdown(post)
            filename = create_safe_filename(post["title"], post["date"])
            file_path = dataset_dir / f"{filename}.md"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            print(f"Saved post {i+1}/{len(posts)}: {post['title'][:50]}...")

        except Exception as e:
            print(f"Error saving post '{post.get('title', 'Unknown')}': {e}")
            continue

    print(f"Scraping complete! Saved {len(posts)} posts to {dataset_dir}")


def get_substack_posts(substack_url: str) -> List[Dict]:
    """
    Fetch all posts from a Substack publication using the substack-api.

    Args:
        substack_url: The normalized substack URL (e.g., "https://nosetgauge.com")

    Returns:
        List of dictionaries containing post data (title, content, date, url, etc.)
    """
    try:
        print(f"Fetching posts from {substack_url} using substack-api")

        newsletter = Newsletter(substack_url)
        api_posts = newsletter.get_posts()

        posts = []
        successful_posts = 0
        skipped_posts = 0

        for i, api_post in enumerate(api_posts):
            try:
                # Add delay to avoid rate limiting (0.5 seconds between requests)
                if i > 0:
                    time.sleep(0.5)

                print(f"Processing post {i+1}/{len(api_posts)}...", end=" ")

                # Get post metadata
                metadata = api_post.get_metadata()

                # Get post content (HTML) - this is where rate limiting often happens
                content_html = api_post.get_content()

                # Extract information from metadata and API post object
                post = {
                    "title": metadata.get("title", "Untitled"),
                    "url": getattr(api_post, "url", metadata.get("canonical_url", "")),
                    "content_html": content_html or "",
                    "author": metadata.get("author", ""),
                }

                # Handle date - try different possible date fields from metadata
                post_date = None
                date_fields = ["post_date", "created_at", "publishedAt", "date"]
                for field in date_fields:
                    if field in metadata and metadata[field]:
                        try:
                            if isinstance(metadata[field], str):
                                # Try to parse ISO format or other common formats
                                post_date = datetime.fromisoformat(
                                    metadata[field].replace("Z", "+00:00")
                                )
                            else:
                                post_date = metadata[field]
                            break
                        except (ValueError, TypeError):
                            continue

                if not post_date:
                    post_date = datetime.now()

                post["date"] = post_date

                posts.append(post)
                successful_posts += 1
                print("✓")

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    # This is likely a crosspost from another substack - skip it
                    print(f"✗ (crosspost/404)")
                    skipped_posts += 1
                elif e.response.status_code == 429:
                    # Rate limited - wait longer and try again
                    print(f"⚠ (rate limited, waiting 5s)")
                    time.sleep(5)
                    try:
                        # Try again after waiting
                        metadata = api_post.get_metadata()
                        content_html = api_post.get_content()

                        post = {
                            "title": metadata.get("title", "Untitled"),
                            "url": getattr(
                                api_post, "url", metadata.get("canonical_url", "")
                            ),
                            "content_html": content_html or "",
                            "author": metadata.get("author", ""),
                        }

                        # Handle date
                        post_date = None
                        date_fields = ["post_date", "created_at", "publishedAt", "date"]
                        for field in date_fields:
                            if field in metadata and metadata[field]:
                                try:
                                    if isinstance(metadata[field], str):
                                        post_date = datetime.fromisoformat(
                                            metadata[field].replace("Z", "+00:00")
                                        )
                                    else:
                                        post_date = metadata[field]
                                    break
                                except (ValueError, TypeError):
                                    continue

                        if not post_date:
                            post_date = datetime.now()

                        post["date"] = post_date
                        posts.append(post)
                        successful_posts += 1
                        print("✓ (retry successful)")

                    except Exception as retry_e:
                        print(f"✗ (retry failed: {retry_e})")
                        skipped_posts += 1
                else:
                    print(f"✗ (HTTP {e.response.status_code})")
                    skipped_posts += 1

            except Exception as e:
                print(f"✗ (error: {str(e)[:50]})")
                skipped_posts += 1
                continue

        print(
            f"\nSummary: {successful_posts} posts fetched, {skipped_posts} posts skipped"
        )
        return posts

    except Exception as e:
        print(f"Error fetching posts from {substack_url}: {e}")
        return []


def convert_post_to_markdown(post: Dict) -> str:
    """
    Convert a post dictionary to markdown format using html2text.

    Args:
        post: Dictionary containing post data (title, content_html, date, url, author)

    Returns:
        Markdown formatted string
    """
    # Configure html2text
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.ignore_emphasis = False
    h.body_width = 0  # Don't wrap lines

    # Convert HTML content to markdown
    content_md = h.handle(post.get("content_html", ""))

    # Create the full markdown document
    markdown_parts = []

    # Add frontmatter-style metadata
    markdown_parts.append(f"# {post.get('title', 'Untitled')}")
    markdown_parts.append("")

    if post.get("author"):
        markdown_parts.append(f"**Author:** {post['author']}")

    if post.get("date"):
        formatted_date = post["date"].strftime("%Y-%m-%d")
        markdown_parts.append(f"**Date:** {formatted_date}")

    if post.get("url"):
        markdown_parts.append(f"**URL:** {post['url']}")

    markdown_parts.append("")
    markdown_parts.append("---")
    markdown_parts.append("")

    # Add the main content
    markdown_parts.append(content_md.strip())

    return "\n".join(markdown_parts)


def create_safe_filename(title: str, date: datetime) -> str:
    """
    Create a safe filename from post title and date.

    Args:
        title: Post title
        date: Post date

    Returns:
        Safe filename string
    """
    # Format date as YYYY-MM-DD
    date_str = date.strftime("%Y-%m-%d")

    # Clean up title for filename
    safe_title = re.sub(r"[^\w\s-]", "", title)  # Remove special chars
    safe_title = re.sub(
        r"[-\s]+", "-", safe_title
    )  # Replace spaces/hyphens with single hyphen
    safe_title = safe_title.strip("-")  # Remove leading/trailing hyphens
    safe_title = safe_title[:50]  # Limit length

    if not safe_title:
        safe_title = "untitled"

    return f"{date_str}-{safe_title}".lower()
