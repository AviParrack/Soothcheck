#!/usr/bin/env env python3
"""
Oscar Moxon Essay Scraper
Scrapes all essays from oscarmoxon.com and saves them as markdown files
"""

import requests
from bs4 import BeautifulSoup
import re
import os
from urllib.parse import urljoin, urlparse
import time
from pathlib import Path


class EssayScraper:
    def __init__(self, base_url="https://www.oscarmoxon.com"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; EssayScraper/1.0)"}
        )
        self.total_words = 0

    def get_essay_links(self, essays_url="/essays/"):
        """Extract all essay links from the essays page"""
        url = urljoin(self.base_url, essays_url)
        print(f"Fetching essay list from: {url}")

        response = self.session.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Find all essay links - they're in post-link class anchors
        essay_links = []

        # Look for links in the post lists (essays, mini essays, studies)
        post_links = soup.find_all("a", class_="post-link")

        for link in post_links:
            href = link.get("href")
            if href:
                full_url = urljoin(self.base_url, href)
                title = link.get_text(strip=True)
                essay_links.append((title, full_url))

        print(f"Found {len(essay_links)} essays")
        return essay_links

    def clean_filename(self, title):
        """Convert title to safe filename"""
        # Remove/replace problematic characters
        filename = re.sub(r"[^\w\s-]", "", title)
        filename = re.sub(r"[-\s]+", "-", filename)
        return filename.strip("-").lower()

    def extract_essay_content(self, url):
        """Extract the main essay content from a URL"""
        print(f"Downloading: {url}")

        response = self.session.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Find the main content - Jekyll posts typically use article or post-content
        content_selectors = [
            "article.post",
            ".post-content",
            ".content",
            "main .wrapper",
            "article",
        ]

        content = None
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                break

        if not content:
            # Fallback: look for the largest text block
            content = soup.find("body")

        if not content:
            raise Exception("Could not find main content")

        # Extract title
        title_elem = soup.find("h1") or soup.find("title")
        title = title_elem.get_text(strip=True) if title_elem else "Untitled"

        # Extract date if available
        date_elem = soup.find("time") or soup.find(class_=re.compile("date|meta"))
        date = date_elem.get_text(strip=True) if date_elem else ""

        # Convert to markdown-ish format
        markdown_content = self.html_to_markdown(content, title, date, url)

        # Count words
        word_count = len(markdown_content.split())
        self.total_words += word_count

        return markdown_content, word_count

    def html_to_markdown(self, content, title, date, url):
        """Convert HTML content to markdown format"""
        # Remove navigation, footer, sidebar elements
        for element in content.find_all(["nav", "footer", "aside", "header"]):
            element.decompose()

        # Remove script and style tags
        for element in content.find_all(["script", "style"]):
            element.decompose()

        # Start building markdown
        lines = []
        lines.append(f"# {title}")
        lines.append("")

        if date:
            lines.append(f"*{date}*")
            lines.append("")

        lines.append(f"*Source: {url}*")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Process content elements
        for element in content.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "blockquote", "ul", "ol", "li"]
        ):
            text = element.get_text(strip=True)
            if not text:
                continue

            if element.name == "h1":
                lines.append(f"# {text}")
            elif element.name == "h2":
                lines.append(f"## {text}")
            elif element.name == "h3":
                lines.append(f"### {text}")
            elif element.name == "h4":
                lines.append(f"#### {text}")
            elif element.name == "h5":
                lines.append(f"##### {text}")
            elif element.name == "h6":
                lines.append(f"###### {text}")
            elif element.name == "blockquote":
                lines.append(f"> {text}")
            elif element.name in ["ul", "ol"]:
                continue  # Handle lists via li elements
            elif element.name == "li":
                lines.append(f"- {text}")
            else:  # paragraphs and other text
                lines.append(text)

            lines.append("")  # Add spacing

        return "\n".join(lines)

    def scrape_all_essays(self, output_dir="corpus"):
        """Main method to scrape all essays"""
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        try:
            # Get all essay links
            essay_links = self.get_essay_links()

            successful_downloads = 0
            failed_downloads = []

            for i, (title, url) in enumerate(essay_links, 1):
                try:
                    print(f"\n[{i}/{len(essay_links)}] Processing: {title}")

                    # Download and convert essay
                    markdown_content, word_count = self.extract_essay_content(url)

                    # Save to file
                    filename = self.clean_filename(title) + ".md"
                    filepath = Path(output_dir) / filename

                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(markdown_content)

                    print(f"✓ Saved: {filename} ({word_count:,} words)")
                    successful_downloads += 1

                    # Be polite to the server
                    time.sleep(1)

                except Exception as e:
                    print(f"✗ Failed to download {title}: {e}")
                    failed_downloads.append((title, url, str(e)))

            # Print summary
            print(f"\n{'='*60}")
            print(f"SCRAPING COMPLETE")
            print(f"{'='*60}")
            print(
                f"Successfully downloaded: {successful_downloads}/{len(essay_links)} essays"
            )
            print(f"Total word count: {self.total_words:,} words")
            print(f"Output directory: {output_dir}/")

            if failed_downloads:
                print(f"\nFailed downloads ({len(failed_downloads)}):")
                for title, url, error in failed_downloads:
                    print(f"  - {title}: {error}")

        except Exception as e:
            print(f"Error during scraping: {e}")


def main():
    scraper = EssayScraper()
    scraper.scrape_all_essays()


if __name__ == "__main__":
    main()
