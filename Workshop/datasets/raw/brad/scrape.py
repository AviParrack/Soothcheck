#!/usr/bin/env python3
"""
USV Blog Scraper
Scrapes specific blog posts from Union Square Ventures website
"""

import requests
from bs4 import BeautifulSoup
import re
import time
from pathlib import Path


class USVScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; BlogScraper/1.0)"}
        )
        self.total_words = 0

    # Hardcoded list of URLs to scrape
    URLS_TO_SCRAPE = [
        "https://www.usv.com/writing/2019/06/simulmedia/",
        "https://www.usv.com/writing/2017/05/protocol-labs/",
        "https://www.usv.com/writing/2017/02/tucows/",
        "https://www.usv.com/writing/2016/10/hillary-clinton-for-president/",
        "https://www.usv.com/writing/2016/02/veniams-new-round-of-funding/",
        "https://www.usv.com/writing/2015/12/introducing-koko/",
        "https://www.usv.com/writing/2015/06/introducing-ob1/",
        "https://www.usv.com/writing/2014/12/introducing-veniam/",
        "https://www.usv.com/writing/2014/02/when-did-america-become-too-afraid-to-explore-a-frontier/",
        "https://www.usv.com/writing/2010/08/internet-architecture-and-innovation/",
        "https://www.usv.com/writing/2010/08/a-threat-to-startups/",
        "https://www.usv.com/writing/2010/07/policies-to-encourage-startup-innovation/",
        "https://www.usv.com/writing/2010/06/web-services-as-governments/",
        "https://www.usv.com/writing/2010/02/software-patents-are-the-problem-not-the-answer/",
        "https://www.usv.com/writing/2010/01/we-need-an-independent-invention-defense-to-minimize-the-damage-of-aggressive-patent-trolls/",
        "https://www.usv.com/writing/2009/10/introducing-tracked-com/",
        "https://www.usv.com/writing/2009/08/chris-and-malcolm-are-both-wrong/",
        "https://www.usv.com/writing/2009/05/hacking-education-2/",
        "https://www.usv.com/writing/2009/04/open-spectrum-is-good-policy/",
        "https://www.usv.com/writing/2009/03/welcome-back-dave/",
        "https://www.usv.com/writing/2009/02/pinch-medias-iphone-app-store-secrets/",
        "https://www.usv.com/writing/2009/01/arguing-from-first-principles/",
        "https://www.usv.com/writing/2008/09/why-the-flow-of-innovation-has-reversed/",
        "https://www.usv.com/writing/2008/07/meetup-the-original-web-meets-world-company/",
        "https://www.usv.com/writing/2008/06/internet-for-everyone/",
        "https://www.usv.com/writing/2008/06/the-weird-economics-of-information/",
        "https://www.usv.com/writing/2008/05/pinch-media-investing-on-a-new-platform/",
        "https://www.usv.com/writing/2008/05/losing-jason/",
        "https://www.usv.com/writing/2008/04/wesabe-steps-out/",
        "https://www.usv.com/writing/2008/04/ab-meta/",
        "https://www.usv.com/writing/2008/04/this-is-nuts/",
        "https://www.usv.com/writing/2008/03/new-fund-same-focus/",
        "https://www.usv.com/writing/2007/12/googles-data-asset/",
        "https://www.usv.com/writing/2007/10/markets-and-philanthropy/",
        "https://www.usv.com/writing/2007/10/hacking-philanthropy-the-transcript/",
        "https://www.usv.com/writing/2007/09/there-are-no-open-web-services/",
        "https://www.usv.com/writing/2007/09/what-i-want-from-bug-labs/",
        "https://www.usv.com/writing/2007/07/aol-time-warner-buys-tacoda/",
        "https://www.usv.com/writing/2007/06/wesabe-is-more-than-a-personal-financial-service/",
        "https://www.usv.com/writing/2007/05/who-do-you-trust-to-edit-your-news/",
        "https://www.usv.com/writing/2007/04/cash-flow-forecasting-isnt-what-it-used-to-be/",
        "https://www.usv.com/writing/2007/02/adaptiveblue/",
        "https://www.usv.com/writing/2007/01/whats-next/",
        "https://www.usv.com/writing/2006/11/customer-service-is-the-new-marketing/",
        "https://www.usv.com/writing/2006/09/history-doesnt-repeat-itself-but-it-does-rhyme/",
        "https://www.usv.com/writing/2006/08/welcome-andrew-parker/",
        "https://www.usv.com/writing/2006/08/scalability/",
        "https://www.usv.com/writing/2006/08/defensibility/",
        "https://www.usv.com/writing/2006/08/information-technology-leverage/",
        "https://www.usv.com/writing/2006/08/potential-to-change-the-structure-of-markets/",
        "https://www.usv.com/writing/2006/08/our-focus/",
        "https://www.usv.com/writing/2006/07/through-the-looking-glass-into-the-net-neutrality-debate/",
        "https://www.usv.com/writing/2006/06/sessions/",
        "https://www.usv.com/writing/2006/06/union-square-sessions-2-public-policy-and-innovation/",
        "https://www.usv.com/writing/2006/06/why-we-admire-craigslist/",
        "https://www.usv.com/writing/2006/05/introducing-bug-labs/",
        "https://www.usv.com/writing/2006/05/a-stray-thought-on-the-micro-chunking-of-media/",
        "https://www.usv.com/writing/2006/04/why-has-the-flow-of-technology-reversed/",
        "https://www.usv.com/writing/2006/03/yes-but/",
        "https://www.usv.com/writing/2006/03/will-computing-ever-be-as-invisible-as-electricity/",
        "https://www.usv.com/writing/2006/02/research-and-development/",
        "https://www.usv.com/writing/2006/02/mathematics-how-much-is-enough/",
        "https://www.usv.com/writing/2006/01/physics-the-second-law-of-thermodynamics/",
        "https://www.usv.com/writing/2006/01/web-services-in-the-mist/",
        "https://www.usv.com/writing/2005/12/a-delicious-eight-months/",
        "https://www.usv.com/writing/2005/11/sessions-top-ten-insights-eight-putting-a-string-on-data/",
        "https://www.usv.com/writing/2005/11/sessions-top-ten-insights-seven-less-control-can-create-more-value/",
        "https://www.usv.com/writing/2005/11/sessions-top-ten-insights-six-reputations-are-not-portable/",
        "https://www.usv.com/writing/2005/11/sessions-top-ten-insights-five/",
        "https://www.usv.com/writing/2005/11/sessions-top-ten-insights-four/",
        "https://www.usv.com/writing/2005/11/sessions-top-ten-insights-three/",
        "https://www.usv.com/writing/2005/10/sessions-top-ten-insights-two/",
        "https://www.usv.com/writing/2005/10/sessions-top-ten-insights-one/",
        "https://www.usv.com/writing/2005/10/we-dont-get-it/",
        "https://www.usv.com/writing/2005/10/web-services-are-different/",
        "https://www.usv.com/writing/2005/10/hello-world/",
    ]

    def extract_article_content(self, url):
        """Extract title and article content from a USV blog post"""
        print(f"Scraping: {url}")

        response = self.session.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Extract title from h1.entry-title
        title_elem = soup.find("h1", class_="entry-title")
        title = title_elem.get_text(strip=True) if title_elem else "Untitled"

        # Extract author and date from blog-post-meta
        author_elem = soup.find("a", class_="blog-post-meta__author")
        author = author_elem.get_text(strip=True) if author_elem else "Unknown Author"

        # Date is typically after the author
        meta_elem = soup.find("div", class_="blog-post-meta")
        date = ""
        if meta_elem:
            meta_text = meta_elem.get_text()
            # Extract date pattern (e.g., "Feb 11, 2016")
            date_match = re.search(r"[A-Za-z]+ \d{1,2}, \d{4}", meta_text)
            if date_match:
                date = date_match.group()

        # Extract main article content from entry-content
        content_elem = soup.find("div", class_="entry-content")

        if not content_elem:
            raise Exception("Could not find article content")

        # Extract text content, preserving basic structure
        article_text = self.extract_clean_text(content_elem)

        # Count words
        word_count = len(article_text.split())
        self.total_words += word_count

        return {
            "title": title,
            "author": author,
            "date": date,
            "url": url,
            "content": article_text,
            "word_count": word_count,
        }

    def extract_clean_text(self, content_elem):
        """Convert HTML content to clean text with basic formatting"""
        # Remove unwanted elements
        for element in content_elem.find_all(["script", "style", "nav", "footer"]):
            element.decompose()

        text_parts = []

        # Process different elements
        for element in content_elem.find_all(
            ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote"]
        ):
            text = element.get_text(strip=True)
            if not text:
                continue

            if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                text_parts.append(f"\n{text}\n")
            elif element.name == "blockquote":
                text_parts.append(f"\n> {text}\n")
            elif element.name == "li":
                text_parts.append(f"• {text}")
            else:  # paragraphs and other text
                text_parts.append(text)

            text_parts.append("")  # Add spacing

        return "\n".join(text_parts).strip()

    def clean_filename(self, title):
        """Convert title to safe filename"""
        filename = re.sub(r"[^\w\s-]", "", title)
        filename = re.sub(r"[-\s]+", "-", filename)
        return filename.strip("-").lower()

    def save_as_markdown(self, article_data, output_dir="usv_articles"):
        """Save article as markdown file"""
        Path(output_dir).mkdir(exist_ok=True)

        filename = self.clean_filename(article_data["title"]) + ".md"
        filepath = Path(output_dir) / filename

        # Create markdown content
        markdown_content = f"""# {article_data['title']}
by {article_data['author']} on {article_data['date']}

{article_data['content']}
"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        return filepath

    def scrape_all_articles(self, output_dir="usv_articles"):
        """Main method to scrape all hardcoded URLs"""
        successful_downloads = 0
        failed_downloads = []

        print(f"Starting scrape of {len(self.URLS_TO_SCRAPE)} articles...")
        print("-" * 60)

        for i, url in enumerate(self.URLS_TO_SCRAPE, 1):
            try:
                print(f"[{i}/{len(self.URLS_TO_SCRAPE)}] Processing...")

                # Extract article data
                article_data = self.extract_article_content(url)

                # Save as markdown
                filepath = self.save_as_markdown(article_data, output_dir)

                print(f"✓ Saved: {filepath.name}")
                print(f"  Title: {article_data['title']}")
                print(f"  Author: {article_data['author']}")
                print(f"  Date: {article_data['date']}")
                print(f"  Words: {article_data['word_count']:,}")
                print()

                successful_downloads += 1

                # Be polite to the server
                time.sleep(1)

            except Exception as e:
                print(f"✗ Failed to scrape {url}")
                print(f"  Error: {e}")
                print()
                failed_downloads.append((url, str(e)))

        # Print summary
        print("=" * 60)
        print("SCRAPING COMPLETE")
        print("=" * 60)
        print(
            f"Successfully scraped: {successful_downloads}/{len(self.URLS_TO_SCRAPE)} articles"
        )
        print(f"Total word count: {self.total_words:,} words")
        print(f"Output directory: {output_dir}/")

        if failed_downloads:
            print(f"\nFailed downloads ({len(failed_downloads)}):")
            for url, error in failed_downloads:
                print(f"  - {url}: {error}")


def main():
    scraper = USVScraper()

    # Add more URLs to the URLS_TO_SCRAPE list in the class
    print("URLs to scrape:")
    for i, url in enumerate(scraper.URLS_TO_SCRAPE, 1):
        print(f"  {i}. {url}")
    print()

    scraper.scrape_all_articles()


if __name__ == "__main__":
    main()
