import asyncio
import logging
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

from playwright.async_api import Page, async_playwright

from backend.config import get_settings

logger = logging.getLogger(__name__)

USER_AGENT = "UzhavanRAGBot/1.0 (+educational research; respects robots.txt)"
_SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".zip", ".rar",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".mp3", ".mp4", ".css", ".js",
)

_EXTRACT_JS = """
() => {
  const text = el => (el.textContent || "").replace(/\\s+/g, " ").trim();
  const headings = [...document.querySelectorAll("h1,h2,h3")].map(text).filter(Boolean);
  const links = [...document.querySelectorAll("a[href]")].map(a => a.href);
  const root = document.querySelector("main, article, #content, .content, [role=main]") || document.body;
  const clone = root.cloneNode(true);
  clone.querySelectorAll("script,style,noscript,nav,footer,header,iframe,svg,form").forEach(e => e.remove());
  const blocks = [...clone.querySelectorAll("h1,h2,h3,h4,p,li,td,th,div")]
    .filter(e => !e.querySelector("p,li,td,th,div,h1,h2,h3,h4"))
    .map(text).filter(t => t.length > 1);
  const body = blocks.length ? blocks.join("\\n") : text(clone);
  return { title: document.title.trim(), headings, links, content: body };
}
"""


@dataclass
class ScrapedPage:
    url: str
    title: str
    headings: list[str]
    content: str
    scraped_at: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))


def _host(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _normalize(url: str) -> str:
    url, _ = urldefrag(url)
    return url.rstrip("/") if urlparse(url).path not in ("", "/") else url.rstrip("/") + "/"


def _is_valid_link(url: str, base_host: str) -> bool:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or _host(url) != base_host:
        return False
    return not p.path.lower().endswith(_SKIP_EXTENSIONS)


def _load_robots(base_url: str) -> RobotFileParser | None:
    parser = RobotFileParser()
    parser.set_url(urljoin(base_url, "/robots.txt"))
    try:
        parser.read()
        return parser
    except Exception as exc:
        logger.warning("Could not read robots.txt (%s); proceeding with default rules", exc)
        return None


async def _fetch(page: Page, url: str, retries: int, timeout_ms: int) -> tuple[ScrapedPage, list[str]] | None:
    for attempt in range(1, retries + 1):
        try:
            response = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            if response is not None and response.status >= 400:
                logger.warning("Skipping %s (HTTP %d)", url, response.status)
                return None
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            data = await page.evaluate(_EXTRACT_JS)
            if not data["content"].strip() and not data["headings"]:
                logger.warning("Skipping %s (no extractable content)", url)
                return None
            scraped = ScrapedPage(
                url=page.url,
                title=data["title"] or url,
                headings=data["headings"][:30],
                content=data["content"],
            )
            return scraped, data["links"]
        except Exception as exc:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            await asyncio.sleep(attempt * 1.5)
    logger.error("Giving up on %s after %d attempts", url, retries)
    return None


async def scrape_site() -> list[ScrapedPage]:
    s = get_settings()
    base_url = s.uzhavan_base_url
    base_host = _host(base_url)
    robots = await asyncio.to_thread(_load_robots, base_url)

    queue: deque[str] = deque([_normalize(base_url)])
    seen: set[str] = {queue[0]}
    results: list[ScrapedPage] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=USER_AGENT)
        page = await context.new_page()
        try:
            while queue and len(results) < s.scrape_max_pages:
                url = queue.popleft()
                if robots is not None and not robots.can_fetch(USER_AGENT, url):
                    logger.info("Blocked by robots.txt: %s", url)
                    continue
                fetched = await _fetch(page, url, s.scrape_retries, s.scrape_timeout_ms)
                if fetched is not None:
                    scraped, links = fetched
                    if all(r.url != scraped.url for r in results):
                        results.append(scraped)
                        logger.info("Scraped %d/%d: %s", len(results), s.scrape_max_pages, scraped.url)
                    for link in links:
                        norm = _normalize(link)
                        if norm not in seen and _is_valid_link(norm, base_host):
                            seen.add(norm)
                            queue.append(norm)
                await asyncio.sleep(random.uniform(s.scrape_delay_min, s.scrape_delay_max))
        finally:
            await browser.close()

    if len(results) < s.scrape_min_pages:
        logger.warning("Only %d pages scraped (minimum target %d)", len(results), s.scrape_min_pages)
    return results


def _run_in_new_loop() -> list[ScrapedPage]:
    loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
    try:
        return loop.run_until_complete(scrape_site())
    finally:
        loop.close()


async def scrape_uzhavan() -> list[ScrapedPage]:
    """Run the async scraper on a dedicated loop so Playwright subprocesses work on Windows."""
    return await asyncio.to_thread(_run_in_new_loop)
