from __future__ import annotations


def scrape_url(url: str | None) -> str:
    if not url:
        return ""
    from trafilatura import extract, fetch_url

    downloaded = fetch_url(url)
    return extract(downloaded) or ""


def parse_pdf(path: str | None) -> str:
    if not path:
        return ""
    import fitz

    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def extract_source_text(source_type: str, *, value: str | None, path: str | None) -> str:
    if source_type == "url":
        return scrape_url(value)
    if source_type == "pdf":
        return parse_pdf(path)
    raise RuntimeError(f"Unsupported source type: {source_type}")
