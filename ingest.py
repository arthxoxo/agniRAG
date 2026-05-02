import fitz  # pymupdf
import trafilatura
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
)


def scrape_url(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    return trafilatura.extract(downloaded) or ""


def parse_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def chunk_text(text: str, source: str) -> list[dict]:
    parts = splitter.split_text(text)
    return [
        {"text": part, "source": source, "index": index}
        for index, part in enumerate(parts)
    ]


def ingest_sources(sources: list[dict], kb_id: str, embedder, vector_store):
    total = 0
    for source in sources:
        if source["type"] == "url":
            text = scrape_url(source["value"])
            source_label = source["value"]
        elif source["type"] == "pdf":
            text = parse_pdf(source["path"])
            source_label = source["path"]
        else:
            continue

        if not text.strip():
            continue

        chunks = chunk_text(text, source_label)
        total += vector_store.upsert_chunks(kb_id, chunks, embedder)

    return total
