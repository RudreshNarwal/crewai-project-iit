"""PDF -> page-tagged chunks -> Chroma vector store -> top-k search."""
import uuid

import chromadb
import pdfplumber


def read_pdf(pdf_file) -> list[str]:
    """Return the text of each page (empty string for pages with no text)."""
    with pdfplumber.open(pdf_file) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]


def chunk(pages: list[str], size: int = 800, overlap: int = 100) -> list[tuple[int, str]]:
    """Split each page into overlapping pieces, keeping its 1-based page number.

    800 characters stays under the 256-token limit of Chroma's default embedder.
    """
    # ponytail: chunks never cross a page break, so every chunk has exactly one
    # page to cite. Split on sentences/sections if answers start missing context.
    chunks = []
    for number, text in enumerate(pages, start=1):
        text = " ".join(text.split())
        if not text:
            continue
        for start in range(0, max(len(text) - overlap, 1), size - overlap):
            chunks.append((number, text[start:start + size]))
    return chunks


def build_index(pages: list[str]) -> chromadb.Collection:
    """Embed the chunks into a fresh in-memory Chroma collection."""
    chunks = chunk(pages)
    if not chunks:
        raise ValueError("No text found in this PDF. Scanned PDFs need OCR first.")
    # Every in-memory Chroma client in a process shares one store, so a fixed
    # collection name would mix different users' PDFs. A random name keeps
    # each upload separate.
    # ponytail: old collections are never deleted; restart the app to free them.
    collection = chromadb.EphemeralClient().create_collection(uuid.uuid4().hex)
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=[text for _, text in chunks],
        metadatas=[{"page": page} for page, _ in chunks],
    )
    return collection


def search(collection: chromadb.Collection, query: str, k: int = 4) -> list[tuple[int, str]]:
    """Return the k most similar chunks as (page, text)."""
    hits = collection.query(query_texts=[query], n_results=min(k, collection.count()))
    return [(meta["page"], text) for meta, text in zip(hits["metadatas"][0], hits["documents"][0])]
