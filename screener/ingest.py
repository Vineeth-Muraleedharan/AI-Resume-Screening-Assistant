"""
Ingestion: load a resume PDF, split it into chunks, and tag each chunk
with the candidate label (derived from filename) and page number.
"""

import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def sanitize_label(filename: str) -> str:
    """
    Turn a filename into a clean candidate label.
    e.g. 'Vineeth_Muraleedharan_RTT_CV.pdf' -> 'Vineeth_Muraleedharan_RTT_CV'
    Strips extension, replaces anything that isn't alphanumeric/underscore/hyphen
    with an underscore (handles '&', spaces, etc.).
    """
    stem = Path(filename).stem
    label = re.sub(r"[^A-Za-z0-9_\-]", "_", stem)
    return label

def _load_pages(path: Path) -> List[Document]:
    """
    Load a resume file into page-level Documents, depending on format.
    PDFs yield one Document per page (via PyPDFLoader, with a 0-indexed
    'page' in metadata). DOCX files have no native page concept, so they
    load as a single Document tagged page=0.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        return loader.load()
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
        pages = loader.load()
        for p in pages:
            p.metadata.setdefault("page", 0)
        return pages
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}' for '{path.name}'. "
            f"Please upload a PDF or DOCX resume."
        )

def load_and_split_resume(
    pdf_path: str,
    candidate_label: str | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[Document]:
    """
    Load a single resume PDF and split it into overlapping chunks.

    Each returned Document has metadata:
      - candidate: label identifying whose resume this is
      - page: 1-indexed page number within the original PDF
      - source: original filename

    Raises ValueError if the PDF has no extractable text (e.g. a scanned image).
    """
    path = Path(pdf_path)
    if candidate_label is None:
        candidate_label = sanitize_label(path.name)

    pages = _load_pages(path)

    total_chars = sum(len(p.page_content.strip()) for p in pages)
    if total_chars < 30:
        raise ValueError(
            f"'{path.name}' has little or no extractable text "
            f"(only {total_chars} characters found). It may be a scanned "
            f"image PDF. Please upload a text-based PDF instead."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Document] = []
    for page in pages:
        page_number = page.metadata.get("page", 0) + 1  # make 1-indexed
        page_chunks = splitter.split_documents([page])
        for chunk in page_chunks:
            chunk.metadata["candidate"] = candidate_label
            chunk.metadata["page"] = page_number
            chunk.metadata["source"] = path.name
            chunks.append(chunk)

    return chunks


def load_and_split_multiple(pdf_paths: List[str]) -> List[Document]:
    """
    Load and split multiple resumes, tagging each with its own candidate label.
    Returns a single combined list of chunks across all candidates.
    """
    all_chunks: List[Document] = []
    for pdf_path in pdf_paths:
        all_chunks.extend(load_and_split_resume(pdf_path))
    return all_chunks


if __name__ == "__main__":
    # Quick manual test — run: python screener/ingest.py
    import sys

    test_files = sys.argv[1:]
    if not test_files:
        print("Usage: python ingest.py <resume1.pdf> <resume2.pdf> ...")
        sys.exit(1)

    chunks = load_and_split_multiple(test_files)
    print(f"Total chunks: {len(chunks)}\n")

    by_candidate = {}
    for c in chunks:
        by_candidate.setdefault(c.metadata["candidate"], []).append(c)

    for candidate, cchunks in by_candidate.items():
        print(f"--- {candidate}: {len(cchunks)} chunks ---")
        for i, c in enumerate(cchunks[:2]):  # show first 2 chunks as a sanity check
            print(f"  [chunk {i}] page {c.metadata['page']} | {len(c.page_content)} chars")
            print(f"    preview: {c.page_content[:120]!r}")
        print()