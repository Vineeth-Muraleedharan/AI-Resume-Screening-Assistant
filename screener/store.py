"""
Embeddings + vector store: builds a FAISS index per candidate from resume
chunks (from ingest.py), and retrieves the most relevant chunks for a given
candidate against a job description (or any query).

Design choice: each candidate gets its OWN small FAISS index, rather than one
shared index with a metadata filter. This guarantees resumes never share
retrieval context with each other (no risk of cross-candidate leakage), and
avoids relying on FAISS/LangChain metadata-filter behavior that can differ
across versions. Resumes are short, so per-candidate indices are cheap.
"""

from typing import Dict, List
from collections import defaultdict

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

EMBEDDING_MODEL = "text-embedding-3-small"


def build_candidate_stores(chunks: List[Document]) -> Dict[str, FAISS]:
    """
    Group chunks by candidate and build one FAISS index per candidate.

    Returns a dict: {candidate_label: FAISS_store}
    """
    by_candidate: Dict[str, List[Document]] = defaultdict(list)
    for chunk in chunks:
        by_candidate[chunk.metadata["candidate"]].append(chunk)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    stores: Dict[str, FAISS] = {}
    for candidate, candidate_chunks in by_candidate.items():
        stores[candidate] = FAISS.from_documents(candidate_chunks, embeddings)

    return stores


def get_candidates(stores: Dict[str, FAISS]) -> List[str]:
    """Return the list of candidate labels currently loaded."""
    return list(stores.keys())


def retrieve_for_candidate(
    stores: Dict[str, FAISS],
    candidate: str,
    query: str,
    k: int = 5,
) -> List[Document]:
    """
    Retrieve the top-k most relevant chunks for ONE candidate, given a query
    (typically the job description text).
    """
    if candidate not in stores:
        raise KeyError(
            f"No resume loaded for candidate '{candidate}'. "
            f"Available candidates: {list(stores.keys())}"
        )

    store = stores[candidate]
    # Don't ask for more chunks than exist in this candidate's index
    k = min(k, store.index.ntotal)
    return store.similarity_search(query, k=k)


def format_retrieved_context(docs: List[Document]) -> str:
    """
    Join retrieved chunks into a single context string for the LLM prompt,
    with page markers for traceability.
    """
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        parts.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


if __name__ == "__main__":
    # Quick manual test — run:
    #   python screener/store.py "<job description text>" <resume1.pdf> <resume2.pdf> ...
    # This calls the OpenAI embeddings API, so it needs OPENAI_API_KEY in .env
    # with active credits.
    import sys
    from dotenv import load_dotenv
    from ingest import load_and_split_multiple

    load_dotenv()

    if len(sys.argv) < 3:
        print("Usage: python store.py '<job description text>' <resume1.pdf> <resume2.pdf> ...")
        sys.exit(1)

    query = sys.argv[1]
    pdf_paths = sys.argv[2:]

    print("Loading and splitting resumes...")
    chunks = load_and_split_multiple(pdf_paths)

    print("Building per-candidate FAISS stores (calls OpenAI embeddings API)...")
    stores = build_candidate_stores(chunks)

    print(f"\nCandidates loaded: {get_candidates(stores)}\n")

    for candidate in get_candidates(stores):
        print(f"=== Top matches for '{candidate}' against query ===")
        results = retrieve_for_candidate(stores, candidate, query, k=5)
        for i, doc in enumerate(results):
            print(f"  [{i}] page {doc.metadata['page']} | {doc.page_content[:100]!r}")
        print()