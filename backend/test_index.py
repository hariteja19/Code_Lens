"""
Quick smoke test for the IndexingPipeline.

Run from the project root:
    backend\\venv\\Scripts\\python.exe backend\\test_index.py

What this tests:
- Qdrant local collection creation
- AST chunking on the backend Python source files
- Dense embedding with nomic-embed-text
- BM25 sparse vector building
- Upsert into Qdrant
"""

import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.indexer.pipeline import IndexingPipeline


def main():
    print("Testing indexing pipeline...")

    # Use local Qdrant (no server required)
    pipeline = IndexingPipeline(qdrant_url="local")

    # Create the Qdrant collection first (this was the missing step)
    print("Creating Qdrant collection...")
    pipeline.create_collection()

    # Index just the indexer sub-package — small, fast, well-understood
    repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indexer")
    repo_id = "test_owner/test_repo"

    print(f"Indexing: {repo_path}")

    def callback(status, progress):
        print(f"  [{progress}%] {status}")

    total = pipeline.index_repo(repo_path, repo_id, progress_callback=callback)
    print(f"\nDone! Indexed {total} chunks.")

    # Verify the collection has data
    info = pipeline.client.get_collection(pipeline.collection_name)
    print(f"Collection points: {info.points_count}")


if __name__ == "__main__":
    main()
