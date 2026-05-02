import unittest

from agni_rag.config import Settings
from agni_rag.core.embeddings import MockEmbedder
from agni_rag.core.llm import MockLLM
from agni_rag.core.rag import RagPipeline
from agni_rag.core.vector_store import MemoryVectorStore


class SmokeTest(unittest.TestCase):
    def test_ingest_and_query(self) -> None:
        settings = Settings()
        pipeline = RagPipeline(
            embedder=MockEmbedder(dim=settings.embed_dim),
            store=MemoryVectorStore(),
            llm=MockLLM(),
            max_chunk_words=settings.max_chunk_words,
            chunk_overlap_words=settings.chunk_overlap_words,
            default_top_k=settings.default_top_k,
        )

        text = "AgniRAG stores customer data locally in a vector database."
        chunks = pipeline.ingest_text("acme", text, source_id="doc-1")
        self.assertGreater(chunks, 0)

        answer = pipeline.answer_question("acme", "Where is data stored?")
        self.assertTrue(answer.answer)
        self.assertGreaterEqual(len(answer.sources), 1)


if __name__ == "__main__":
    unittest.main()
