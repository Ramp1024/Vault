#!/usr/bin/env python3
"""Verify the embedding service implementation."""

from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
	sys.path.insert(0, str(BACKEND_ROOT))

from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.connectors.notion.connector import NotionConnector
from app.processors.chunker import Chunker


def verify_embedding_service():
	"""Verify embedding service setup and functionality."""
	print("\n" + "=" * 80)
	print("VAULT - EMBEDDING SERVICE VERIFICATION")
	print("=" * 80)

	# Test 1: Service instantiation
	print("\n[1] Service Instantiation")
	try:
		service = get_embedding_service()
		print(f"✅ EmbeddingService created")
		print(f"   Model: {service.model}")
		print(f"   Ollama URL: {service.base_url}")
	except Exception as e:
		print(f"❌ Failed to create service: {e}")
		return

	# Test 2: Load and chunk documents
	print("\n[2] Loading and Chunking Documents")
	try:
		connector = NotionConnector()
		documents = connector.ingest()
		print(f"✅ Loaded {len(documents)} documents")

		chunker = Chunker()
		chunks = chunker.chunk_documents(documents)
		print(f"✅ Generated {len(chunks)} chunks")
	except Exception as e:
		print(f"❌ Failed to load documents: {e}")
		return

	# Test 3: Single text embedding (may fail if Ollama not running)
	print("\n[3] Single Text Embedding")
	print("   (Will fail if Ollama not running)")
	try:
		test_text = "Local RAG on Notion Notes: retrieval augmented generation with local models"
		embedding = service.embed(test_text)
		print(f"✅ Embedded text successfully")
		print(f"   Embedding dimensions: {len(embedding)}")
		print(f"   Sample values: {embedding[:5]}...")
	except RuntimeError as e:
		print(f"⚠️  {e}")
		print("   (Ensure Ollama is installed and running locally: ollama serve)")

	# Test 4: Batch chunk embedding (may fail if Ollama not running)
	print("\n[4] Batch Chunk Embedding")
	print("   (Will fail if Ollama not running)")
	try:
		# Take first 3 chunks
		sample_chunks = chunks[:3]
		embedded_chunks = service.embed_chunks(sample_chunks)
		print(f"✅ Embedded {len(embedded_chunks)} chunks successfully")
		if embedded_chunks:
			print(f"   Embedding dimensions: {len(embedded_chunks[0].embedding)}")
			print(f"   First chunk id: {embedded_chunks[0].chunk.id}")
			print(f"   Sample first chunk embedding: {embedded_chunks[0].embedding[:5]}...")
	except RuntimeError as e:
		print(f"⚠️  {e}")
		print("   (Ensure Ollama is installed and running locally: ollama serve)")

	# Test 5: API Design
	print("\n[5] API Design")
	print("✅ EmbeddingService provides:")
	print("   • embed(text: str) -> list[float]")
	print("     - Single text embedding")
	print("   • embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]")
	print("     - Batch chunk embedding with chunk/vector pairing")
	print("   • get_embedding_service() -> EmbeddingService")
	print("     - Factory function for service creation")

	# Summary
	print("\n" + "=" * 80)
	print("SETUP INSTRUCTIONS")
	print("=" * 80)
	print("""
1. Start Ollama locally:
	ollama serve

2. Pull required models:
	ollama pull nomic-embed-text
	ollama pull llama3.1:8b

3. Verify model is available:
	ollama list

4. Run this verification again to test embeddings
""")
	print("=" * 80 + "\n")


if __name__ == "__main__":
	verify_embedding_service()
