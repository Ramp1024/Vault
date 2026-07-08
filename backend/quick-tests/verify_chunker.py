#!/usr/bin/env python3
"""Verify the adaptive chunker implementation."""

from pathlib import Path
import sys

sys.path.append(str(Path.cwd().parent))

from app.connectors.notion.connector import NotionConnector
from app.processors.chunker import Chunker

def verify_chunker():
	"""Verify chunker output and constraints."""
	print("\n" + "=" * 80)
	print("VAULT - CHUNKER VERIFICATION")
	print("=" * 80)

	# Load documents
	print("\n[1] Loading Documents")
	connector = NotionConnector()
	documents = connector.ingest()
	print(f"✅ Loaded {len(documents)} documents")

	# Chunk documents
	print("\n[2] Chunking Documents")
	chunker = Chunker()
	chunks = chunker.chunk_documents(documents)
	print(f"✅ Generated {len(chunks)} chunks")

	# Summary by document
	print("\n[3] Chunks per Document")
	print("-" * 80)
	doc_chunk_counts = {}
	for chunk in chunks:
		if chunk.document_id not in doc_chunk_counts:
			doc_chunk_counts[chunk.document_id] = []
		doc_chunk_counts[chunk.document_id].append(chunk)

	total_chunks = 0
	for document in documents:
		doc_chunks = doc_chunk_counts.get(document.id, [])
		chunk_count = len(doc_chunks)
		total_chunks += chunk_count
		
		# Determine if single chunk (small) or multiple
		indicator = "📄" if chunk_count == 1 else "📑"
		print(f"{indicator} {document.title:40s} → {chunk_count:3d} chunk{'s' if chunk_count != 1 else ' '}")

	# Verify constraints
	print("\n[4] Constraint Verification")
	print("-" * 80)

	verification_results = {
		"document_titles_prepended": 0,
		"deterministic_ids": 0,
		"small_docs_single_chunk": 0,
		"total_checked": 0,
	}

	for document in documents:
		doc_chunks = doc_chunk_counts.get(document.id, [])
		word_count = len(document.content.split())

		for chunk in doc_chunks:
			verification_results["total_checked"] += 1

			# Verify document title is prepended
			if f"**{document.title}**" in chunk.content:
				verification_results["document_titles_prepended"] += 1

			# Verify deterministic ID
			expected_id = f"{document.id}_{chunk.chunk_index}"
			if chunk.id == expected_id:
				verification_results["deterministic_ids"] += 1

		# Verify small documents have single chunk
		if word_count <= 250 and len(doc_chunks) == 1:
			verification_results["small_docs_single_chunk"] += 1

	print(f"✅ Document titles prepended: {verification_results['document_titles_prepended']}/{verification_results['total_checked']}")
	print(f"✅ Deterministic IDs: {verification_results['deterministic_ids']}/{verification_results['total_checked']}")
	print(f"✅ Small docs as single chunk: {verification_results['small_docs_single_chunk']}")

	# Display sample chunks
	print("\n[5] Sample Chunks (First 3 from Large Documents)")
	print("-" * 80)

	samples_shown = 0
	for document in documents:
		doc_chunks = doc_chunk_counts.get(document.id, [])
		if len(doc_chunks) > 1 and samples_shown < 3:
			chunk = doc_chunks[0]
			print(f"\n📖 From: {document.title}")
			print(f"   ID: {chunk.id}")
			print(f"   Index: {chunk.chunk_index}")
			if "section_title" in chunk.metadata:
				print(f"   Section: {chunk.metadata['section_title']}")
			print(f"   Content:")
			print("   " + "-" * 76)
			
			# Show first 500 chars of content
			content_preview = chunk.content[:500]
			lines = content_preview.split("\n")
			for line in lines:
				print(f"   {line}")
			if len(chunk.content) > 500:
				print(f"   ... ({len(chunk.content) - 500} more chars)")
			print("   " + "-" * 76)
			samples_shown += 1

	# Statistics
	print("\n[6] Chunking Statistics")
	print("-" * 80)
	avg_chunk_size = sum(len(c.content.split()) for c in chunks) / len(chunks) if chunks else 0
	max_chunk_size = max((len(c.content.split()) for c in chunks), default=0)
	min_chunk_size = min((len(c.content.split()) for c in chunks), default=0)

	print(f"Average chunk size: {avg_chunk_size:.1f} words")
	print(f"Max chunk size: {max_chunk_size} words (limit: 250)")
	print(f"Min chunk size: {min_chunk_size} words")
	print(f"Total words chunked: {sum(len(c.content.split()) for c in chunks)}")

	# Final status
	print("\n" + "=" * 80)
	if (
		verification_results["document_titles_prepended"] == verification_results["total_checked"]
		and verification_results["deterministic_ids"] == verification_results["total_checked"]
	):
		print("🎉 CHUNKER VERIFICATION SUCCESSFUL")
	else:
		print("⚠️  CHUNKER VERIFICATION INCOMPLETE")
	print("=" * 80 + "\n")

if __name__ == "__main__":
	verify_chunker()
