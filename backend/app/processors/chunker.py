from app.models.document import Document
from app.models.chunk import Chunk
import re

MAX_WORDS_PER_CHUNK = 250
HEADING_PATTERN = re.compile(r"^(#{1,3}\s+.+)$", re.MULTILINE)
# Split on sentence-ending punctuation only when followed by an uppercase letter,
# opening quote, or bracket — the hallmarks of a genuine new sentence.
# Two negative lookbehinds suppress false splits on:
#   (?<![A-Z][a-z]\.)   — 2-letter abbreviations: Mr. Dr. Ms. Sr. Jr.
#   (?<![A-Z][a-z][a-z]\.) — 3-letter abbreviations: Mrs. etc.
# Note: single-capital initials (A.) are not suppressed; they are rare in
# notes and would otherwise block acronym endings like U.S.A.
SENTENCE_PATTERN = re.compile(
    r"(?<![A-Z][a-z]\.)"       # not after Mr. Dr. Ms. Sr. Jr.
    r"(?<![A-Z][a-z][a-z]\.)"  # not after Mrs. etc.
    r"(?<=[.!?])\s+(?=[A-Z\"(\[])"
)

class Chunker:
	"""Source-agnostic, structure-aware adaptive chunker for documents.
	
	Implements hierarchical chunking strategy:
	1. Small documents: return as single chunk
	2. Heading-based: split by markdown sections
	3. Paragraph grouping: group paragraphs until word limit
	4. Sentence fallback: split paragraphs by sentences
	5. Word fallback: split by words/tokens if necessary
	"""

	def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
		"""Convert a list of documents into chunks.
		
		Args:
			documents: List of Document objects to chunk
			
		Returns:
			List of Chunk objects with document context prepended
		"""
		chunks: list[Chunk] = []

		for document in documents:
			chunks.extend(self._chunk_document(document))

		return chunks

	def _chunk_document(self, document: Document) -> list[Chunk]:
		"""Chunk a single document using adaptive strategy.
		
		Args:
			document: Document to chunk
			
		Returns:
			List of Chunk objects
		"""
		word_count = len(document.content.split())

		# Small documents: return as single chunk
		if word_count <= MAX_WORDS_PER_CHUNK:
			return [
				self._create_chunk(
					document=document,
					content=document.content,
					chunk_index=0,
					section_title=None,
				)
			]

		# Large documents: try heading-based chunking
		return self._chunk_by_sections(document)

	def _chunk_by_sections(self, document: Document) -> list[Chunk]:
		"""Chunk document by markdown heading sections.
		
		Args:
			document: Document to chunk
			
		Returns:
			List of chunks, each optionally starting with a section title
		"""
		sections = self._extract_sections(document.content)
		chunks: list[Chunk] = []
		chunk_index = 0

		for section_title, section_content in sections:
			# If section fits in one chunk, create it directly
			section_word_count = len(section_content.split())
			if section_word_count <= MAX_WORDS_PER_CHUNK:
				chunks.append(
					self._create_chunk(
						document=document,
						content=section_content,
						chunk_index=chunk_index,
						section_title=section_title,
					)
				)
				chunk_index += 1
			else:
				# Section is too large: group paragraphs
				paragraph_chunks = self._group_paragraphs(
					section_content, document, chunk_index, section_title
				)
				chunks.extend(paragraph_chunks)
				chunk_index += len(paragraph_chunks)

		return chunks

	def _extract_sections(self, content: str) -> list[tuple[str | None, str]]:
		"""Extract sections from markdown content by heading level.
		
		Args:
			content: Document content with markdown formatting
			
		Returns:
			List of (section_title, section_content) tuples
		"""
		sections: list[tuple[str | None, str]] = []
		current_section_title: str | None = None
		current_section_lines: list[str] = []

		lines = content.split("\n")

		for line in lines:
			match = HEADING_PATTERN.match(line)
			if match:
				# Save previous section if any
				if current_section_lines:
					section_content = "\n".join(current_section_lines).strip()
					if section_content:
						sections.append((current_section_title, section_content))
					current_section_lines = []

				# Extract heading text (remove # symbols)
				current_section_title = re.sub(r"^#+\s+", "", line).strip()
			else:
				current_section_lines.append(line)

		# Save final section
		if current_section_lines:
			section_content = "\n".join(current_section_lines).strip()
			if section_content:
				sections.append((current_section_title, section_content))

		return sections if sections else [(None, content)]

	def _group_paragraphs(
		self,
		content: str,
		document: Document,
		start_chunk_index: int,
		section_title: str | None,
	) -> list[Chunk]:
		"""Group consecutive paragraphs until word limit is reached.
		
		Args:
			content: Section content to chunk
			document: Parent document
			start_chunk_index: Starting chunk index
			section_title: Section title if applicable
			
		Returns:
			List of chunks
		"""
		# Split by blank lines to get paragraphs
		paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
		chunks: list[Chunk] = []
		chunk_index = start_chunk_index
		current_group: list[str] = []
		current_word_count = 0

		for paragraph in paragraphs:
			paragraph_word_count = len(paragraph.split())

			# If single paragraph exceeds limit, split it
			if paragraph_word_count > MAX_WORDS_PER_CHUNK:
				# First, flush current group if any
				if current_group:
					group_content = "\n\n".join(current_group)
					chunks.append(
						self._create_chunk(
							document=document,
							content=group_content,
							chunk_index=chunk_index,
							section_title=section_title,
						)
					)
					chunk_index += 1
					current_group = []
					current_word_count = 0

				# Then split the large paragraph
				paragraph_chunks = self._split_paragraph(
					paragraph, document, chunk_index, section_title
				)
				chunks.extend(paragraph_chunks)
				chunk_index += len(paragraph_chunks)
			else:
				# Check if adding this paragraph would exceed limit
				if (
					current_word_count + paragraph_word_count
					> MAX_WORDS_PER_CHUNK
				):
					# Flush current group
					if current_group:
						group_content = "\n\n".join(current_group)
						chunks.append(
							self._create_chunk(
								document=document,
								content=group_content,
								chunk_index=chunk_index,
								section_title=section_title,
							)
						)
						chunk_index += 1
						current_group = []
						current_word_count = 0

				# Add paragraph to current group
				current_group.append(paragraph)
				current_word_count += paragraph_word_count

		# Flush final group
		if current_group:
			group_content = "\n\n".join(current_group)
			chunks.append(
				self._create_chunk(
					document=document,
					content=group_content,
					chunk_index=chunk_index,
					section_title=section_title,
				)
			)

		return chunks

	def _split_paragraph(
		self,
		paragraph: str,
		document: Document,
		start_chunk_index: int,
		section_title: str | None,
	) -> list[Chunk]:
		"""Split a single paragraph by sentences if it exceeds word limit.
		
		Args:
			paragraph: Paragraph text to split
			document: Parent document
			start_chunk_index: Starting chunk index
			section_title: Section title if applicable
			
		Returns:
			List of chunks
		"""
		sentences = self._split_sentences(paragraph)
		chunks: list[Chunk] = []
		chunk_index = start_chunk_index
		current_group: list[str] = []
		current_word_count = 0

		for sentence in sentences:
			sentence_word_count = len(sentence.split())

			# If single sentence exceeds limit, split by words
			if sentence_word_count > MAX_WORDS_PER_CHUNK:
				# First, flush current group if any
				if current_group:
					group_content = " ".join(current_group)
					chunks.append(
						self._create_chunk(
							document=document,
							content=group_content,
							chunk_index=chunk_index,
							section_title=section_title,
						)
					)
					chunk_index += 1
					current_group = []
					current_word_count = 0

				# Then split sentence by words
				word_chunks = self._split_by_words(
					sentence, document, chunk_index, section_title
				)
				chunks.extend(word_chunks)
				chunk_index += len(word_chunks)
			else:
				# Check if adding this sentence would exceed limit
				if (
					current_word_count + sentence_word_count
					> MAX_WORDS_PER_CHUNK
				):
					# Flush current group
					if current_group:
						group_content = " ".join(current_group)
						chunks.append(
							self._create_chunk(
								document=document,
								content=group_content,
								chunk_index=chunk_index,
								section_title=section_title,
							)
						)
						chunk_index += 1
						current_group = []
						current_word_count = 0

				# Add sentence to current group
				current_group.append(sentence)
				current_word_count += sentence_word_count

		# Flush final group
		if current_group:
			group_content = " ".join(current_group)
			chunks.append(
				self._create_chunk(
					document=document,
					content=group_content,
					chunk_index=chunk_index,
					section_title=section_title,
				)
			)

		return chunks

	def _split_sentences(self, text: str) -> list[str]:
		"""Split text into sentences using regex pattern.
		
		Args:
			text: Text to split
			
		Returns:
			List of sentences
		"""
		# Split by period, exclamation, question mark followed by space
		sentences = SENTENCE_PATTERN.split(text)
		return [s.strip() for s in sentences if s.strip()]

	def _split_by_words(
		self,
		text: str,
		document: Document,
		start_chunk_index: int,
		section_title: str | None,
	) -> list[Chunk]:
		"""Final fallback: split text into chunks by word count.
		
		Args:
			text: Text to split
			document: Parent document
			start_chunk_index: Starting chunk index
			section_title: Section title if applicable
			
		Returns:
			List of chunks
		"""
		words = text.split()
		chunks: list[Chunk] = []
		chunk_index = start_chunk_index

		for i in range(0, len(words), MAX_WORDS_PER_CHUNK):
			chunk_words = words[i : i + MAX_WORDS_PER_CHUNK]
			chunk_content = " ".join(chunk_words)
			chunks.append(
				self._create_chunk(
					document=document,
					content=chunk_content,
					chunk_index=chunk_index,
					section_title=section_title,
				)
			)
			chunk_index += 1

		return chunks

	def _create_chunk(
		self,
		document: Document,
		content: str,
		chunk_index: int,
		section_title: str | None,
	) -> Chunk:
		"""Create a Chunk object with document context prepended.
		
		Args:
			document: Parent document
			content: Chunk content
			chunk_index: Index of chunk within document
			section_title: Section title if applicable
			
		Returns:
			Chunk object with document context
		"""
		# Generate deterministic chunk ID
		chunk_id = f"{document.id}_{chunk_index}"

		# Prepend document title and section title to content
		context_lines = [f"**{document.title}**"]
		if section_title:
			context_lines.append(f"### {section_title}")
		context_lines.append(content)
		
		prepended_content = "\n\n".join(context_lines)

		# Add section_title to metadata if available
		chunk_metadata = dict(document.metadata)
		if section_title:
			chunk_metadata["section_title"] = section_title

		return Chunk(
			id=chunk_id,
			document_id=document.id,
			document_title=document.title,
			content=prepended_content,
			chunk_index=chunk_index,
			metadata=chunk_metadata,
		)
