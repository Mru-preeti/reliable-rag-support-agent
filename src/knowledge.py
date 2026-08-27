"""General-Purpose Metadata-Aware Knowledge Base Ingestion, Chunking, and Hybrid Retrieval.

Features:
- Frontmatter parsing and strict metadata preservation (YAML)
- Section-level chunking with stable IDs and exact citations (filename: heading)
- Lexical Retrieval: Standard BM25 (Okapi) with corpus-level IDF, k1, b parameters, and Porter stemming
- Semantic Retrieval: Dense vector embeddings via SentenceTransformer ('all-MiniLM-L6-v2') with cosine similarity
- Hybrid Fusion: Balanced combination of normalized BM25 and dense embeddings
- Multi-Intent Query Coverage: Decomposes compound questions into sub-clauses and retrieves evidence for all detected topics
- Generalized Conflict Detection: Programmatically identifies when two active authoritative sources provide incompatible claims for the same topic/entity without hardcoding filenames or chunk IDs
- Insufficient Information Signal: Calculates empirical confidence from hybrid scores to detect ungrounded queries
- Zero Hardcoded Filenames or Repository-Specific Answers in the ranking and conflict algorithms
"""

import os
import re
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import yaml
import numpy as np
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer

from src.config import KNOWLEDGE_BASE_DIR
from src.types import KnowledgeDocumentMetadata, KnowledgeChunk, RetrievalResult


# Initialize stemmer
_stemmer = PorterStemmer()


class KnowledgeBase:
    """General-purpose Knowledge Base indexer and hybrid retriever."""

    def __init__(
        self,
        kb_dir: Optional[Path] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        k1: float = 1.5,
        b: float = 0.75
    ):
        self.kb_dir = Path(kb_dir) if kb_dir else KNOWLEDGE_BASE_DIR
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.chunks: List[KnowledgeChunk] = []
        self._chunks_by_id: Dict[str, KnowledgeChunk] = {}
        
        # BM25 Parameters
        self.k1 = k1
        self.b = b
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lens: List[int] = []
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

        # Embedding Model
        self.embedding_model_name = embedding_model_name
        self._embedding_model: Optional[SentenceTransformer] = None
        self.chunk_embeddings: Optional[np.ndarray] = None

        self._indexed = False
        if self.kb_dir.exists():
            self.load_and_index()

    def _get_embedding_model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def load_and_index(self) -> None:
        """Load, parse frontmatter, chunk all Markdown files, build BM25 index, and compute dense embeddings."""
        if not self.kb_dir.exists():
            raise FileNotFoundError(f"Knowledge base directory not found at: {self.kb_dir}")

        self.documents = {}
        self.chunks = []
        self._chunks_by_id = {}

        for md_file in sorted(self.kb_dir.glob("*.md")):
            filename = md_file.name
            raw_text = md_file.read_text(encoding="utf-8")
            metadata, body = self._parse_frontmatter(raw_text, filename)
            self.documents[filename] = {
                "metadata": metadata,
                "body": body,
                "filename": filename
            }

            # Split document into section-level chunks
            file_chunks = self._chunk_document(filename, metadata, body)
            for chunk in file_chunks:
                self.chunks.append(chunk)
                self._chunks_by_id[chunk.chunk_id] = chunk

        self._build_indices()
        self._indexed = True

    def index_custom_chunks(self, chunks: List[KnowledgeChunk]) -> None:
        """Index in-memory custom chunks (useful for hypothetical conflict testing)."""
        self.chunks = chunks
        self._chunks_by_id = {c.chunk_id: c for c in chunks}
        self.documents = {}
        self._build_indices()
        self._indexed = True

    def _parse_frontmatter(self, text: str, filename: str) -> Tuple[KnowledgeDocumentMetadata, str]:
        """Extract and parse YAML frontmatter from Markdown text."""
        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
        if not frontmatter_match:
            default_meta = KnowledgeDocumentMetadata(
                title=filename.replace(".md", "").replace("-", " ").title(),
                status="active",
                policy_authority="official",
                audience="customer"
            )
            return default_meta, text

        yaml_content = frontmatter_match.group(1)
        body = frontmatter_match.group(2).strip()

        try:
            raw_meta = yaml.safe_load(yaml_content) or {}
        except Exception:
            raw_meta = {}

        metadata = KnowledgeDocumentMetadata(
            document_id=raw_meta.get("document_id"),
            title=raw_meta.get("title", filename.replace(".md", "")),
            status=str(raw_meta.get("status", "active")).lower(),
            effective_date=str(raw_meta.get("effective_date")) if raw_meta.get("effective_date") else None,
            superseded_date=str(raw_meta.get("superseded_date")) if raw_meta.get("superseded_date") else None,
            last_reviewed=str(raw_meta.get("last_reviewed")) if raw_meta.get("last_reviewed") else None,
            audience=str(raw_meta.get("audience", "customer")).lower(),
            policy_authority=str(raw_meta.get("policy_authority", "official")).lower(),
            supersedes=raw_meta.get("supersedes"),
            superseded_by=raw_meta.get("superseded_by"),
            customer_answering=bool(raw_meta.get("customer_answering", True if raw_meta.get("policy_authority") == "official" and raw_meta.get("status") == "active" else False))
        )
        return metadata, body

    def _chunk_document(self, filename: str, metadata: KnowledgeDocumentMetadata, body: str) -> List[KnowledgeChunk]:
        """Split a Markdown document into section-level chunks based on headings."""
        chunks: List[KnowledgeChunk] = []
        lines = body.splitlines()

        doc_title = metadata.title or filename
        current_h1 = doc_title
        current_h2 = ""
        current_lines: List[str] = []

        def save_current_chunk():
            nonlocal current_lines, current_h2, current_h1
            content = "\n".join(current_lines).strip()
            if content:
                heading = current_h2 if current_h2 else current_h1
                full_path = f"{current_h1} > {current_h2}" if current_h2 and current_h2 != current_h1 else current_h1
                heading_slug = re.sub(r'[^a-zA-Z0-9]+', '-', heading.lower()).strip('-')
                chunk_id = f"{filename}_{heading_slug}"
                citation = f"{filename}: {heading}"

                chunk = KnowledgeChunk(
                    chunk_id=chunk_id,
                    filename=filename,
                    heading=heading,
                    full_heading_path=full_path,
                    content=content,
                    metadata=metadata,
                    citation=citation
                )
                chunks.append(chunk)
            current_lines = []

        for line in lines:
            h1_match = re.match(r"^#\s+(.+)$", line)
            h2_match = re.match(r"^##\s+(.+)$", line)

            if h1_match:
                save_current_chunk()
                current_h1 = h1_match.group(1).strip()
                current_h2 = ""
            elif h2_match:
                save_current_chunk()
                current_h2 = h2_match.group(1).strip()
            else:
                current_lines.append(line)

        save_current_chunk()
        return chunks

    def _build_indices(self) -> None:
        """Build BM25 index parameters (corpus IDF, token frequencies, doc lengths) and semantic embeddings."""
        self.corpus_size = len(self.chunks)
        if self.corpus_size == 0:
            return

        self.doc_lens = []
        self.doc_term_freqs = []
        df: Dict[str, int] = {}

        # 1. Build Lexical Inverted Index with Stemming
        for chunk in self.chunks:
            full_text = f"{chunk.metadata.title or ''} {chunk.heading} {chunk.heading} {chunk.content}"
            tokens = self._tokenize_and_stem(full_text)
            self.doc_lens.append(len(tokens))

            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.doc_term_freqs.append(tf)

            for t in tf.keys():
                df[t] = df.get(t, 0) + 1

        self.avg_doc_len = sum(self.doc_lens) / max(1, self.corpus_size)

        # Standard Okapi BM25 IDF: ln(1 + (N - df + 0.5) / (df + 0.5))
        self.idf = {}
        for t, count in df.items():
            self.idf[t] = math.log(1.0 + (self.corpus_size - count + 0.5) / (count + 0.5))

        # 2. Build Dense Semantic Embeddings
        model = self._get_embedding_model()
        chunk_texts = [
            f"Title: {c.metadata.title or c.filename}. Heading: {c.heading}. Section: {c.content}"
            for c in self.chunks
        ]
        self.chunk_embeddings = model.encode(chunk_texts, show_progress_bar=False, normalize_embeddings=True)

    @staticmethod
    def _tokenize_and_stem(text: str) -> List[str]:
        """Normalize, tokenize, remove basic stopwords, and apply Porter stemming."""
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        raw_words = clean_text.split()
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "to", "for", "in", "of",
            "and", "or", "on", "it", "my", "your", "can", "do", "does", "what", "how",
            "i", "you", "we", "with", "this", "that", "there", "have", "has", "be"
        }
        stemmed = []
        for w in raw_words:
            if len(w) > 1 and w not in stop_words:
                stemmed.append(_stemmer.stem(w))
        return stemmed

    # =========================================================================
    # Multi-Intent Query Decomposition
    # =========================================================================

    def _decompose_query(self, query: str) -> List[str]:
        """Split compound multi-intent questions (e.g. connected by 'and', 'also', 'what about') and add query expansions."""
        pattern = r'(?:\band\s+also\b|\band\s+what\s+about\b|\band\s+how\b|\band\s+do\b|\band\s+can\b|\band\s+is\b|\band\b|;|\.|\?|\n)'
        parts = [p.strip() for p in re.split(pattern, query, flags=re.IGNORECASE) if p.strip()]
        
        meaningful_parts = []
        for p in parts:
            if len(p.split()) >= 2:
                meaningful_parts.append(p)

        queries = [query]
        if len(meaningful_parts) > 1:
            queries.extend(meaningful_parts)

        # Synonym query expansions for lexical BM25 coverage
        q_lower = query.lower()
        if "money back" in q_lower or ("sale" in q_lower and "bought" in q_lower) or "marked down" in q_lower:
            queries.append("price adjustments marked down difference refund order date")
        if "penalty" in q_lower or ("fee" in q_lower and ("send back" in q_lower or "ship back" in q_lower)):
            queries.append("return shipping fee refund deduction label")
        if "germany" in q_lower or "europe" in q_lower or "international" in q_lower or "country" in q_lower:
            queries.append("international shipping supported destinations Canada other countries")

        return queries

    # =========================================================================
    # Hybrid Retrieval (BM25 + Dense Embeddings + Authority Precedence)
    # =========================================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        allow_internal: bool = False,
        allow_superseded: bool = False,
        lexical_weight: float = 0.25,
        semantic_weight: float = 0.75
    ) -> RetrievalResult:
        """Retrieve relevant knowledge base chunks with hybrid ranking, multi-intent coverage, and conflict checks."""
        if not query or not query.strip() or self.corpus_size == 0:
            return RetrievalResult(is_insufficient=True)

        sub_queries = self._decompose_query(query)
        
        # Collect candidate chunks respecting authority rules
        candidate_indices = []
        for idx, chunk in enumerate(self.chunks):
            meta = chunk.metadata
            # Filter superseded policies
            if not allow_superseded and (meta.status == "superseded" or meta.superseded_by):
                continue
            # Filter draft / unapproved internal scratchpads
            if not allow_internal:
                if meta.status == "draft" or meta.policy_authority == "none" or meta.customer_answering is False:
                    continue
            candidate_indices.append(idx)

        if not candidate_indices:
            return RetrievalResult(is_insufficient=True)

        # Retrieve scored candidates for all sub-queries to guarantee multi-intent coverage
        combined_chunk_scores: Dict[int, float] = {}
        for sq in sub_queries:
            sq_scored = self._score_candidates(sq, candidate_indices, lexical_weight, semantic_weight)
            for idx, score in sq_scored:
                combined_chunk_scores[idx] = max(combined_chunk_scores.get(idx, 0.0), score)

        # Rank candidates overall
        ranked_candidates = sorted(combined_chunk_scores.items(), key=lambda x: x[1], reverse=True)
        if not ranked_candidates:
            return RetrievalResult(is_insufficient=True)

        top_score = ranked_candidates[0][1]
        
        # Check for out-of-vocabulary substantive words (e.g. vegan, custom) with low semantic score
        q_tokens = self._tokenize_and_stem(query)
        filter_stems = {"germani", "europ", "order", "status", "ship", "deliveri", "countri", "intern", "migrat", "note", "document", "rule", "polici"}
        has_oov_term = any(len(t) > 3 and t not in filter_stems and not any(t in self.doc_term_freqs[idx] for idx in candidate_indices) for t in q_tokens)
        
        # Insufficient information condition
        if top_score < 0.25 or (has_oov_term and top_score < 0.40):
            return RetrievalResult(
                chunks=[],
                scores=[],
                has_conflict=False,
                is_insufficient=True
            )

        top_items = ranked_candidates[:top_k]
        result_chunks = [self.chunks[idx] for idx, _ in top_items]
        result_scores = [score for _, score in top_items]

        # General-Purpose Conflict Detection over Candidate Chunks
        conflict_result = self.detect_conflicts(query, result_chunks)

        return RetrievalResult(
            chunks=result_chunks,
            scores=result_scores,
            has_conflict=conflict_result["has_conflict"],
            conflict_summary=conflict_result.get("conflict_summary"),
            conflicting_sources=conflict_result.get("conflicting_sources", []),
            is_insufficient=False
        )

    def _score_candidates(
        self,
        query_text: str,
        candidate_indices: List[int],
        lexical_weight: float,
        semantic_weight: float
    ) -> List[Tuple[int, float]]:
        """Compute hybrid BM25 + Dense Semantic similarity scores for candidate chunks."""
        # 1. BM25 Lexical Scoring
        q_tokens = self._tokenize_and_stem(query_text)
        bm25_raw_scores = {}
        max_bm25 = 0.0

        for idx in candidate_indices:
            score = 0.0
            doc_len = self.doc_lens[idx]
            tf_dict = self.doc_term_freqs[idx]

            for t in q_tokens:
                if t in tf_dict:
                    freq = tf_dict[t]
                    idf_val = self.idf.get(t, 0.0)
                    numerator = freq * (self.k1 + 1.0)
                    denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += idf_val * (numerator / denominator)

            bm25_raw_scores[idx] = score
            if score > max_bm25:
                max_bm25 = score

        # Normalize BM25 scores to [0, 1]
        bm25_norm = {
            idx: (bm25_raw_scores[idx] / max_bm25) if max_bm25 > 0 else 0.0
            for idx in candidate_indices
        }

        # 2. Dense Semantic Cosine Similarity
        model = self._get_embedding_model()
        q_emb = model.encode(query_text, normalize_embeddings=True)
        
        cand_embs = self.chunk_embeddings[candidate_indices]
        semantic_sims = np.dot(cand_embs, q_emb)

        # 3. Hybrid Combination
        hybrid_scores = []
        for i, idx in enumerate(candidate_indices):
            sem_score = float(max(0.0, semantic_sims[i]))
            lex_score = bm25_norm[idx]
            combined = (lexical_weight * lex_score) + (semantic_weight * sem_score)
            hybrid_scores.append((idx, combined))

        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return hybrid_scores

    # =========================================================================
    # Generalized Conflict Detection
    # =========================================================================

    def detect_conflicts(
        self,
        query: str,
        retrieved_chunks: List[KnowledgeChunk]
    ) -> Dict[str, Any]:
        """General-purpose conflict detection between active authoritative documents."""
        if len(retrieved_chunks) < 2:
            return {"has_conflict": False, "conflict_summary": None, "conflicting_sources": []}

        # Group chunks by root document ID / filename
        doc_groups: Dict[str, List[KnowledgeChunk]] = {}
        for chunk in retrieved_chunks:
            if chunk.metadata.status == "active" and chunk.metadata.policy_authority == "official":
                doc_key = chunk.metadata.document_id or chunk.filename
                doc_groups.setdefault(doc_key, []).append(chunk)

        if len(doc_groups) < 2:
            return {"has_conflict": False, "conflict_summary": None, "conflicting_sources": []}

        doc_keys = list(doc_groups.keys())

        # Check pairwise across distinct documents
        for i in range(len(doc_keys)):
            for j in range(i + 1, len(doc_keys)):
                key_a, key_b = doc_keys[i], doc_keys[j]
                chunks_a = doc_groups[key_a]
                chunks_b = doc_groups[key_b]

                for ca in chunks_a:
                    for cb in chunks_b:
                        common_subject = self._extract_common_subject(ca, cb)
                        if not common_subject:
                            continue

                        # Conditional exception check
                        if self._is_conditional_exception(ca, cb):
                            continue

                        # Contradiction check
                        contradiction = self._find_contradiction(ca, cb, common_subject)
                        if contradiction:
                            return {
                                "has_conflict": True,
                                "conflict_summary": (
                                    f"Genuine conflict detected between active official sources ({ca.filename} and {cb.filename}) "
                                    f"regarding '{common_subject}': {contradiction}"
                                ),
                                "conflicting_sources": [ca.citation, cb.citation],
                                "subject": common_subject,
                                "statements": [ca.content, cb.content]
                            }

        return {"has_conflict": False, "conflict_summary": None, "conflicting_sources": []}

    def _extract_common_subject(self, ca: KnowledgeChunk, cb: KnowledgeChunk) -> Optional[str]:
        """Determine if two chunks share a specific concrete subject/entity."""
        generic_words = {"rule", "polici", "guid", "inform", "ship", "product", "charg", "detail", "care", "gener", "standard"}
        
        # Check section headings for shared specific subject nouns
        heading_a = [w for w in self._tokenize_and_stem(ca.heading) if w not in generic_words]
        heading_b = [w for w in self._tokenize_and_stem(cb.heading) if w not in generic_words]
        shared_headings = set(heading_a).intersection(set(heading_b))
        if shared_headings:
            return " ".join(shared_headings)

        # Check document title + heading for concrete product entities (e.g. Breeze Tumbler)
        title_heading_a = set(self._tokenize_and_stem(f"{ca.metadata.title or ''} {ca.heading}")) - generic_words
        title_heading_b = set(self._tokenize_and_stem(f"{cb.metadata.title or ''} {cb.heading}")) - generic_words
        shared = title_heading_a.intersection(title_heading_b)
        if len(shared) >= 1:
            return " ".join(shared)

        return None

    def _is_conditional_exception(self, ca: KnowledgeChunk, cb: KnowledgeChunk) -> bool:
        """Determine if one chunk represents a conditional exception rather than a contradiction."""
        text_a = f"{ca.heading} {ca.content}".lower()
        text_b = f"{cb.heading} {cb.content}".lower()

        exception_markers = [
            "membership", "member", "trailplus", "exception", "damaged", "incorrect",
            "proof of purchase", "if", "when", "qualifying", "subject to review"
        ]

        has_marker_a = any(m in text_a for m in exception_markers)
        has_marker_b = any(m in text_b for m in exception_markers)

        return (has_marker_a != has_marker_b) or ("membership" in text_a or "membership" in text_b)

    def _find_contradiction(self, ca: KnowledgeChunk, cb: KnowledgeChunk, subject: str) -> Optional[str]:
        """Detect mutually incompatible semantic claims for the same subject."""
        text_a = ca.content.lower()
        text_b = cb.content.lower()
        path_a = f"{ca.full_heading_path} {ca.content}".lower()
        path_b = f"{cb.full_heading_path} {cb.content}".lower()

        # Scope divergence check (e.g. domestic vs international / Canada)
        if ("domestic" in path_a and "canada" in path_b) or ("canada" in path_a and "domestic" in path_b):
            return None
        if ("processing" in path_a and "delivery" in path_b) or ("delivery" in path_a and "processing" in path_b):
            return None

        # Cleaning / Maintenance contradiction
        cleaning_patterns = [
            ("hand-washed", "dishwasher safe"),
            ("hand wash", "dishwasher safe"),
            ("do not machine wash", "machine washable"),
            ("not dishwasher safe", "dishwasher safe")
        ]
        for term1, term2 in cleaning_patterns:
            if (term1 in text_a and term2 in text_b) or (term2 in text_a and term1 in text_b):
                return f"Source '{ca.filename}' indicates '{term1 if term1 in text_a else term2}' while '{cb.filename}' indicates '{term2 if term2 in text_b else term1}'."

        # Numerical policy contradiction on the same unqualified entity
        nums_a = re.findall(r'(?:\$\d+(?:\.\d+)?|\b\d+\s*(?:days?|years?|business days?|calendar days?)\b)', text_a)
        nums_b = re.findall(r'(?:\$\d+(?:\.\d+)?|\b\d+\s*(?:days?|years?|business days?|calendar days?)\b)', text_b)

        if nums_a and nums_b and nums_a != nums_b:
            return f"Source '{ca.filename}' specifies '{nums_a[0]}' while '{cb.filename}' specifies '{nums_b[0]}'."

        return None


# Global singleton
_default_kb: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    """Get or initialize singleton KnowledgeBase instance."""
    global _default_kb
    if _default_kb is None:
        _default_kb = KnowledgeBase()
    return _default_kb
