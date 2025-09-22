import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
from typing import List, Dict, Any, Tuple, Optional
from django.conf import settings
import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class VectorDatabase:
    """Enhanced FAISS-based vector database with cosine similarity and semantic search"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

        # Use IndexFlatIP for cosine similarity (Inner Product after normalization)
        self.index = faiss.IndexFlatIP(self.dimension)

        self.documents = []
        self.metadata = []
        self.document_chunks = []  # Store processed chunks

        # File paths for persistence
        self.index_path = os.path.join(settings.VECTOR_DB_PATH, 'faiss_cosine_index.idx')
        self.docs_path = os.path.join(settings.VECTOR_DB_PATH, 'documents.pkl')
        self.metadata_path = os.path.join(settings.VECTOR_DB_PATH, 'metadata.pkl')
        self.chunks_path = os.path.join(settings.VECTOR_DB_PATH, 'chunks.pkl')

        # Load existing index if available
        self._load_index()

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better embeddings"""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove very short lines (likely navigation/footer text)
        lines = text.split('\n')
        meaningful_lines = [line.strip() for line in lines if len(line.strip()) > 20]

        return ' '.join(meaningful_lines)

    def _create_semantic_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create semantically meaningful chunks from text"""
        chunks = []

        # Split into sentences for better semantic coherence
        sentences = re.split(r'[.!?]+', text)

        # Create overlapping chunks of 3-5 sentences
        chunk_size = 4
        overlap = 1

        for i in range(0, len(sentences), chunk_size - overlap):
            chunk_sentences = sentences[i:i + chunk_size]
            chunk_text = '. '.join(s.strip() for s in chunk_sentences if s.strip())

            if len(chunk_text) > 50:  # Only meaningful chunks
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_id': len(chunks),
                    'chunk_text': chunk_text[:200] + '...' if len(chunk_text) > 200 else chunk_text,
                    'original_length': len(text),
                    'chunk_start': i
                })
                chunks.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })

        # Always include a full document chunk for comprehensive search
        if text and len(text) > 100:
            full_metadata = metadata.copy()
            full_metadata.update({
                'chunk_id': 'full',
                'chunk_text': text[:300] + '...' if len(text) > 300 else text,
                'is_full_document': True
            })
            chunks.append({
                'text': text,
                'metadata': full_metadata
            })

        return chunks

    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]]):
        """Add documents to the vector database with improved chunking and cosine similarity"""
        if not documents:
            return

        logger.info(f"Adding {len(documents)} documents to vector database")

        all_chunks = []
        all_chunk_metadata = []

        for doc_idx, (document, doc_metadata) in enumerate(zip(documents, metadata)):
            # Preprocess document
            processed_doc = self._preprocess_text(document)

            if not processed_doc:
                continue

            # Create semantic chunks
            chunks = self._create_semantic_chunks(processed_doc, doc_metadata)

            for chunk in chunks:
                all_chunks.append(chunk['text'])
                all_chunk_metadata.append(chunk['metadata'])

        if not all_chunks:
            logger.warning("No valid chunks created from documents")
            return

        # Generate embeddings for all chunks
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
        embeddings = self.model.encode(all_chunks, convert_to_tensor=False, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to FAISS index
        self.index.add(embeddings)

        # Store chunks and metadata
        self.document_chunks.extend(all_chunks)
        self.metadata.extend(all_chunk_metadata)

        # Also store original documents for reference
        self.documents.extend(documents)

        # Save to disk
        self._save_index()

        logger.info(f"Successfully added {len(all_chunks)} chunks to vector database")

    def search(self,
               query: str,
               top_k: int = 20,
               similarity_threshold: float = 0.3,
               search_mode: str = 'hybrid'
               ) -> List[Dict[str, Any]]:
        """Enhanced search with multiple modes and better ranking"""

        if self.index.ntotal == 0:
            return []

        # Preprocess query
        processed_query = self._preprocess_text(query)
        if not processed_query:
            processed_query = query

        logger.info(f"Searching for: '{query}' (processed: '{processed_query[:100]}...')")

        # Generate query embedding
        query_embedding = self.model.encode([processed_query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search with higher initial k for better filtering
        search_k = min(top_k * 3, self.index.ntotal)
        similarities, indices = self.index.search(query_embedding, search_k)

        # Process results
        results = []
        seen_urls = set()  # For deduplication

        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.metadata) and similarity >= similarity_threshold:
                metadata = self.metadata[idx]

                # Skip duplicates from same URL (keep highest scoring)
                url = metadata.get('url', '')
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                # Get document text
                doc_text = self.document_chunks[idx] if idx < len(self.document_chunks) else ""

                result = {
                    'document': doc_text,
                    'metadata': metadata,
                    'similarity_score': float(similarity),
                    'cosine_similarity': float(similarity),  # Since we're using normalized vectors
                    'relevance_score': self._calculate_relevance_score(query, doc_text, similarity),
                    'match_explanation': self._generate_match_explanation(query, doc_text, metadata)
                }
                results.append(result)

        # Enhanced ranking
        results = self._rank_results(query, results, search_mode)

        # Return top results
        final_results = results[:top_k]

        logger.info(f"Found {len(final_results)} results above threshold {similarity_threshold}")

        return final_results

    def _calculate_relevance_score(self, query: str, document: str, cosine_sim: float) -> float:
        """Calculate enhanced relevance score combining multiple factors"""

        # Base score from cosine similarity
        relevance = cosine_sim

        # Boost for exact keyword matches
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        keyword_overlap = len(query_words.intersection(doc_words)) / len(query_words) if query_words else 0

        # Boost for document quality indicators
        quality_boost = 0
        if len(document) > 200:  # Substantial content
            quality_boost += 0.1
        if any(term in document.lower() for term in ['ceo', 'executive', 'director', 'manager', 'president']):
            quality_boost += 0.15  # Executive content boost
        if any(term in document.lower() for term in ['biography', 'bio', 'profile', 'about']):
            quality_boost += 0.1  # Profile content boost

        # Combine scores
        final_score = relevance + (keyword_overlap * 0.2) + quality_boost

        return min(final_score, 1.0)  # Cap at 1.0

    def _generate_match_explanation(self, query: str, document: str, metadata: Dict[str, Any]) -> str:
        """Generate explanation of why this result matches"""
        explanations = []

        query_lower = query.lower()
        doc_lower = document.lower()

        # Check for direct matches
        query_words = set(query_lower.split())
        doc_words = set(doc_lower.split())
        matching_words = query_words.intersection(doc_words)

        if matching_words:
            explanations.append(f"Keyword matches: {', '.join(sorted(matching_words))}")

        # Check for executive-related content
        exec_terms = ['ceo', 'cto', 'cfo', 'executive', 'director', 'president', 'manager']
        found_exec_terms = [term for term in exec_terms if term in doc_lower]
        if found_exec_terms:
            explanations.append(f"Executive terms: {', '.join(found_exec_terms)}")

        # Check structured data
        if 'structured_data' in metadata and metadata['structured_data']:
            struct_data = metadata['structured_data']
            if 'person' in struct_data:
                explanations.append("Contains person details")
            if 'company' in struct_data:
                explanations.append("Contains company information")

        # Check for semantic similarity
        if not explanations:
            explanations.append("Semantic similarity match")

        return " | ".join(explanations)

    def _rank_results(self, query: str, results: List[Dict[str, Any]], mode: str = 'hybrid') -> List[Dict[str, Any]]:
        """Advanced result ranking with multiple strategies"""

        if mode == 'similarity':
            # Pure similarity ranking
            return sorted(results, key=lambda x: x['cosine_similarity'], reverse=True)

        elif mode == 'relevance':
            # Relevance-based ranking
            return sorted(results, key=lambda x: x['relevance_score'], reverse=True)

        elif mode == 'hybrid':
            # Hybrid ranking (default)
            for result in results:
                # Combine similarity and relevance with weights
                hybrid_score = (result['cosine_similarity'] * 0.6 + result['relevance_score'] * 0.4)

                # Boost for high-quality content
                metadata = result['metadata']
                if 'structured_data' in metadata and metadata['structured_data']:
                    if 'person' in metadata['structured_data']:
                        hybrid_score += 0.1

                # Boost for executive pages
                if any(term in result['document'].lower() for term in ['ceo', 'executive', 'leadership']):
                    hybrid_score += 0.05

                result['hybrid_score'] = hybrid_score

            return sorted(results, key=lambda x: x.get('hybrid_score', x['cosine_similarity']), reverse=True)

        else:
            # Default to similarity
            return sorted(results, key=lambda x: x['cosine_similarity'], reverse=True)

    def advanced_search(self,
                        query: str,
                        filters: Optional[Dict[str, Any]] = None,
                        boost_terms: Optional[List[str]] = None,
                        top_k: int = 20) -> List[Dict[str, Any]]:
        """Advanced search with filters and boost terms"""

        # Get base results
        results = self.search(query, top_k=top_k * 2, search_mode='hybrid')

        # Apply filters
        if filters:
            filtered_results = []
            for result in results:
                metadata = result['metadata']

                # Filter by domain
                if 'domain' in filters:
                    url = metadata.get('url', '')
                    if not any(domain in url for domain in filters['domain']):
                        continue

                # Filter by content type
                if 'has_person_data' in filters and filters['has_person_data']:
                    if not (metadata.get('structured_data', {}).get('person')):
                        continue

                # Filter by minimum similarity
                if 'min_similarity' in filters:
                    if result['cosine_similarity'] < filters['min_similarity']:
                        continue

                filtered_results.append(result)

            results = filtered_results

        # Apply boost terms
        if boost_terms:
            for result in results:
                boost_score = 0
                doc_lower = result['document'].lower()

                for term in boost_terms:
                    if term.lower() in doc_lower:
                        boost_score += 0.1

                result['boosted_score'] = result.get('hybrid_score', result['cosine_similarity']) + boost_score

            results = sorted(results, key=lambda x: x.get('boosted_score', x['cosine_similarity']), reverse=True)

        return results[:top_k]

    def get_similar_documents(self, document_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to a specific document"""
        try:
            # Find the document by ID
            target_metadata = None
            target_idx = None

            for idx, metadata in enumerate(self.metadata):
                if metadata.get('id') == document_id:
                    target_metadata = metadata
                    target_idx = idx
                    break

            if target_idx is None:
                return []

            # Get the embedding of the target document
            target_embedding = self.index.reconstruct(target_idx).reshape(1, -1)

            # Search for similar documents
            similarities, indices = self.index.search(target_embedding, top_k + 1)  # +1 to exclude self

            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx != target_idx and idx < len(self.metadata):  # Exclude the target document itself
                    metadata = self.metadata[idx]
                    doc_text = self.document_chunks[idx] if idx < len(self.document_chunks) else ""

                    result = {
                        'document': doc_text,
                        'metadata': metadata,
                        'similarity_score': float(similarity),
                        'cosine_similarity': float(similarity)
                    }
                    results.append(result)

            return results[:top_k]

        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return []

    def _save_index(self):
        """Save index and data to disk"""
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, self.index_path)

        # Save documents
        with open(self.docs_path, 'wb') as f:
            pickle.dump(self.documents, f)

        # Save chunks
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.document_chunks, f)

        # Save metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def _load_index(self):
        """Load index and data from disk"""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)

            if os.path.exists(self.docs_path):
                with open(self.docs_path, 'rb') as f:
                    self.documents = pickle.load(f)

            if os.path.exists(self.chunks_path):
                with open(self.chunks_path, 'rb') as f:
                    self.document_chunks = pickle.load(f)

            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.document_chunks),
            'index_size': self.index.ntotal,
            'dimension': self.dimension,
            'similarity_metric': 'cosine',
            'index_type': 'IndexFlatIP (normalized for cosine similarity)'
        }

    def rebuild_index(self):
        """Rebuild the entire index (useful for upgrades)"""
        logger.info("Rebuilding vector database index...")

        if not self.documents:
            logger.warning("No documents to rebuild index from")
            return

        # Clear current index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.document_chunks = []
        metadata_backup = self.metadata.copy()
        self.metadata = []

        # Re-add all documents (this will recreate chunks and embeddings)
        docs_per_metadata = {}
        for i, meta in enumerate(metadata_backup):
            doc_id = meta.get('id')
            if doc_id and i < len(self.documents):
                docs_per_metadata[doc_id] = (self.documents[i], meta)

        documents = []
        metadata = []
        for doc_id in sorted(docs_per_metadata.keys()):
            doc, meta = docs_per_metadata[doc_id]
            documents.append(doc)
            metadata.append(meta)

        # Clear and re-add
        self.documents = []
        self.add_documents(documents, metadata)

        logger.info("Index rebuild complete")