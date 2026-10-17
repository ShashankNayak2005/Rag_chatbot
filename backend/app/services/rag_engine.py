import re
from typing import List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.app.services.chunker import DocumentChunk
from backend.app.schemas import QueryResponse, RetrievedChunk, SourceReference

EXACT_NOT_FOUND_STRING = "ANSWER_NOT_FOUND_IN_DOCUMENT"


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences or discrete lines while protecting common abbreviations."""
    protected = text
    for abbr in ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Inc.", "Ltd.", "Corp.", "vs.", "etc."]:
        protected = protected.replace(abbr, abbr.replace(".", "@@DOT@@"))
    raw_parts = re.split(r"[\r\n]+|(?<=[.!?])\s+", protected)
    return [p.replace("@@DOT@@", ".").strip() for p in raw_parts if p.strip()]


SEMANTIC_DOMAINS = {
    "founding": {
        "q_patterns": [
            r"\b(?:founded|founding|established|establishment|inception|incorporated)\b",
            r"\bwhen was (?:the )?(?:company|firm|datavex|it)\b",
            r"\b(?:what|which|in what|in which) year .*\b(?:start|founded|established|launch|begin)\b",
            r"\b(?:start|started|launch|launched|began) (?:operations|business)\b",
        ],
        "s_patterns": [
            r"\bfounded\b", r"\bestablished\b", r"\bstarted\b", r"\b(19\d\d|20\d\d)\b",
        ],
        "expansion": "founded establishment",
    },
    "location": {
        "q_patterns": [
            r"\b(?:headquarters|hq|main office|corporate office)\b",
            r"\b(?:located|location|based|situated|headquartered)\b",
            r"\bwhere is (?:the )?(?:company|firm|headquarters|office|datavex|it)\b",
            r"\b(?:which city|what city|in which city)\b",
        ],
        "s_patterns": [
            r"\bheadquarters\b", r"\blocat\b", r"\bbased\b", r"\bcity\b", r"\bhq\b", r"\boffice\b",
        ],
        "expansion": "headquarters location city",
    },
    "workforce": {
        "q_patterns": [
            r"\bhow many (?:employees|people|staff|workers|members|personnel)\b",
            r"\bhow many .* (?:work|employed|are there)\b",
            r"\b(?:headcount|staff|workforce|employees?|team size|worker count)\b",
            r"\bnumber of (?:employees|people|staff|workers)\b",
        ],
        "s_patterns": [
            r"\bemployees?\b", r"\bstaff\b", r"\bheadcount\b", r"\bworkforce\b",
        ],
        "expansion": "employees workforce staff headcount",
    },
    "services": {
        "q_patterns": [
            r"\b(?:main )?services?\b",
            r"\bsolutions?\b",
            r"\b(?:what does .* do|what are the services|what services)\b",
            r"\b(?:offerings?|products?|specialization)\b",
        ],
        "s_patterns": [
            r"\bservices?\b", r"\bsolutions?\b", r"\bproducts?\b", r"\bengineering\b", r"\bmain services\b",
        ],
        "expansion": "services solutions products",
    },
}



class RAGEngine:
    """TF-IDF Vector retrieval and evidence-based answer generation engine."""

    def __init__(self, chunks: List[DocumentChunk], min_similarity_threshold: float = 0.15):
        if not chunks:
            raise ValueError("RAGEngine cannot be initialized with empty chunks list.")
        
        self.chunks: List[DocumentChunk] = chunks
        self.min_similarity_threshold: float = min_similarity_threshold
        
        # Initialize TfidfVectorizer with unigram + bigram and sublinear tf scaling
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            norm="l2",
        )
        
        self.corpus = [c.text for c in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def _expand_query(self, question: str) -> str:
        """Expand question with related domain vocabulary to bridge vocabulary mismatch."""
        q_lower = question.lower()
        expansions = []
        for domain in SEMANTIC_DOMAINS.values():
            if any(re.search(pat, q_lower) for pat in domain["q_patterns"]):
                expansions.append(domain["expansion"])
        if expansions:
            return f"{question} " + " ".join(expansions)
        return question

    def retrieve(self, question: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """Compute cosine similarity between the question vector and all indexed chunks."""
        # Query with expansion to capture semantic synonyms
        expanded_q = self._expand_query(question)
        q_vec = self.vectorizer.transform([expanded_q])

        # If question contains no recognized vocabulary words
        if q_vec.nnz == 0:
            # Fallback to raw question transform
            q_vec = self.vectorizer.transform([question])
            if q_vec.nnz == 0:
                return []

        # Cosine similarity between 1 x M and N x M -> 1 x N array
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]

        # Rank indices descending
        ranked_indices = np.argsort(sims)[::-1]

        results: List[Tuple[DocumentChunk, float]] = []
        for idx in ranked_indices[:top_k]:
            score = float(sims[idx])
            if score > 0.0:
                results.append((self.chunks[idx], score))

        return results

    def _score_sentence(self, question: str, sentence: str, base_sim: float) -> float:
        """Score an individual sentence or line against question criteria."""
        score = base_sim
        q_lower = question.lower()
        s_lower = sentence.lower()

        # 1. Semantic domain intent matching
        for domain in SEMANTIC_DOMAINS.values():
            q_matches = any(re.search(pat, q_lower) for pat in domain["q_patterns"])
            s_matches = any(re.search(pat, s_lower) for pat in domain["s_patterns"])
            if q_matches and s_matches:
                score += 1.0

        # 2. Key-value match (e.g., 'Founded: 2020' when query contains 'founded')
        generic_keys = {"company", "system", "overview", "report", "document", "following"}
        if ":" in sentence:
            key_part = sentence.split(":", 1)[0].strip().lower()
            key_words = re.findall(r"\b[a-zA-Z0-9_-]+\b", key_part)
            for kw in key_words:
                if len(kw) > 2 and kw not in generic_keys and (kw in q_lower or any(kw[:5] in qw for qw in q_lower.split())):
                    score += 0.6

        # 3. Interrogative type matching
        if re.search(r"\bwhen\b|\bwhat year\b|\bwhich year\b", q_lower):
            if re.search(r"\b(19\d\d|20\d\d|\d{1,4})\b", sentence):
                score += 0.4
        elif re.search(r"\bwhere\b|\bwhich city\b", q_lower):
            if any(w in s_lower for w in ["headquarters", "located", "based", "city", "country", "place", "office"]):
                score += 0.4
        elif re.search(r"\bhow many\b|\bnumber of\b|\bcount\b|\bheadcount\b", q_lower):
            if re.search(r"\b\d+\b", sentence):
                score += 0.4
        elif re.search(r"\bwho\b", q_lower):
            if any(w in s_lower for w in ["dr.", "mr.", "mrs.", "founder", "founded by", "ceo", "president"]):
                score += 0.4

        # 4. Specificity boost: reward non-generic content words
        generic_words = {"company", "system", "overview", "report", "document", "following"}
        s_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", s_lower)) - generic_words
        q_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", q_lower)) - generic_words
        if s_tokens & q_tokens:
            score += 0.35

        return score

    def _extract_salient_sentences(self, chunk_text: str, question: str) -> str:
        """Extract the specific sentence or line that directly answers the query."""
        sentences = split_into_sentences(chunk_text)

        if not sentences:
            return chunk_text.strip()
        if len(sentences) == 1:
            return sentences[0]

        # Check if this chunk is structured as key-value pairs or list items
        is_key_value = any(":" in s and len(s.split()) < 15 for s in sentences)

        try:
            sent_vecs = self.vectorizer.transform(sentences)
            q_vec = self.vectorizer.transform([self._expand_query(question)])
            raw_scores = cosine_similarity(q_vec, sent_vecs)[0]

            adjusted_scores = [
                self._score_sentence(question, s, float(raw_scores[i]))
                for i, s in enumerate(sentences)
            ]

            if is_key_value:
                # For key-value or bullet lists, return ONLY the single most relevant line
                best_idx = int(np.argmax(adjusted_scores))
                if adjusted_scores[best_idx] > 0.0:
                    return sentences[best_idx]
            else:
                # For narrative prose, select top positive-scoring sentences preserving order
                positive_indices = [i for i in np.argsort(adjusted_scores)[::-1] if adjusted_scores[i] > 0.0]
                if positive_indices:
                    selected_indices = sorted(positive_indices[:2])
                    return " ".join(sentences[i] for i in selected_indices)

        except Exception:
            pass

        return sentences[0]

    def _has_evidence(self, question: str, text: str) -> bool:
        """Check if at least one meaningful query word or semantic group matches text."""
        # 1. Direct lexical token check
        q_tokens = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", question.lower())
        stop_words = self.vectorizer.get_stop_words() or set()
        generic_words = {"company", "system", "overview", "report", "document", "following"}
        meaningful_tokens = [t for t in q_tokens if t not in stop_words and t not in generic_words]

        text_lower = text.lower()
        if meaningful_tokens and any(token in text_lower for token in meaningful_tokens):
            return True

        # 2. Semantic domain alignment check
        q_lower = question.lower()
        for domain in SEMANTIC_DOMAINS.values():
            q_matches = any(re.search(pat, q_lower) for pat in domain["q_patterns"])
            s_matches = any(re.search(pat, text_lower) for pat in domain["s_patterns"])
            if q_matches and s_matches:
                return True

        return False

    def answer_question(
        self,
        question: str,
        top_k: int = 3,
        threshold: Optional[float] = None,
    ) -> QueryResponse:
        """Process a natural language question and return evidence or ANSWER_NOT_FOUND_IN_DOCUMENT."""
        cutoff = threshold if threshold is not None else self.min_similarity_threshold
        retrieved = self.retrieve(question, top_k=top_k)

        # Build retrieved chunk representations
        retrieved_dto = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                page_number=chunk.page_number,
                similarity_score=round(score, 4),
                text=chunk.text,
            )
            for chunk, score in retrieved
        ]

        # Condition 1: No matches returned or empty vector
        if not retrieved:
            return QueryResponse(
                question=question,
                answer=EXACT_NOT_FOUND_STRING,
                supporting_passage=None,
                source_reference=None,
                similarity_score=0.0,
                retrieved_chunks=[],
            )

        best_chunk, best_score = retrieved[0]

        # Condition 2: Max similarity score is below the strict threshold
        # Condition 3: No lexical/semantic evidence in best chunk
        if best_score < cutoff or not self._has_evidence(question, best_chunk.text):
            return QueryResponse(
                question=question,
                answer=EXACT_NOT_FOUND_STRING,
                supporting_passage=None,
                source_reference=None,
                similarity_score=round(best_score, 4),
                retrieved_chunks=retrieved_dto,
            )

        # Construct answer by extracting the most salient sentence from the top passage
        answer_sentence = self._extract_salient_sentences(best_chunk.text, question)

        return QueryResponse(
            question=question,
            answer=answer_sentence,
            supporting_passage=best_chunk.text,
            source_reference=SourceReference(
                filename=best_chunk.source_filename,
                page_number=best_chunk.page_number,
                chunk_id=best_chunk.chunk_id,
            ),
            similarity_score=round(best_score, 4),
            retrieved_chunks=retrieved_dto,
        )

