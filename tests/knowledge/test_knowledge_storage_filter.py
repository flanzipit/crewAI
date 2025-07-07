from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from unittest.mock import patch

class DummyCollection:
    def query(self, query_texts, n_results, where=None):
        return {
            "ids": [["1", "2", "3"]],
            "metadatas": [[{"idx": 1}, {"idx": 2}, {"idx": 3}]],
            "documents": [["doc1", "doc2", "doc3"]],
            "distances": [[0.1, 0.5, 0.7]],
        }


def test_search_filters_by_score_threshold():
    with patch.object(KnowledgeStorage, "_set_embedder_config"):
        storage = KnowledgeStorage()
    storage.collection = DummyCollection()
    results = storage.search(["query"], limit=3, score_threshold=0.5)
    scores = [r["score"] for r in results]
    assert scores == [0.1, 0.5]
