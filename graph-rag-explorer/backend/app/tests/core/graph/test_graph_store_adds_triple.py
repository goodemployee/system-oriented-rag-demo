import tempfile
from app.core.graph.graph_store import GraphStore


def  test_加入合法_triple_時_GraphStore_會寫入_graph():
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/graph.json"
        store = GraphStore(path=path)

        store.add_triples([{
            "subject": "西瓜",
            "predicate": "含有",
            "object": "水",
        }])

        assert "西瓜" in store.graph.nodes
        assert "水" in store.graph.nodes
        assert store.graph.has_edge("西瓜", "水")
