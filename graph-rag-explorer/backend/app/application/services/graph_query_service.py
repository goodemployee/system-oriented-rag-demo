from app.core.graph.graph_store import GraphStore

class GraphQueryService:
    def __init__(self, store: GraphStore):
        self.store = store

    def get_related(self, node: str) -> list[dict]:
        return self.store.search_related(node)

    def get_visual_elements(self) -> dict:
        g = self.store.graph

        nodes = [{"data": {"id": n, "label": n}} for n in g.nodes]
        edges = [
            {"data": {
                "source": u,
                "target": v,
                "label": d.get("relation", "")
            }}
            for u, v, d in g.edges(data=True)
        ]

        return {
            "elements": {"nodes": nodes, "edges": edges},
            "meta": {
                "node_count": len(nodes),
                "edge_count": len(edges),
            }
        }
