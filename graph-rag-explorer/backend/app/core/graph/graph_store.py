from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

import networkx as nx
from networkx.readwrite import json_graph

from app.config.paths import GRAPH_STORE_PATH


class Triple(TypedDict):
    """
    知識圖譜中的基本三元組結構。
    """
    subject: str
    predicate: Optional[str]
    object: str


class GraphStore:
    """
    GraphStore 負責知識圖譜的儲存與查詢。

    - 內部使用 NetworkX DiGraph
    - 對外僅暴露「三元組層級」的操作
    """

    def __init__(self, path: Optional[str] = None) -> None:
        """
        建立 GraphStore。

        Args:
            path: 圖譜儲存路徑，None 則使用預設 GRAPH_STORE_PATH。
        """
        self.path: Path = Path(path) if path else GRAPH_STORE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.graph: nx.DiGraph = nx.DiGraph()
        self.load()

    def add_triples(self, triples: List[Triple]) -> None:
        """
        將多個三元組加入圖譜並立即儲存。

        Args:
            triples: 三元組清單。
        """
        for t in triples:
            s: Optional[str] = t.get("subject")
            p: Optional[str] = t.get("predicate")
            o: Optional[str] = t.get("object")

            if not s or not o:
                continue

            self.graph.add_node(s, type="entity")
            self.graph.add_node(o, type="entity")
            self.graph.add_edge(s, o, relation=p)

        self.save()

    def search_related(self, node: str) -> List[Triple]:
        """
        查詢指定節點的直接相連關係。

        Args:
            node: 節點名稱。

        Returns:
            與該節點直接相連的三元組清單。
        """
        if node not in self.graph:
            return []

        result: List[Triple] = []

        for neighbor in self.graph.neighbors(node): # type: ignore
            rel: Optional[str] = self.graph.edges[node, neighbor].get("relation")
            result.append(
                {
                    "subject": node,
                    "predicate": rel,
                    "object": neighbor,
                }
            )

        return result

    def save(self) -> None:
        """
        將目前圖譜序列化為 JSON 儲存。
        """
        data = json_graph.node_link_data(self.graph, edges="links")
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        """
        從儲存檔案載入圖譜（若存在）。
        """
        if not self.path.exists():
            return

        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.graph = json_graph.node_link_graph(data)
