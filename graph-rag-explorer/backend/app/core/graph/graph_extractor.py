from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Protocol

from app.capabilities.textgen.text_generator import TextGenerator
from app.core.llm.llm import LLM
from app.core.graph.graph_store import Triple

class GraphExtractor:
    """
    GraphExtractor
    -----------------
    基於 LLM 的三元組抽取模組。

    職責：
    - 組 prompt
    - 呼叫 LLM
    - 嘗試解析 JSON / 半結構輸出
    - 正規化為系統內使用的 Triple 結構
    """

    def __init__(
        self,
        llm: TextGenerator,
        max_input_chars: int = 400,
    ) -> None:
        self._generate = llm.generate
        self.max_input_chars: int = max_input_chars


    # ----------------------------------------------------------
    # 三元組抽取
    # ----------------------------------------------------------
    def extract_triples(self, text: str) -> List[Triple]:
        """
        從輸入文字中抽取所有 (subject, predicate, object) 三元組。

        Args:
            text: 原始輸入文字。

        Returns:
            正規化後的 Triple 清單。
        """
        truncated_text: str = text[: self.max_input_chars]

        prompt = f"""
我要做知識圖譜, 請幫我找三元組. 只輸出 JSON 陣列.
格式為[{{"subject":"","predicate":"","object":""}}]
請用繁體中文。
不要給json以外的描述.
object是連接詞的意思.

文字如下：
{truncated_text.strip()}

請輸出結果：
        """

        try:
            result: str = self._generate(prompt)[0]["generated_text"]
            triples = self._parse_triples(result)
            print(f"📊 GraphExtractor：解析到 {len(triples)} 個三元組。")
            return triples
        except Exception as e:
            print(f"❌ GraphExtractor 抽取失敗: {e}")
            return []

    # ----------------------------------------------------------
    # 輔助：解析 JSON / 類 JSON
    # ----------------------------------------------------------
    def _parse_triples(self, text: str) -> List[Triple]:
        """
        嘗試解析模型輸出的 JSON 或半結構文字。

        Args:
            text: LLM 原始輸出。

        Returns:
            正規化後的 Triple 清單。
        """
        raw_triples: List[dict[str, Any]] = []

        # 1️⃣ 直接 JSON
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                raw_triples = parsed
        except Exception:
            pass

        # 2️⃣ 擷取最後一段 JSON 陣列
        if not raw_triples:
            matches = re.findall(r"\[[\s\S]*?\]", text)
            for candidate in reversed(matches):
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, list):
                        raw_triples = parsed
                        break
                except Exception:
                    continue

        # 3️⃣ fallback：自然語言猜測（保守）
        if not raw_triples:
            lines = re.findall(r"(.+?)\s*[，,。]\s*", text)
            for line in lines:
                if any(kw in line for kw in ("注意", "格式", "範例", "說明")):
                    continue
                if "：" in line:
                    left, right = line.split("：", 1)
                    raw_triples.append(
                        {
                            "subject": left.strip(),
                            "predicate": "描述",
                            "object": right.strip(),
                        }
                    )

        return self._normalize_triples(raw_triples)

    # ----------------------------------------------------------
    # 正規化
    # ----------------------------------------------------------
    def _normalize_triples(
        self,
        triples: List[dict[str, Any]],
    ) -> List[Triple]:
        """
        將原始解析結果正規化為系統內的 Triple。

        Args:
            triples: 尚未保證結構正確的三元組資料。

        Returns:
            正規化後的 Triple 清單。
        """
        normalized: List[Triple] = []

        for t in triples:
            s = t.get("subject")
            p = t.get("predicate")
            o = t.get("object")

            if not s or not p:
                continue

            # ⭐ object 缺失時的保守 fallback
            if not o:
                o = p
                p = "隱含"

            normalized.append(
                {
                    "subject": str(s).strip(),
                    "predicate": str(p).strip(),
                    "object": str(o).strip(),
                }
            )

        return normalized

    # ----------------------------------------------------------
    # 輔助：句子是否像「關係描述」
    # ----------------------------------------------------------
    def looks_like_relation(self, text: str) -> bool:
        """
        判斷一句話是否看起來在描述實體關係。
        用於 GraphBuilder 的排序優先度。

        Args:
            text: 輸入句子。

        Returns:
            是否像關係句。
        """
        if not text:
            return False

        t = text.strip()

        if len(t) < 4 or len(t) > self.max_input_chars:
            return False

        relation_keywords = [
            "是", "為", "屬於", "包含", "擁有", "位於", "導致", "造成", "代表",
            "使用", "需要", "提供", "等於", "意味著", "由", "產生", "描述", "稱為",
            "關於", "包括", "形成", "構成", "依賴", "根據", "包含於",
        ]

        keyword_hit: bool = any(k in t for k in relation_keywords)
        multi_entity_like: bool = bool(
            re.search(r"[\u4e00-\u9fff]{2,}.+[\u4e00-\u9fff]{2,}", t)
        )

        if t.endswith(("？", "!", "！")):
            return False

        return keyword_hit and multi_entity_like
