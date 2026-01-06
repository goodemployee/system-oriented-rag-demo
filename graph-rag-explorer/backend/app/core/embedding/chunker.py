# backend/app/core/chunker.py
from llama_index.core import SimpleDirectoryReader
from typing import List, Dict, TypedDict
import re

# ===== 可調參數 =====
# 以「字元數」近似控制 chunk 大小（避免跨句硬切）
MAX_CHARS_PER_CHUNK = 400        # 每個 chunk 目標上限（字元）
SENTENCE_OVERLAP    = 1          # 句子重疊數（避免語境斷裂）
MIN_SENT_LEN        = 2          # 太短的句子會與相鄰句合併
# 句子邊界：中英標點皆支援；也把「另外/此外/而且/並且/且」視為斷句提示
END_TOKENS_PATTERN = r"(?:。|．|\.|！|!|？|\?|；|;|：|:|,)"
JOINER_TOKENS      = r"(?:另外|此外|而且|並且|且)"

SPLIT_RE = re.compile(
    rf"{END_TOKENS_PATTERN}\s*|(?:[，,]?\s*{JOINER_TOKENS}\s*)",
    flags=re.UNICODE
)

def _clean_text(t: str) -> str:
    # 正規化空白，去除多餘空行與前後空白
    t = re.sub(r"\r\n?", "\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()

def _split_sentences(text: str) -> List[str]:
    """
    簡單保守版：
    只根據明確句號、問號、感嘆號等符號切句，不處理「另外/此外」等詞。
    適合逐步擴充規則。
    """
    if not text:
        return []

    # 句尾符號（僅切這些）
    END_TOKENS_PATTERN = r"[。．\.！？!?；;：:]"

    # 以句尾符號為基礎切分，並保留句尾符號
    parts = re.split(f"({END_TOKENS_PATTERN})", text)

    sentences: List[str] = []
    curr = ""

    for part in parts:
        if not part:
            continue
        curr += part.strip()
        # 碰到句尾符號 → 完整句子
        if re.match(END_TOKENS_PATTERN, part):
            s = curr.strip()
            if s:
                sentences.append(s)
            curr = ""

    # 收尾：若最後沒有標點也當作一句
    if curr.strip():
        sentences.append(curr.strip())

    return sentences


def _pack_chunks(sentences: List[str]) -> List[str]:
    """依字元上限打包成 chunks，僅在句界合併；支援句子重疊。"""
    chunks: List[str] = []
    i = 0
    n = len(sentences)
    while i < n:
        # 起點包含上一個 chunk 的尾端重疊句
        start = max(0, i - SENTENCE_OVERLAP if chunks else i)
        buf, length = [], 0
        j = i
        # 累加句子直到接近 MAX_CHARS_PER_CHUNK
        while j < n:
            s = sentences[j]
            add_len = len(s) + (1 if buf else 0)
            if buf and (length + add_len) > MAX_CHARS_PER_CHUNK:
                break
            buf.append(s)
            length += add_len
            j += 1
        # 如果沒有塞進任何新句子（極端長句），強制切一條
        if not buf:
            buf = [sentences[i]]
            j = i + 1
        # 真正的 chunk 內容（含需要的重疊起點）
        chunk_text = " ".join(sentences[start:j]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        i = j
    return chunks


class DocumentChunk(TypedDict):
    id: int
    text: str
    length: int
    # 如果你實際還有其他欄位，逐步補上即可

def split_document(file_path: str) -> List[DocumentChunk]:
    """
    以「句界」切割文件為 chunks，避免多句黏在一起。
    - 支援 PDF / TXT（由 LlamaIndex 讀取）
    - 僅在句界合併，不做跨句硬切
    - 可調整 MAX_CHARS_PER_CHUNK / SENTENCE_OVERLAP
    """
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()

    all_chunks: List[DocumentChunk] = []
    idx = 0
    for doc in documents:
        text = doc.get_text() if hasattr(doc, "get_text") else getattr(doc, "text", "")
        sentences = _split_sentences(text or "")
        packed = _pack_chunks(sentences) if sentences else []
        for ch in packed:
            all_chunks.append({
                "id": idx,
                "text": ch,
                "length": len(ch)
            })
            idx += 1

    # 若文件極短（如只有「GraphRAG 是一種水果」），仍保證至少一個 chunk
    if not all_chunks:
        content = _clean_text(documents[0].get_text() if documents else "")
        if content:
            all_chunks.append({"id": 0, "text": content, "length": len(content)})

    return all_chunks
