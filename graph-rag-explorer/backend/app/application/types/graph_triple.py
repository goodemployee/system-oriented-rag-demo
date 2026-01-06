from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GraphTriple:
    subject: str
    predicate: str
    object: str
    confidence: Optional[float] = None
    source_text: Optional[str] = None
