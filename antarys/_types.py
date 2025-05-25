from typing import Dict, List, Optional, Any, TypedDict
from dataclasses import dataclass


class VectorRecord(TypedDict, total=False):
    """Vector record structure"""
    id: str
    values: List[float]
    metadata: Optional[Dict[str, Any]]


class SearchResult(TypedDict, total=False):
    """Search result structure"""
    id: str
    score: float
    values: Optional[List[float]]
    metadata: Optional[Dict[str, Any]]


class SearchResults(TypedDict):
    """Search results structure"""
    matches: List[SearchResult]


class BatchSearchResults(TypedDict):
    """Batch search results structure"""
    results: List[SearchResults]


@dataclass
class SearchParams:
    """Search parameters"""
    vector: Optional[List[float]] = None
    top_k: int = 10
    include_values: bool = False
    include_metadata: bool = True
    filter: Optional[Dict] = None
    use_ann: bool = True
    ef_search: int = 100
    threshold: float = 0.0


@dataclass
class UpsertParams:
    """Upsert parameters"""
    vectors: List[VectorRecord]
    batch_size: int = 1000
    show_progress: bool = False


@dataclass
class FilterParams:
    """Filter parameters"""
    field: str
    value: Any
