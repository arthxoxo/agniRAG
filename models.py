from pydantic import BaseModel


class IngestRequest(BaseModel):
    kb_id: str
    sources: list[dict]


class QueryRequest(BaseModel):
    kb_id: str
    query: str
