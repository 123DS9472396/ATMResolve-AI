from __future__ import annotations
from pydantic import BaseModel, Field

class CaseCreate(BaseModel):
    complaint_text: str = Field(min_length=8, max_length=2000)
    transaction_id: str = Field(min_length=4, max_length=40)

class ReviewRequest(BaseModel):
    decision: str
    reviewer_note: str = Field(default="", max_length=1000)
    edited_recommendation: str | None = Field(default=None, max_length=2000)

class PolicySearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
