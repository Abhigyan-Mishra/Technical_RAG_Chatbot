"""
Pydantic models for structuring LLM outputs, ensuring consistency and
facilitating reliable parsing. These models are crucial for the advanced
RAG pipeline, especially for detailed referencing and faithfulness checks.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class AnswerReference(BaseModel):
    """
    Represents a single referenced statement within the comprehensive answer.
    Each piece of information in the answer should ideally be backed by such a reference.
    """
    statement: str = Field(description="A specific statement or piece of information extracted or inferred for the answer.")
    source_filename: str = Field(description="The original filename of the document from which this statement is derived (e.g., 'manual_v1.pdf').")
    page_number: int = Field(description="The page number (1-indexed) in the source document where the supporting information is found.")
    verbatim_quote: str = Field(description="A short, direct verbatim quote (1-3 sentences) from the document that directly supports the statement.")

    @validator('page_number')
    def page_number_must_be_positive(cls, v):
        if v < 1:
            raise ValueError('Page number must be 1-indexed and positive.')
        return v

class StructuredAnswer(BaseModel):
    """
    Defines the expected structure for the LLM's final answer.
    This promotes detailed, verifiable, and grounded responses.
    """
    comprehensive_answer: str = Field(description="The overall, synthesized answer to the user's question, constructed from the referenced statements. Can be an empty string if no answer is found.")
    detailed_references: List[AnswerReference] = Field(default_factory=list, description="A list of detailed statements, each with its precise source document, page number, and a supporting verbatim quote. Defaults to an empty list.")
    confidence_score: Optional[float] = Field(default=None, description="A score from 0.0 (no confidence) to 1.0 (very high confidence) indicating the LLM's confidence that its answer is accurate and fully supported *only* by the provided context. Can be null if confidence cannot be determined or if no answer is generated.")
    issues_or_missing_info: Optional[str] = Field(default="None", description="Describes any ambiguities in the query, missing information in the provided context that prevented a complete answer, or parts of the question that couldn't be addressed. Should be 'None' if the answer is complete and unambiguous based on context, or if no answer context was found.")

    @validator('confidence_score')
    def confidence_score_must_be_in_range_if_not_none(cls, v):
        if v is not None and not (0.0 <= v <= 1.0): # Only validate if not None
            raise ValueError('Confidence score, if provided, must be between 0.0 and 1.0.')
        return v

class FaithfulnessCheck(BaseModel):
    """
    Represents the LLM's self-assessment of whether its generated answer
    is faithful to the provided context documents.
    """
    is_faithful: bool = Field(description="True if the entire generated answer is fully supported by the provided context documents and introduces no external information. False otherwise.")
    explanation: str = Field(description="A brief explanation for the faithfulness assessment, especially highlighting any parts of the answer that might not be fully supported or if external information was (incorrectly) used.")
    conflicting_info_present: bool = Field(description="True if the provided context documents contained conflicting information relevant to the query, which might affect the answer's construction or interpretation.")
    unsupported_statements: Optional[List[str]] = Field(default_factory=list, description="A list of specific statements from the generated answer that were found to be NOT directly supported by the provided context. Defaults to an empty list if faithful or if not applicable.")
