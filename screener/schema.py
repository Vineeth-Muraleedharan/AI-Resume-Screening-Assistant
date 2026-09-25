"""
Pydantic schema for a single resume-vs-JD evaluation result.
Used with PydanticOutputParser to force the LLM into structured, validated output.
"""

from typing import List
from pydantic import BaseModel, Field


class ResumeEvaluation(BaseModel):
    candidate: str = Field(description="Candidate label this evaluation is for")
    match_score: int = Field(
        ge=0, le=100, description="Overall match score against the JD, 0-100"
    )
    matching_skills: List[str] = Field(
        description="Skills/requirements from the JD that the resume clearly demonstrates"
    )
    missing_skills: List[str] = Field(
        description="Skills/requirements from the JD not found in the resume"
    )
    summary: str = Field(
        description="2-3 sentence summary of the candidate's fit for this role"
    )
    strengths: List[str] = Field(
        description="Key strengths relevant to this JD"
    )
    weaknesses: List[str] = Field(
        description="Key gaps or weaknesses relevant to this JD"
    )
    recommendation: str = Field(
        description="One of: 'Strong Fit', 'Possible Fit', 'Not a Fit'"
    )
    justification: str = Field(
        description="Explanation for the recommendation, grounded only in the resume text provided"
    )
    insufficient_jd_detail: bool = Field(
        default=False,
        description="True if the job description was too vague/short to evaluate confidently",
    )