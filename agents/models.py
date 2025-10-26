from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SoapTest(BaseModel):
    name: str = ""
    result: str = ""
    evidence_turn: Optional[int] = None
    reference_range: Optional[str] = None


class SoapObjective(BaseModel):
    vitals: Dict[str, Any] = Field(default_factory=dict)
    exam: str = ""
    tests: List[SoapTest] = Field(default_factory=list)


class SoapAssessment(BaseModel):
    problem: str = ""
    differential: List[str] = Field(default_factory=list)
    rationale: str = ""


class SoapPlanItem(BaseModel):
    type: str = ""
    item: str = ""
    rationale: Optional[str] = None


class SoapSection(BaseModel):
    subjective: str = ""
    objective: SoapObjective = Field(default_factory=SoapObjective)
    assessment: List[SoapAssessment] = Field(default_factory=list)
    plan: List[SoapPlanItem] = Field(default_factory=list)


class DiagnosisSection(BaseModel):
    final: str = ""
    differential: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class MetaSection(BaseModel):
    note_status: str = "draft"
    turn_range: List[int] = Field(default_factory=list)
    markdown: Optional[str] = None


class NotePayload(BaseModel):
    soap: SoapSection
    diagnosis: DiagnosisSection
    meta: MetaSection = Field(default_factory=MetaSection)


__all__ = [
    "NotePayload",
    "SoapSection",
    "DiagnosisSection",
    "MetaSection",
]
