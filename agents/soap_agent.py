# agents/soap_agent.py
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from utilities.utility import query_model

from .models import NotePayload


@dataclass
class SoapAgentConfig:
    add_objective_cache: bool = True
    include_markdown: bool = False
    temperature: float = 0.0
    max_output_tokens: int = 600


class QueryModelChatClient:
    """Minimal adapter around utilities.utility.query_model for SOAP generation."""

    def __init__(self, backend: str, tries: int = 6, timeout: float = 2.0):
        self.backend = backend
        self.tries = tries
        self.timeout = timeout

    def chat(
        self,
        *,
        system: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        if not messages:
            raise ValueError("messages must contain at least one entry")
        prompt = messages[-1]["content"]
        return query_model(
            self.backend,
            prompt,
            system,
            tries=self.tries,
            timeout=self.timeout,
            clip_prompt=False,
        )


class SoapAgent:
    def __init__(self, llm_client, scenario, config: SoapAgentConfig):
        self.llm = llm_client
        self.scenario = scenario
        self.cfg = config
        self._lines: List[str] = []  # "[turn] Role: content"
        self.objective_cache: Optional[Dict[str, Any]] = None

    def observe(self, role: str, text: str, turn: int) -> None:
        safe_role = role.strip().title()
        self._lines.append(f"[{turn}] {safe_role}: {text.strip()}")

    def transcript(self) -> str:
        return "\n".join(self._lines)

    def _system_prompt(self) -> str:
        return (
            "You are a meticulous clinical scribe for a simulated case.\n"
            "Summarize only facts stated in the transcript or the provided objective cache.\n"
            "Subjective = patient-reported; Objective = exam/vitals/measurements.\n"
            "For each test include evidence_turn pointing to the turn index where it appeared.\n"
            "If unknown, leave fields empty; do not invent data.\n"
            "Output STRICT JSON matching this schema keys only: "
            "soap.subjective, soap.objective.vitals, soap.objective.exam, soap.objective.tests[], "
            "soap.assessment[], soap.plan[], diagnosis.final, diagnosis.differential, diagnosis.confidence, meta.note_status, meta.turn_range.\n"
            "No markdown or prose -- JSON only."
        )

    def _user_prompt(
        self, turn_range: tuple[int, int], objective_cache: Optional[Dict[str, Any]]
    ) -> str:
        transcript = self.transcript()
        t = f"TRANSCRIPT (turns {turn_range[0]}-{turn_range[1]}):\n{transcript}"
        if self.cfg.add_objective_cache and objective_cache:
            t += "\n\nOBJECTIVE CACHE (authoritative, do not contradict):\n" + json.dumps(
                objective_cache
            )
        t += "\n\nNow output only the JSON object."
        return t

    def generate(self, turn_range: tuple[int, int]) -> Dict[str, Any]:
        sys = self._system_prompt()
        user = self._user_prompt(turn_range, self.objective_cache)
        raw = self.llm.chat(
            system=sys,
            messages=[{"role": "user", "content": user}],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_output_tokens,
        )
        payload = self._validate_or_repair(raw, sys, user)
        if self.cfg.include_markdown:
            payload.setdefault("meta", {})["markdown"] = self._to_markdown(payload)
        payload.setdefault("meta", {})["note_status"] = "final"
        payload["meta"]["turn_range"] = list(turn_range)
        return payload

    def _validate_or_repair(self, raw: str, sys: str, user: str) -> Dict[str, Any]:
        def extract(s: str) -> str:
            i, j = s.find("{"), s.rfind("}")
            if i == -1 or j == -1:
                raise ValueError("no json braces")
            return s[i : j + 1]

        try:
            payload = NotePayload.model_validate_json(extract(raw)).model_dump()
            return payload
        except Exception:
            raw2 = self.llm.chat(
                system=sys,
                messages=[
                    {
                        "role": "user",
                        "content": user
                        + "\n\nYour prior output was invalid. Reprint strictly valid JSON only.",
                    }
                ],
                temperature=0.0,
                max_tokens=self.cfg.max_output_tokens,
            )
            try:
                return NotePayload.model_validate_json(extract(raw2)).model_dump()
            except Exception:
                # Minimal fallback
                return NotePayload(
                    soap={
                        "subjective": "",
                        "objective": {"vitals": {}, "exam": "", "tests": []},
                        "assessment": [],
                        "plan": [],
                    },
                    diagnosis={"final": "", "differential": [], "confidence": 0.0},
                    meta={"note_status": "error"},
                ).model_dump()

    def _to_markdown(self, payload: Dict[str, Any]) -> str:
        soap_section = payload["soap"]
        objective = soap_section["objective"]
        tests = (
            ", ".join(
                f"{t['name']}: {t['result']} [turn {t.get('evidence_turn', '?')}]"
                for t in objective["tests"]
            )
            or "_None_"
        )
        assessments = (
            "\n".join(
                f"- {a['problem']} -> diff: {', '.join(a['differential'])}. "
                f"Rationale: {a['rationale']}"
                for a in soap_section["assessment"]
            )
            or "_None_"
        )
        plan = (
            "\n".join(
                f"- [{p['type']}] {p['item']}"
                f"{f' (why: {p['rationale']})' if p.get('rationale') else ''}"
                for p in soap_section["plan"]
            )
            or "_None_"
        )
        return (
            "# SOAP Note\n"
            "## S (Subjective)\n"
            + (soap_section["subjective"] or "_None_")
            + "\n\n"
            "## O (Objective)\n"
            f"**Vitals:** {objective['vitals'] or '{}'}\n"
            f"**Exam:** {objective['exam'] or '_None_'}\n"
            f"**Studies:** {tests}\n\n"
            "## A (Assessment)\n"
            + assessments
            + "\n\n"
            "## P (Plan)\n"
            + plan
            + "\n\n"
            "## Final diagnosis\n"
            f"**{payload['diagnosis'].get('final', '')}** "
            f"(confidence {payload['diagnosis'].get('confidence', 0):.2f}; "
            f"diff: {', '.join(payload['diagnosis'].get('differential', []))})\n"
        )

