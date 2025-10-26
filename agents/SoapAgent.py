import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from utilities.utility import query_model, persona_card_from_json


@dataclass
class SoapAgentConfig:
    add_objective_cache: bool = True
    include_markdown: bool = False
    temperature: float = 0.0
    max_output_tokens: int = 900


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
            max_tokens=max_tokens,
            clip_prompt=False,
        )


class SoapAgent:
    def __init__(self, llm_client, scenario, config: SoapAgentConfig, enable_big5=False):
        self.llm = llm_client
        self.scenario = scenario
        self.cfg = config
        self._lines: List[str] = []  # "[turn] Role: content"
        self.enable_big5 = enable_big5
        self.objective_cache: Optional[Dict[str, Any]] = None

    def observe(self, role: str, text: str, turn: int) -> None:
        safe_role = role.strip().title()
        self._lines.append(f"[{turn}] {safe_role}: {text.strip()}")

    def transcript(self) -> str:
        return "\n".join(self._lines)

    def _system_prompt(self) -> str:
        return (
"""You are a clinical transcriber agent specialized in generating structured medical documentation from doctor-patient conversations. Your task is to listen to the dialogue between the doctor and the patient and produce a complete and professional SOAP note based on the information discussed. 
Your output must always follow the SOAP format, divided clearly into the four sections below. Each section should be written in one or two paragraphs, using complete sentences, concise clinical language, and maintaining proper medical tone and formatting.
S: Subjective — Summarize the patient's main complaints, reported symptoms, medical history, lifestyle factors, and any relevant context shared during the conversation. Focus on what the patient describes or feels.
O: Objective — Document the doctor's observations, examination findings, vital signs, and any diagnostic test results or measurable data mentioned. Include objective facts that can be verified or measured.
A: Assessment — Provide the doctor’s medical impression or reasoning based on the subjective and objective information. List possible diagnoses, differential considerations, and relevant clinical insights inferred from the discussion.
P: Plan — Outline the doctor’s proposed management plan, including treatments, medications, lifestyle recommendations, follow-up instructions, or any further diagnostic evaluations discussed during the encounter.
Important: 
- Do not include any dialogue or quotations.
- Only summarize and structure information from the conversation.
- Maintain a professional, factual, and concise medical writing style."""
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
        t += "\n\nNow output the SOAP note report."
        return t

    def generate(self, turn_range: tuple[int, int]) -> Dict[str, Any]:
        sys = self._system_prompt()
        if self.enable_big5:
            doctor_big5 = "agent_personas/doc_pos.json"
            sys = sys + persona_card_from_json(doctor_big5)
        user = self._user_prompt(turn_range, self.objective_cache)
        raw = self.llm.chat(
            system=sys,
            messages=[{"role": "user", "content": user}],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_output_tokens,
        )

        return raw