"""Stage 2: LLM constraint proposal (augmentation only).

The LLM here is a **proposal engine**, never a filter generator. Unlike the
legacy :class:`LLMIntentAnalyzer`, it must not emit executable filters or rewrite
the search subject. It only nominates *candidate* constraints — each carrying the
query fragment it believes is evidence, a confidence, and a short rationale — and
hands them to the validation pipeline, which decides whether any may execute.

Only enumerable string fields (a closed set of known values) can be proposed
against; free-text fields are search subjects, not constraints, so they are never
shown to the model. Failure is always safe: any backend/parse error yields no
candidates, so the deterministic analyzer's output is used unchanged.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from app.models.metadata_schema import FieldType, MetadataField, MetadataSchema
from app.models.prompt import Prompt
from app.services.llm import LLM

logger = logging.getLogger(__name__)

# Candidate objects are flat (field/value/confidence/evidence/rationale are all
# scalars), so matching brace-balanced-free ``{...}`` spans extracts each one
# whether the model returns a JSON array, a single bare object, or several
# objects — local models frequently ignore "return an array" and emit one object.
_JSON_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass(frozen=True)
class CandidateConstraint:
    """A constraint the LLM *proposes* — advisory input to the validator.

    Attributes:
        field: The metadata field the model believes the query narrows on.
        value: The proposed value (expected to be one of the field's known
            ``allowed_values``; the validator snaps/rejects non-canonical values).
        confidence: The model's self-reported confidence in ``[0, 1]``.
        evidence: The verbatim query fragment the model cites as support. The
            validator verifies this fragment actually appears in the query — a
            proposal the query does not contain cannot ground a filter.
        rationale: A short human-readable justification, kept only for
            explainability in audits; it never affects acceptance.
    """

    field: str
    value: str
    confidence: float
    evidence: str
    rationale: str = ""


class LLMConstraintProposer:
    """Ask an LLM to nominate candidate metadata constraints for a query.

    The proposer is schema-scoped: it only ever offers the model the enumerable
    fields and their exact allowed values, so the model cannot invent fields or
    values. It returns raw :class:`CandidateConstraint` objects; all trust
    decisions belong to the downstream validator.
    """

    def __init__(
        self,
        llm: LLM,
        schema: MetadataSchema,
        *,
        max_candidates: int = 6,
    ) -> None:
        self.llm = llm
        self.schema = schema
        self.max_candidates = max_candidates
        self._fields = [
            f for f in schema if f.type is FieldType.STRING and f.is_enumerable
        ]

    def propose(self, query: str) -> list[CandidateConstraint]:
        """Return candidate constraints for ``query`` (empty on any failure)."""
        normalized = " ".join(query.split()).strip()
        if not self._fields or not normalized:
            return []
        try:
            raw = self.llm.generate(build_proposal_prompt(normalized, self._fields))
        except Exception:  # pragma: no cover - backend dependent
            logger.warning(
                "LLM constraint proposal failed; proposing nothing", exc_info=True
            )
            return []
        return self._parse(raw)

    def _parse(self, raw: str) -> list[CandidateConstraint]:
        if not raw:
            return []
        candidates: list[CandidateConstraint] = []
        for match in _JSON_OBJECT.finditer(raw):
            try:
                obj = json.loads(match.group())
            except json.JSONDecodeError:
                continue
            candidate = self._coerce(obj)
            if candidate is not None:
                candidates.append(candidate)
            if len(candidates) >= self.max_candidates:
                break
        return candidates

    @staticmethod
    def _coerce(item: object) -> CandidateConstraint | None:
        if not isinstance(item, dict):
            return None
        field = item.get("field")
        value = item.get("value")
        if not isinstance(field, str) or not field.strip():
            return None
        if not isinstance(value, str) or not value.strip():
            return None
        evidence = item.get("evidence")
        rationale = item.get("rationale")
        return CandidateConstraint(
            field=field.strip(),
            value=value.strip(),
            confidence=_coerce_confidence(item.get("confidence")),
            evidence=evidence.strip() if isinstance(evidence, str) else "",
            rationale=rationale.strip() if isinstance(rationale, str) else "",
        )


def _coerce_confidence(value: object) -> float:
    """Coerce a model-reported confidence into ``[0, 1]``; default 0.5.

    A missing/unparseable confidence should neither auto-accept nor auto-reject,
    so it defaults to the midpoint and lets the grounding + threshold gates
    decide.
    """
    if isinstance(value, bool):
        return 0.5
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        try:
            return max(0.0, min(1.0, float(value.strip())))
        except ValueError:
            return 0.5
    return 0.5


_SYSTEM = (
    "You are a constraint PROPOSAL engine for a personal knowledge base search. "
    "You do NOT decide filters, answer the question, or rewrite the query. You "
    "only nominate candidate metadata constraints that a separate validator will "
    "verify and may reject.\n"
    "\n"
    "Rules:\n"
    "1. Only propose a constraint when the query NARROWS the search by that "
    "field. A term the user is searching FOR (the subject) is never a "
    "constraint. Example: in 'where did I mention BM25?', BM25 is the subject — "
    "propose nothing.\n"
    "2. Only use the fields and exact values listed below. Never invent a field "
    "or a value. 'value' must be copied exactly from that field's allowed "
    "values.\n"
    "3. 'evidence' must be the VERBATIM span from the user's question that "
    "signals the constraint (e.g. the word 'completed' for status=Done). If you "
    "cannot cite a real span from the question, do not propose the constraint.\n"
    "4. When unsure, propose nothing. Fewer, well-grounded proposals are better "
    "than speculative ones.\n"
    "5. Respond with a SINGLE JSON array (possibly empty) and nothing else."
)

_OUTPUT_CONTRACT = (
    "Output JSON shape (array; empty [] when nothing applies):\n"
    "[\n"
    "  {\n"
    '    "field": "<field name>",\n'
    '    "value": "<one of that field\'s allowed values>",\n'
    '    "confidence": <number 0..1>,\n'
    '    "evidence": "<verbatim span from the question>",\n'
    '    "rationale": "<short reason>"\n'
    "  }\n"
    "]"
)


def build_proposal_prompt(query: str, fields: list[MetadataField]) -> Prompt:
    """Build the constraint-proposal prompt from the query and enumerable fields.

    Only the fields' names, allowed values, and known aliases are exposed, so the
    model is structurally unable to propose an out-of-schema field or value.
    """
    lines: list[str] = []
    for field in fields:
        values = ", ".join(field.allowed_values)
        entry = f"- {field.name}: [{values}]"
        if field.description:
            entry += f"  # {field.description}"
        lines.append(entry)
        aliases = [
            f"{value} <- {', '.join(alias_list)}"
            for value, alias_list in field.value_aliases
            if alias_list
        ]
        if aliases:
            lines.append(f"    aliases: {'; '.join(aliases)}")
    schema_block = "\n".join(lines)
    user = (
        "Filterable fields and their exact allowed values:\n"
        f"{schema_block}\n\n"
        f"{_OUTPUT_CONTRACT}\n\n"
        f'User question: "{query.strip()}"\n\n'
        "Return only the JSON array."
    )
    return Prompt(system=_SYSTEM, user=user)
