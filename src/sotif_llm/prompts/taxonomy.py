"""Prompt design space taxonomy.

Defines the continuous design space over which the safety predictor operates.
This is the direct analog of the turbine design parameters (rated power, diameter,
hub height, specific power) in the error_predictor paper.

Each prompt is assigned a design vector z ∈ [0,1]^d, where d = len(DESIGN_DIMS).
The safety predictor (quantile GP) interpolates/extrapolates the expected SAE
anomaly distance across this space, defining "probably safe" and "possibly safe"
regions — exactly as the MVP defines trust regions in the turbine design space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Design Space Dimensions
# ---------------------------------------------------------------------------
# Each dimension is continuous ∈ [0, 1] and has a semantic meaning.
# A prompt's position in this space determines its "design parameters."

DESIGN_DIMS = {
    "task_complexity": (
        "How cognitively demanding the task is. "
        "0 = trivial (greetings, single-fact lookup), "
        "1 = very hard (multi-step reasoning, novel algorithm design)."
    ),
    "instruction_density": (
        "Constraints per unit of prompt length. "
        "0 = open-ended ('write something about X'), "
        "1 = heavily constrained ('exactly 3 paragraphs, formal tone, "
        "include citations, output as JSON')."
    ),
    "persona_depth": (
        "Degree of persona / roleplay framing. "
        "0 = no persona (direct instruction), "
        "0.5 = light context ('as a teacher...'), "
        "1 = deep immersive roleplay ('you are Commander X aboard...')."
    ),
    "topic_sensitivity": (
        "Semantic proximity to safety-relevant content. "
        "0 = entirely benign (recipes, arithmetic), "
        "0.5 = dual-use (chemistry, cybersecurity education), "
        "1 = boundary-adjacent (fictional violence, ethical hacking tutorials)."
    ),
    "domain_specificity": (
        "How specialized the domain knowledge required is. "
        "0 = general knowledge, "
        "1 = deep expert knowledge (quantum chemistry, compiler theory)."
    ),
    "output_constraints": (
        "Structural demands on the output format. "
        "0 = free-form text, "
        "1 = rigid format (JSON schema, specific CSV columns, code with tests)."
    ),
}


# ---------------------------------------------------------------------------
# Categorical Taxonomy  (used to *generate* prompts; the continuous design
# vector is computed *from* the categorical choices)
# ---------------------------------------------------------------------------

class TaskType(Enum):
    CODE_GENERATION = auto()
    CODE_DEBUGGING = auto()
    CODE_REFACTORING = auto()
    CODE_REVIEW = auto()
    TEXT_SUMMARIZATION = auto()
    TEXT_GENERATION = auto()
    EXPLANATION = auto()
    QUESTION_ANSWERING = auto()
    DATA_ANALYSIS = auto()
    MATH_REASONING = auto()
    INSTRUCTION_FOLLOWING = auto()
    TRANSLATION = auto()
    CLASSIFICATION = auto()
    BRAINSTORMING = auto()
    COMPARISON = auto()
    PLANNING = auto()
    CONVERSION = auto()
    EDITING = auto()


class Domain(Enum):
    # Software & CS
    ALGORITHMS = auto()
    WEB_DEVELOPMENT = auto()
    DATABASES = auto()
    SYSTEMS_PROGRAMMING = auto()
    MACHINE_LEARNING = auto()
    DATA_SCIENCE = auto()
    DEVOPS = auto()
    # Science & Engineering
    PHYSICS = auto()
    CHEMISTRY = auto()
    BIOLOGY = auto()
    MATHEMATICS = auto()
    STATISTICS = auto()
    ENVIRONMENTAL_SCIENCE = auto()
    ENGINEERING = auto()
    ENERGY = auto()
    # Business & Social
    FINANCE = auto()
    MARKETING = auto()
    MANAGEMENT = auto()
    ECONOMICS = auto()
    LAW = auto()
    # Humanities & Arts
    HISTORY = auto()
    PHILOSOPHY = auto()
    LINGUISTICS = auto()
    CREATIVE_WRITING = auto()
    MUSIC = auto()
    # Daily Life
    COOKING = auto()
    TRAVEL = auto()
    FITNESS = auto()
    EDUCATION = auto()
    GENERAL_KNOWLEDGE = auto()


class ComplexityLevel(Enum):
    TRIVIAL = 1       # Single fact, simple greeting, basic arithmetic
    EASY = 2          # Short task, one clear step
    MODERATE = 3      # Multi-step, some reasoning required
    HARD = 4          # Complex reasoning, synthesis of ideas
    EXPERT = 5        # Novel problem-solving, deep domain expertise


class SensitivityLevel(Enum):
    BENIGN = 0          # No conceivable safety concern
    NEUTRAL = 1         # Standard professional topics
    DUAL_USE = 2        # Legitimate but could be misused (security, chemistry)
    FICTIONAL_EDGE = 3  # Fictional framing of sensitive themes


class OutputFormat(Enum):
    FREE_TEXT = auto()
    BULLET_POINTS = auto()
    NUMBERED_LIST = auto()
    JSON = auto()
    CSV = auto()
    MARKDOWN_TABLE = auto()
    CODE_ONLY = auto()
    CODE_WITH_TESTS = auto()
    STRUCTURED_REPORT = auto()


# ---------------------------------------------------------------------------
# Design Vector Scoring
# ---------------------------------------------------------------------------
# Maps categorical prompt attributes to continuous design vector components.

# Task complexity scores by (TaskType, ComplexityLevel)
TASK_COMPLEXITY_SCORES: dict[TaskType, float] = {
    TaskType.QUESTION_ANSWERING: 0.15,
    TaskType.CLASSIFICATION: 0.15,
    TaskType.TRANSLATION: 0.20,
    TaskType.TEXT_SUMMARIZATION: 0.25,
    TaskType.EDITING: 0.25,
    TaskType.BRAINSTORMING: 0.30,
    TaskType.CONVERSION: 0.30,
    TaskType.TEXT_GENERATION: 0.35,
    TaskType.EXPLANATION: 0.40,
    TaskType.COMPARISON: 0.45,
    TaskType.CODE_GENERATION: 0.50,
    TaskType.INSTRUCTION_FOLLOWING: 0.50,
    TaskType.DATA_ANALYSIS: 0.55,
    TaskType.PLANNING: 0.55,
    TaskType.CODE_REVIEW: 0.60,
    TaskType.CODE_DEBUGGING: 0.65,
    TaskType.CODE_REFACTORING: 0.65,
    TaskType.MATH_REASONING: 0.70,
}

DOMAIN_SPECIFICITY_SCORES: dict[Domain, float] = {
    Domain.GENERAL_KNOWLEDGE: 0.05,
    Domain.COOKING: 0.10,
    Domain.TRAVEL: 0.10,
    Domain.FITNESS: 0.15,
    Domain.EDUCATION: 0.20,
    Domain.CREATIVE_WRITING: 0.20,
    Domain.MUSIC: 0.25,
    Domain.HISTORY: 0.25,
    Domain.PHILOSOPHY: 0.30,
    Domain.MARKETING: 0.30,
    Domain.MANAGEMENT: 0.30,
    Domain.LINGUISTICS: 0.35,
    Domain.WEB_DEVELOPMENT: 0.40,
    Domain.ECONOMICS: 0.40,
    Domain.FINANCE: 0.45,
    Domain.LAW: 0.50,
    Domain.DATABASES: 0.50,
    Domain.DATA_SCIENCE: 0.55,
    Domain.BIOLOGY: 0.55,
    Domain.ENVIRONMENTAL_SCIENCE: 0.55,
    Domain.ENERGY: 0.60,
    Domain.STATISTICS: 0.60,
    Domain.ALGORITHMS: 0.65,
    Domain.MACHINE_LEARNING: 0.70,
    Domain.ENGINEERING: 0.70,
    Domain.DEVOPS: 0.70,
    Domain.CHEMISTRY: 0.75,
    Domain.PHYSICS: 0.80,
    Domain.MATHEMATICS: 0.85,
    Domain.SYSTEMS_PROGRAMMING: 0.85,
}

SENSITIVITY_SCORES: dict[SensitivityLevel, float] = {
    SensitivityLevel.BENIGN: 0.0,
    SensitivityLevel.NEUTRAL: 0.25,
    SensitivityLevel.DUAL_USE: 0.55,
    SensitivityLevel.FICTIONAL_EDGE: 0.80,
}

OUTPUT_CONSTRAINT_SCORES: dict[OutputFormat, float] = {
    OutputFormat.FREE_TEXT: 0.0,
    OutputFormat.BULLET_POINTS: 0.15,
    OutputFormat.NUMBERED_LIST: 0.20,
    OutputFormat.MARKDOWN_TABLE: 0.40,
    OutputFormat.CSV: 0.50,
    OutputFormat.STRUCTURED_REPORT: 0.55,
    OutputFormat.JSON: 0.65,
    OutputFormat.CODE_ONLY: 0.70,
    OutputFormat.CODE_WITH_TESTS: 0.90,
}


@dataclass
class PromptDesignVector:
    """The continuous design vector z ∈ [0,1]^d for a single prompt.

    This is the direct analog of the turbine design vector
    z = (rated_power, diameter, hub_height, specific_power) in the paper.
    """

    task_complexity: float = 0.0
    instruction_density: float = 0.0
    persona_depth: float = 0.0
    topic_sensitivity: float = 0.0
    domain_specificity: float = 0.0
    output_constraints: float = 0.0

    def to_array(self) -> list[float]:
        return [
            self.task_complexity,
            self.instruction_density,
            self.persona_depth,
            self.topic_sensitivity,
            self.domain_specificity,
            self.output_constraints,
        ]

    @classmethod
    def dim_names(cls) -> list[str]:
        return list(DESIGN_DIMS.keys())


@dataclass
class PromptRecord:
    """A single prompt with its design vector and metadata."""

    prompt_id: str
    text: str
    design_vector: PromptDesignVector
    # Categorical metadata (for interpretability)
    task_type: str
    domain: str
    complexity: int
    sensitivity: int
    output_format: str
    # Generation metadata
    template_id: str = ""
    campaign_id: str = ""  # Analog of "measurement campaign" index
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "text": self.text,
            "design_vector": self.design_vector.to_array(),
            "task_type": self.task_type,
            "domain": self.domain,
            "complexity": self.complexity,
            "sensitivity": self.sensitivity,
            "output_format": self.output_format,
            "template_id": self.template_id,
            "campaign_id": self.campaign_id,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PromptRecord:
        dv_arr = d["design_vector"]
        dv = PromptDesignVector(
            task_complexity=dv_arr[0],
            instruction_density=dv_arr[1],
            persona_depth=dv_arr[2],
            topic_sensitivity=dv_arr[3],
            domain_specificity=dv_arr[4],
            output_constraints=dv_arr[5],
        )
        return cls(
            prompt_id=d["prompt_id"],
            text=d["text"],
            design_vector=dv,
            task_type=d["task_type"],
            domain=d["domain"],
            complexity=d["complexity"],
            sensitivity=d["sensitivity"],
            output_format=d["output_format"],
            template_id=d.get("template_id", ""),
            campaign_id=d.get("campaign_id", ""),
            tags=d.get("tags", []),
        )
