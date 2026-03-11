"""Compositional prompt generator.

Generates thousands of unique prompts by combining atomic primitives from
the seed pools. Each prompt is assigned a continuous design vector z ∈ [0,1]^d,
making it a point in the "prompt design space" — the analog of the turbine
design space in the error_predictor paper.

The generator produces "evaluation campaigns" — cohorts of prompts that sample
a specific region of design space. This mirrors how each wind turbine prototype
measurement campaign covers a specific turbine design.

Usage:
    python -m sotif_llm.prompts.generator          # Generate full database
    python -m sotif_llm.prompts.generator --n 1000  # Generate 1000 prompts
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from . import primitives as P
from .taxonomy import (
    ComplexityLevel,
    Domain,
    OutputFormat,
    PromptDesignVector,
    PromptRecord,
    SensitivityLevel,
    TaskType,
    TASK_COMPLEXITY_SCORES,
    DOMAIN_SPECIFICITY_SCORES,
    SENSITIVITY_SCORES,
    OUTPUT_CONSTRAINT_SCORES,
)


# Map string keys to enums for lookup
_TASK_KEY_TO_ENUM: dict[str, TaskType] = {
    "code_generation": TaskType.CODE_GENERATION,
    "code_debugging": TaskType.CODE_DEBUGGING,
    "code_refactoring": TaskType.CODE_REFACTORING,
    "code_review": TaskType.CODE_REVIEW,
    "text_summarization": TaskType.TEXT_SUMMARIZATION,
    "text_generation": TaskType.TEXT_GENERATION,
    "explanation": TaskType.EXPLANATION,
    "question_answering": TaskType.QUESTION_ANSWERING,
    "data_analysis": TaskType.DATA_ANALYSIS,
    "math_reasoning": TaskType.MATH_REASONING,
    "instruction_following": TaskType.INSTRUCTION_FOLLOWING,
    "translation": TaskType.TRANSLATION,
    "classification": TaskType.CLASSIFICATION,
    "brainstorming": TaskType.BRAINSTORMING,
    "comparison": TaskType.COMPARISON,
    "planning": TaskType.PLANNING,
    "conversion": TaskType.CONVERSION,
    "editing": TaskType.EDITING,
}


class PromptGenerator:
    """Compositional prompt generator with design-space parameterization.

    Generates prompts in 4 layers:
    1. Task primitive selection (what to do)
    2. Domain/topic overlay (what domain)
    3. Complexity modifiers (how hard/constrained)
    4. Sensitivity gradient (proximity to safety boundary)

    Each layer contributes to the continuous design vector.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._counter = 0
        # Precompute domain -> concept mapping
        self._domain_concepts = self._build_domain_concept_map()

    def _build_domain_concept_map(self) -> dict[str, list[str]]:
        """Map Domain enum values to concept pools."""
        mapping: dict[str, list[str]] = {}
        # Direct mappings
        concept_keys = {
            Domain.ALGORITHMS: "algorithms",
            Domain.WEB_DEVELOPMENT: "web_development",
            Domain.MACHINE_LEARNING: "machine_learning",
            Domain.DATA_SCIENCE: "machine_learning",
            Domain.PHYSICS: "physics",
            Domain.CHEMISTRY: "chemistry",
            Domain.BIOLOGY: "biology",
            Domain.MATHEMATICS: "mathematics",
            Domain.STATISTICS: "mathematics",
            Domain.FINANCE: "finance",
            Domain.ENGINEERING: "engineering",
            Domain.ENERGY: "energy",
        }
        for domain, key in concept_keys.items():
            mapping[domain.name] = P.CONCEPTS.get(key, P.CONCEPTS["general"])
        # Default to general for unmapped domains
        for domain in Domain:
            if domain.name not in mapping:
                mapping[domain.name] = P.CONCEPTS["general"]
        return mapping

    def _make_id(self, text: str) -> str:
        """Deterministic prompt ID from content hash."""
        h = hashlib.sha256(text.encode()).hexdigest()[:12]
        self._counter += 1
        return f"p_{self._counter:05d}_{h}"

    # ------------------------------------------------------------------
    # Slot fillers — each returns a random value from the appropriate pool
    # ------------------------------------------------------------------

    def _fill_slot(self, slot_name: str, domain: Domain, language: str | None = None) -> str:
        """Fill a single template slot with a random value from its pool."""
        lang = language or self.rng.choice(P.LANGUAGES)

        fillers: dict[str, callable] = {
            "language": lambda: lang,
            "simple_task": lambda: self.rng.choice(P.SIMPLE_TASKS),
            "medium_task": lambda: self.rng.choice(P.MEDIUM_TASKS),
            "complex_task": lambda: self.rng.choice(P.COMPLEX_TASKS),
            "advanced_task": lambda: self.rng.choice(P.ADVANCED_TASKS),
            "expert_task": lambda: self.rng.choice(P.EXPERT_TASKS),
            "optimization_target": lambda: self.rng.choice(P.OPTIMIZATION_TARGETS),
            "architecture_pattern": lambda: self.rng.choice(P.ARCHITECTURE_PATTERNS),
            "requirement_1": lambda: P.REQUIREMENTS[self.rng.randint(0, len(P.REQUIREMENTS) - 1)],
            "requirement_2": lambda: P.REQUIREMENTS[self.rng.randint(0, len(P.REQUIREMENTS) - 1)],
            "requirement_3": lambda: P.REQUIREMENTS[self.rng.randint(0, len(P.REQUIREMENTS) - 1)],
            "buggy_snippet": lambda: self.rng.choice(
                P.BUGGY_SNIPPETS.get(lang, P.BUGGY_SNIPPETS["Python"])
            ),
            "refactoring_goal": lambda: self.rng.choice(P.REFACTORING_GOALS),
            "refactor_snippet": lambda: self.rng.choice(
                P.REFACTOR_SNIPPETS.get(lang, P.REFACTOR_SNIPPETS["Python"])
            ),
            "design_pattern": lambda: self.rng.choice(P.DESIGN_PATTERNS),
            "concept": lambda: self.rng.choice(self._domain_concepts.get(domain.name, P.CONCEPTS["general"])),
            "concept_a": lambda: self.rng.choice(self._domain_concepts.get(domain.name, P.CONCEPTS["general"])),
            "concept_b": lambda: self.rng.choice(self._domain_concepts.get(domain.name, P.CONCEPTS["general"])),
            "related_concept": lambda: self.rng.choice(self._domain_concepts.get(domain.name, P.CONCEPTS["general"])),
            "alternative": lambda: self.rng.choice(self._domain_concepts.get(domain.name, P.CONCEPTS["general"])),
            "domain_context": lambda: domain.name.lower().replace("_", " "),
            "application_domain": lambda: domain.name.lower().replace("_", " "),
            "audience": lambda: self.rng.choice(P.AUDIENCES),
            "analogy_domain": lambda: self.rng.choice(["cooking", "sports", "construction", "music", "gardening"]),
            "topic": lambda: self._pick_topic(domain),
            "brainstorm_topic": lambda: self.rng.choice(P.BRAINSTORM_TOPICS),
            "genre": lambda: self.rng.choice(P.GENRES),
            "tone": lambda: self.rng.choice(P.TONES),
            "length_desc": lambda: self.rng.choice(["brief", "500-word", "detailed", "comprehensive"]),
            "length": lambda: str(self.rng.choice([2, 3, 5])),
            "element_1": lambda: self.rng.choice(["a specific example", "a metaphor", "historical context", "data"]),
            "element_2": lambda: self.rng.choice(["a counterargument", "an analogy", "a practical tip", "a quote"]),
            "perspective": lambda: self.rng.choice(P.PERSPECTIVES),
            "passage": lambda: self.rng.choice(P.PASSAGES),
            "summary_style": lambda: self.rng.choice(P.SUMMARY_STYLES),
            "content_type": lambda: self.rng.choice(P.CONTENT_TYPES),
            "focus_area": lambda: self.rng.choice(P.FOCUS_AREAS),
            "data_description": lambda: self.rng.choice(P.DATA_DESCRIPTIONS),
            "statistic": lambda: self.rng.choice(P.STATISTICS),
            "plot_type": lambda: self.rng.choice(P.PLOT_TYPES),
            "analysis_technique": lambda: self.rng.choice(P.ANALYSIS_TECHNIQUES),
            "math_problem": lambda: self.rng.choice(P.MATH_PROBLEMS),
            "math_statement": lambda: self.rng.choice(P.MATH_STATEMENTS),
            "proof_technique": lambda: self.rng.choice(P.PROOF_TECHNIQUES),
            "step_1": lambda: P.INSTRUCTION_STEPS[self.rng.randint(0, len(P.INSTRUCTION_STEPS) - 1)],
            "step_2": lambda: P.INSTRUCTION_STEPS[self.rng.randint(0, len(P.INSTRUCTION_STEPS) - 1)],
            "step_3": lambda: P.INSTRUCTION_STEPS[self.rng.randint(0, len(P.INSTRUCTION_STEPS) - 1)],
            "final_output": lambda: self.rng.choice(["a JSON report", "a markdown document", "a CSV file"]),
            "constraint": lambda: self.rng.choice(P.REQUIREMENTS),
            "source_lang": lambda: self.rng.choice(P.LANGUAGES_NATURAL[:5]),
            "target_lang": lambda: self.rng.choice(P.LANGUAGES_NATURAL[5:]),
            "text_to_translate": lambda: self.rng.choice(P.PASSAGES)[:200],
            "item_type": lambda: self.rng.choice(["requirements", "bug reports", "feature requests", "comments"]),
            "categories": lambda: ", ".join(self.rng.sample(["critical", "high", "medium", "low", "enhancement", "bug", "feature"], 3)),
            "item": lambda: self.rng.choice(["The login page crashes when using special characters in the password field", "Add dark mode support to the dashboard"]),
            "items": lambda: "1. Item A\n2. Item B\n3. Item C",
            "ranking_criterion": lambda: self.rng.choice(["priority", "effort", "impact", "urgency"]),
            "dimension_1": lambda: self.rng.choice(["performance", "ease of use", "scalability"]),
            "dimension_2": lambda: self.rng.choice(["cost", "maintainability", "community support"]),
            "dimension_3": lambda: self.rng.choice(["security", "flexibility", "learning curve"]),
            "tradeoff_dimension": lambda: self.rng.choice(["performance vs readability", "consistency vs availability", "speed vs accuracy"]),
            "plan_type": lambda: self.rng.choice(P.PLAN_TYPES),
            "goal": lambda: self.rng.choice(P.GOALS),
            "count": lambda: str(self.rng.choice([3, 5, 7, 10])),
            "source_format": lambda: self.rng.choice(["JSON", "CSV", "YAML", "XML"]),
            "target_format": lambda: self.rng.choice(["JSON", "CSV", "YAML", "markdown table"]),
            "content": lambda: '{"name": "Alice", "age": 30, "city": "NYC"}',
            "improvement_goal": lambda: self.rng.choice(["clarity", "conciseness", "professionalism", "engagement"]),
            "quality_1": lambda: self.rng.choice(["concise", "precise", "engaging"]),
            "quality_2": lambda: self.rng.choice(["professional", "clear", "impactful"]),
            "draft_text": lambda: self.rng.choice(P.DRAFT_TEXTS),
            "role": lambda: self.rng.choice(P.ROLES),
            "company_type": lambda: self.rng.choice(P.COMPANY_TYPES),
        }

        filler = fillers.get(slot_name)
        if filler is None:
            return f"[{slot_name}]"
        return filler()

    def _pick_topic(self, domain: Domain) -> str:
        """Pick a topic relevant to the domain."""
        domain_topic_keys = {
            Domain.COOKING: "cooking",
            Domain.TRAVEL: "travel",
            Domain.FITNESS: "fitness",
            Domain.CREATIVE_WRITING: "creative_writing",
            Domain.EDUCATION: "education",
        }
        key = domain_topic_keys.get(domain)
        if key and key in P.TOPICS:
            return self.rng.choice(P.TOPICS[key])
        # Fall back to technology or general topics
        all_topics = []
        for topics in P.TOPICS.values():
            all_topics.extend(topics)
        return self.rng.choice(all_topics)

    # ------------------------------------------------------------------
    # Persona layer
    # ------------------------------------------------------------------

    def _apply_persona(self, text: str, depth: float, domain: Domain) -> tuple[str, float]:
        """Wrap a prompt with a persona layer. Returns (wrapped_text, persona_depth_score)."""
        if depth < 0.15:
            return text, 0.0
        elif depth < 0.35:
            prefix = self.rng.choice(P.PERSONAS["light"])
            return f"{prefix}{text}", 0.25
        elif depth < 0.70:
            template = self.rng.choice(P.PERSONAS["moderate"])
            prefix = template.format(
                role=self.rng.choice(P.ROLES),
                company_type=self.rng.choice(P.COMPANY_TYPES),
                audience=self.rng.choice(P.AUDIENCES),
            )
            return f"{prefix}{text}", 0.55
        else:
            template = self.rng.choice(P.PERSONAS["deep"])
            prefix = template.format(
                domain_context=domain.name.lower().replace("_", " "),
            )
            return f"{prefix}{text}", 0.85

    # ------------------------------------------------------------------
    # Format constraint layer
    # ------------------------------------------------------------------

    def _apply_format(self, text: str, output_format: OutputFormat) -> str:
        """Append format constraints to a prompt."""
        if output_format == OutputFormat.FREE_TEXT:
            return text
        fmt_key = output_format.name.lower()
        constraint = P.FORMAT_CONSTRAINTS.get(fmt_key, "")
        if constraint:
            return f"{text}\n\n{constraint}"
        return text

    # ------------------------------------------------------------------
    # Sensitivity layer
    # ------------------------------------------------------------------

    def _apply_sensitivity(self, text: str, level: SensitivityLevel) -> str:
        """Optionally blend in sensitivity-graded content."""
        if level == SensitivityLevel.BENIGN:
            return text
        # For higher sensitivity levels, we prepend context
        topics = P.SENSITIVITY_TOPICS.get(level.value, [])
        if not topics:
            return text
        topic = self.rng.choice(topics)
        if level == SensitivityLevel.NEUTRAL:
            return text  # No modification needed — topic already embedded
        elif level == SensitivityLevel.DUAL_USE:
            return f"For educational purposes, {topic}\n\n{text}"
        else:  # FICTIONAL_EDGE
            return f"{topic}\n\nAdditionally, {text}"

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate_single(
        self,
        task_key: str | None = None,
        domain: Domain | None = None,
        complexity: int | None = None,
        sensitivity: SensitivityLevel | None = None,
        output_format: OutputFormat | None = None,
        persona_depth: float | None = None,
        campaign_id: str = "",
    ) -> PromptRecord:
        """Generate a single prompt with full design vector.

        Any parameter left as None is randomly sampled.
        """
        # Sample unspecified parameters
        if task_key is None:
            task_key = self.rng.choice(list(P.TASK_TEMPLATES.keys()))
        if domain is None:
            domain = self.rng.choice(list(Domain))
        if sensitivity is None:
            sensitivity = self.rng.choice(list(SensitivityLevel))
        if output_format is None:
            output_format = self.rng.choice(list(OutputFormat))
        if persona_depth is None:
            persona_depth = self.rng.random()

        # Pick a template at the desired complexity
        templates = P.TASK_TEMPLATES.get(task_key, P.TASK_TEMPLATES["explanation"])
        if complexity is not None:
            matching = [t for t in templates if t["complexity"] == complexity]
            if not matching:
                matching = templates
        else:
            matching = templates
        template_spec = self.rng.choice(matching)
        actual_complexity = template_spec["complexity"]

        # Choose a language for code tasks
        language = self.rng.choice(P.LANGUAGES) if "code" in task_key else None

        # Fill template slots
        filled_slots = {}
        for slot in template_spec["slots"]:
            filled_slots[slot] = self._fill_slot(slot, domain, language)
        try:
            text = template_spec["template"].format(**filled_slots)
        except KeyError:
            text = template_spec["template"]
        template_id = f"{task_key}_c{actual_complexity}"

        # Apply layers
        text, persona_score = self._apply_persona(text, persona_depth, domain)
        text = self._apply_format(text, output_format)
        text = self._apply_sensitivity(text, sensitivity)

        # Compute design vector
        task_enum = _TASK_KEY_TO_ENUM.get(task_key, TaskType.EXPLANATION)
        base_complexity = TASK_COMPLEXITY_SCORES.get(task_enum, 0.5)
        # Scale by actual template complexity
        complexity_score = max(0.0, min(1.0, base_complexity + (actual_complexity - 3) * 0.1))

        # Instruction density = constraint density proxy
        n_constraints = text.count("\n") + text.count(":") + text.count(".")
        n_words = max(1, len(text.split()))
        instruction_density = min(1.0, n_constraints / (n_words * 0.3))

        dv = PromptDesignVector(
            task_complexity=complexity_score,
            instruction_density=instruction_density,
            persona_depth=persona_score,
            topic_sensitivity=SENSITIVITY_SCORES.get(sensitivity, 0.0),
            domain_specificity=DOMAIN_SPECIFICITY_SCORES.get(domain, 0.3),
            output_constraints=OUTPUT_CONSTRAINT_SCORES.get(output_format, 0.0),
        )

        prompt_id = self._make_id(text)

        return PromptRecord(
            prompt_id=prompt_id,
            text=text,
            design_vector=dv,
            task_type=task_key,
            domain=domain.name,
            complexity=actual_complexity,
            sensitivity=sensitivity.value,
            output_format=output_format.name,
            template_id=template_id,
            campaign_id=campaign_id,
            tags=[],
        )

    def generate_campaign(
        self,
        n: int,
        campaign_id: str,
        *,
        task_key: str | None = None,
        domain: Domain | None = None,
        complexity: int | None = None,
        sensitivity: SensitivityLevel | None = None,
    ) -> list[PromptRecord]:
        """Generate a cohort of related prompts (an evaluation campaign).

        Analogous to a single wind turbine measurement campaign that measures
        a specific turbine design under varying conditions.
        """
        records = []
        seen_texts: set[str] = set()
        attempts = 0
        while len(records) < n and attempts < n * 3:
            attempts += 1
            rec = self.generate_single(
                task_key=task_key,
                domain=domain,
                complexity=complexity,
                sensitivity=sensitivity,
                campaign_id=campaign_id,
            )
            # Deduplicate
            if rec.text not in seen_texts:
                seen_texts.add(rec.text)
                records.append(rec)
        return records

    def generate_benign_database(self, n: int = 5000) -> list[PromptRecord]:
        """Generate the Phase 1 benign prompt database.

        Stratified sampling ensures uniform coverage of the design space:
        - All task types represented
        - All domains represented
        - Complexity levels 1-5 evenly distributed
        - Sensitivity levels 0-1 only (benign / neutral)
        - All output formats represented

        This is the "5,000 standard, harmless prompts" from the scope.
        """
        records: list[PromptRecord] = []
        seen: set[str] = set()

        task_keys = list(P.TASK_TEMPLATES.keys())
        domains = list(Domain)
        benign_sensitivities = [SensitivityLevel.BENIGN, SensitivityLevel.NEUTRAL]
        formats = list(OutputFormat)

        # Stratified generation: cycle through all combinations
        campaign_idx = 0
        attempts = 0
        while len(records) < n and attempts < n * 5:
            attempts += 1
            task_key = task_keys[len(records) % len(task_keys)]
            domain = domains[len(records) % len(domains)]
            sensitivity = benign_sensitivities[len(records) % len(benign_sensitivities)]
            output_format = formats[len(records) % len(formats)]
            persona_depth = (len(records) % 5) / 5.0

            # Assign campaigns (groups of ~20 prompts sharing a region)
            if len(records) % 20 == 0:
                campaign_idx += 1
            campaign_id = f"benign_{campaign_idx:04d}"

            rec = self.generate_single(
                task_key=task_key,
                domain=domain,
                sensitivity=sensitivity,
                output_format=output_format,
                persona_depth=persona_depth,
                campaign_id=campaign_id,
            )
            if rec.text not in seen:
                seen.add(rec.text)
                records.append(rec)

        return records

    def generate_boundary_database(self, n: int = 1000) -> list[PromptRecord]:
        """Generate boundary-testing prompts (sensitivity levels 2-3).

        These prompts are legitimate but closer to the safety boundary.
        Used to test the SOTIF envelope's sensitivity.
        """
        records: list[PromptRecord] = []
        seen: set[str] = set()

        boundary_sensitivities = [SensitivityLevel.DUAL_USE, SensitivityLevel.FICTIONAL_EDGE]

        campaign_idx = 0
        attempts = 0
        while len(records) < n and attempts < n * 5:
            attempts += 1
            sensitivity = boundary_sensitivities[len(records) % len(boundary_sensitivities)]
            if len(records) % 20 == 0:
                campaign_idx += 1

            rec = self.generate_single(
                sensitivity=sensitivity,
                campaign_id=f"boundary_{campaign_idx:04d}",
            )
            if rec.text not in seen:
                seen.add(rec.text)
                records.append(rec)

        return records

    def generate_full_database(
        self, n_benign: int = 5000, n_boundary: int = 1000
    ) -> list[PromptRecord]:
        """Generate the complete prompt database for the SOTIF framework."""
        benign = self.generate_benign_database(n_benign)
        boundary = self.generate_boundary_database(n_boundary)
        return benign + boundary

    @staticmethod
    def save_database(records: list[PromptRecord], path: Path) -> None:
        """Save prompt database to JSONL."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec.to_dict()) + "\n")

    @staticmethod
    def load_database(path: Path) -> list[PromptRecord]:
        """Load prompt database from JSONL."""
        records = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    records.append(PromptRecord.from_dict(json.loads(line)))
        return records

    def stats(self, records: list[PromptRecord]) -> dict:
        """Compute summary statistics of the prompt database."""
        task_counts: dict[str, int] = defaultdict(int)
        domain_counts: dict[str, int] = defaultdict(int)
        complexity_counts: dict[int, int] = defaultdict(int)
        sensitivity_counts: dict[int, int] = defaultdict(int)
        campaign_counts: dict[str, int] = defaultdict(int)

        dv_sums = [0.0] * 6
        for rec in records:
            task_counts[rec.task_type] += 1
            domain_counts[rec.domain] += 1
            complexity_counts[rec.complexity] += 1
            sensitivity_counts[rec.sensitivity] += 1
            campaign_counts[rec.campaign_id] += 1
            for i, v in enumerate(rec.design_vector.to_array()):
                dv_sums[i] += v

        n = len(records)
        dv_means = [s / n for s in dv_sums] if n > 0 else dv_sums

        return {
            "total_prompts": n,
            "unique_campaigns": len(campaign_counts),
            "task_type_distribution": dict(task_counts),
            "domain_distribution": dict(domain_counts),
            "complexity_distribution": dict(complexity_counts),
            "sensitivity_distribution": dict(sensitivity_counts),
            "design_vector_means": dict(zip(PromptDesignVector.dim_names(), dv_means)),
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate SOTIF-LLM prompt database")
    parser.add_argument("--n-benign", type=int, default=5000)
    parser.add_argument("--n-boundary", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/prompts/database.jsonl")
    args = parser.parse_args()

    gen = PromptGenerator(seed=args.seed)
    records = gen.generate_full_database(args.n_benign, args.n_boundary)

    out_path = Path(args.output)
    gen.save_database(records, out_path)

    stats = gen.stats(records)
    print(f"Generated {stats['total_prompts']} prompts in {stats['unique_campaigns']} campaigns")
    print(f"Design vector means: {stats['design_vector_means']}")

    # Also save stats
    stats_path = out_path.parent / "database_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
