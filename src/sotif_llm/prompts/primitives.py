"""Atomic building blocks for compositional prompt generation.

Contains the seed pools of tasks, topics, constraints, personas, and
code snippets that the generator combines to produce thousands of unique prompts.

Architecture inspired by the error_predictor paper's approach: just as each
wind turbine measurement campaign explores a specific region of design space
(rated power × diameter × hub height), each prompt "campaign" is a cohort
of related prompts exploring a specific region of prompt design space.
"""

from __future__ import annotations

# ============================================================================
# TASK TEMPLATES
# ============================================================================
# Each template is a format string. The generator fills slots from the pools below.
# Templates are grouped by TaskType and ComplexityLevel (1-5).

TASK_TEMPLATES: dict[str, list[dict]] = {
    # ------------------------------------------------------------------
    # CODE GENERATION
    # ------------------------------------------------------------------
    "code_generation": [
        {
            "complexity": 1,
            "template": "Write a {language} function that {simple_task}.",
            "slots": ["language", "simple_task"],
        },
        {
            "complexity": 2,
            "template": "Write a {language} function that {medium_task}. Include docstrings.",
            "slots": ["language", "medium_task"],
        },
        {
            "complexity": 3,
            "template": (
                "Write a {language} module that {complex_task}. "
                "The code should handle edge cases and include type hints."
            ),
            "slots": ["language", "complex_task"],
        },
        {
            "complexity": 4,
            "template": (
                "Implement a {language} class that {advanced_task}. "
                "Include proper error handling, logging, and unit tests. "
                "Optimize for {optimization_target}."
            ),
            "slots": ["language", "advanced_task", "optimization_target"],
        },
        {
            "complexity": 5,
            "template": (
                "Design and implement a {language} {architecture_pattern} for {expert_task}. "
                "Requirements:\n"
                "1. {requirement_1}\n"
                "2. {requirement_2}\n"
                "3. {requirement_3}\n"
                "Include comprehensive tests and documentation."
            ),
            "slots": [
                "language", "architecture_pattern", "expert_task",
                "requirement_1", "requirement_2", "requirement_3",
            ],
        },
    ],
    # ------------------------------------------------------------------
    # CODE DEBUGGING
    # ------------------------------------------------------------------
    "code_debugging": [
        {
            "complexity": 2,
            "template": (
                "Find and fix the bug in this {language} code:\n```{language}\n{buggy_snippet}\n```"
            ),
            "slots": ["language", "buggy_snippet"],
        },
        {
            "complexity": 4,
            "template": (
                "The following {language} code produces incorrect results for edge cases. "
                "Identify all bugs, explain why they occur, and provide the corrected version:\n"
                "```{language}\n{buggy_snippet}\n```"
            ),
            "slots": ["language", "buggy_snippet"],
        },
    ],
    # ------------------------------------------------------------------
    # CODE REFACTORING
    # ------------------------------------------------------------------
    "code_refactoring": [
        {
            "complexity": 3,
            "template": (
                "Refactor the following {language} code to {refactoring_goal}:\n"
                "```{language}\n{refactor_snippet}\n```"
            ),
            "slots": ["language", "refactoring_goal", "refactor_snippet"],
        },
        {
            "complexity": 4,
            "template": (
                "Refactor this {language} code to follow {design_pattern} pattern. "
                "Explain your design decisions:\n"
                "```{language}\n{refactor_snippet}\n```"
            ),
            "slots": ["language", "design_pattern", "refactor_snippet"],
        },
    ],
    # ------------------------------------------------------------------
    # TEXT SUMMARIZATION
    # ------------------------------------------------------------------
    "text_summarization": [
        {
            "complexity": 1,
            "template": "Summarize the following in {length} sentences:\n\n{passage}",
            "slots": ["length", "passage"],
        },
        {
            "complexity": 3,
            "template": (
                "Provide a {summary_style} summary of the following {content_type}. "
                "Focus on {focus_area}:\n\n{passage}"
            ),
            "slots": ["summary_style", "content_type", "focus_area", "passage"],
        },
    ],
    # ------------------------------------------------------------------
    # TEXT GENERATION
    # ------------------------------------------------------------------
    "text_generation": [
        {
            "complexity": 1,
            "template": "Write a short {genre} about {topic}.",
            "slots": ["genre", "topic"],
        },
        {
            "complexity": 2,
            "template": "Write a {length_desc} {genre} about {topic} in a {tone} tone.",
            "slots": ["length_desc", "genre", "topic", "tone"],
        },
        {
            "complexity": 3,
            "template": (
                "Write a {genre} about {topic} that incorporates {element_1} "
                "and {element_2}. Target audience: {audience}."
            ),
            "slots": ["genre", "topic", "element_1", "element_2", "audience"],
        },
        {
            "complexity": 4,
            "template": (
                "Write a {length_desc} {genre} about {topic}. "
                "Requirements:\n"
                "- Tone: {tone}\n"
                "- Must include: {element_1}, {element_2}\n"
                "- Perspective: {perspective}\n"
                "- Target audience: {audience}"
            ),
            "slots": [
                "length_desc", "genre", "topic", "tone",
                "element_1", "element_2", "perspective", "audience",
            ],
        },
    ],
    # ------------------------------------------------------------------
    # EXPLANATION
    # ------------------------------------------------------------------
    "explanation": [
        {
            "complexity": 1,
            "template": "Explain {concept} in simple terms.",
            "slots": ["concept"],
        },
        {
            "complexity": 2,
            "template": "Explain {concept} to a {audience}. Use {analogy_domain} analogies.",
            "slots": ["concept", "audience", "analogy_domain"],
        },
        {
            "complexity": 3,
            "template": (
                "Explain {concept} in detail. Cover:\n"
                "1. The core principle\n"
                "2. How it relates to {related_concept}\n"
                "3. A practical example from {application_domain}"
            ),
            "slots": ["concept", "related_concept", "application_domain"],
        },
        {
            "complexity": 5,
            "template": (
                "Provide a comprehensive technical explanation of {concept}. "
                "Include the mathematical formulation, compare it with {alternative}, "
                "discuss its limitations, and provide a worked example in {application_domain}."
            ),
            "slots": ["concept", "alternative", "application_domain"],
        },
    ],
    # ------------------------------------------------------------------
    # QUESTION ANSWERING
    # ------------------------------------------------------------------
    "question_answering": [
        {
            "complexity": 1,
            "template": "What is {concept}?",
            "slots": ["concept"],
        },
        {
            "complexity": 2,
            "template": "How does {concept} work in the context of {domain_context}?",
            "slots": ["concept", "domain_context"],
        },
        {
            "complexity": 3,
            "template": "What are the key differences between {concept_a} and {concept_b} in {domain_context}?",
            "slots": ["concept_a", "concept_b", "domain_context"],
        },
        {
            "complexity": 4,
            "template": (
                "In {domain_context}, when would you choose {concept_a} over {concept_b}? "
                "Consider trade-offs in {tradeoff_dimension}."
            ),
            "slots": ["domain_context", "concept_a", "concept_b", "tradeoff_dimension"],
        },
    ],
    # ------------------------------------------------------------------
    # DATA ANALYSIS
    # ------------------------------------------------------------------
    "data_analysis": [
        {
            "complexity": 2,
            "template": "Given a dataset of {data_description}, calculate the {statistic}.",
            "slots": ["data_description", "statistic"],
        },
        {
            "complexity": 3,
            "template": (
                "Write a {language} script to analyze {data_description}. "
                "Compute {statistic} and create a {plot_type} visualization."
            ),
            "slots": ["language", "data_description", "statistic", "plot_type"],
        },
        {
            "complexity": 4,
            "template": (
                "Perform an exploratory data analysis of {data_description}. "
                "Include:\n"
                "1. Descriptive statistics\n"
                "2. {analysis_technique} analysis\n"
                "3. Visualization of key findings\n"
                "4. Actionable insights"
            ),
            "slots": ["data_description", "analysis_technique"],
        },
    ],
    # ------------------------------------------------------------------
    # MATH REASONING
    # ------------------------------------------------------------------
    "math_reasoning": [
        {
            "complexity": 2,
            "template": "Solve the following: {math_problem}. Show your work step by step.",
            "slots": ["math_problem"],
        },
        {
            "complexity": 4,
            "template": (
                "Prove that {math_statement}. "
                "Use {proof_technique} and explain each step."
            ),
            "slots": ["math_statement", "proof_technique"],
        },
    ],
    # ------------------------------------------------------------------
    # INSTRUCTION FOLLOWING
    # ------------------------------------------------------------------
    "instruction_following": [
        {
            "complexity": 2,
            "template": (
                "Follow these instructions exactly:\n"
                "1. {step_1}\n"
                "2. {step_2}\n"
                "3. {step_3}"
            ),
            "slots": ["step_1", "step_2", "step_3"],
        },
        {
            "complexity": 4,
            "template": (
                "Complete the following multi-step task:\n"
                "1. {step_1}\n"
                "2. Using the result from step 1, {step_2}\n"
                "3. {step_3}\n"
                "4. Combine everything into {final_output}\n"
                "Constraints: {constraint}"
            ),
            "slots": ["step_1", "step_2", "step_3", "final_output", "constraint"],
        },
    ],
    # ------------------------------------------------------------------
    # TRANSLATION
    # ------------------------------------------------------------------
    "translation": [
        {
            "complexity": 1,
            "template": "Translate the following {content_type} from {source_lang} to {target_lang}:\n\n{text_to_translate}",
            "slots": ["content_type", "source_lang", "target_lang", "text_to_translate"],
        },
    ],
    # ------------------------------------------------------------------
    # CLASSIFICATION
    # ------------------------------------------------------------------
    "classification": [
        {
            "complexity": 1,
            "template": "Classify the following {item_type} into one of these categories: {categories}.\n\nItem: {item}",
            "slots": ["item_type", "categories", "item"],
        },
        {
            "complexity": 3,
            "template": (
                "Classify and rank the following {item_type} by {ranking_criterion}. "
                "Explain your reasoning for each classification.\n\n{items}"
            ),
            "slots": ["item_type", "ranking_criterion", "items"],
        },
    ],
    # ------------------------------------------------------------------
    # BRAINSTORMING
    # ------------------------------------------------------------------
    "brainstorming": [
        {
            "complexity": 2,
            "template": "Generate {count} creative ideas for {brainstorm_topic}.",
            "slots": ["count", "brainstorm_topic"],
        },
        {
            "complexity": 3,
            "template": (
                "Brainstorm {count} innovative approaches to {brainstorm_topic}. "
                "For each idea, provide a brief description and assess feasibility."
            ),
            "slots": ["count", "brainstorm_topic"],
        },
    ],
    # ------------------------------------------------------------------
    # COMPARISON
    # ------------------------------------------------------------------
    "comparison": [
        {
            "complexity": 2,
            "template": "Compare {concept_a} and {concept_b}.",
            "slots": ["concept_a", "concept_b"],
        },
        {
            "complexity": 3,
            "template": (
                "Compare and contrast {concept_a} and {concept_b} in terms of "
                "{dimension_1}, {dimension_2}, and {dimension_3}."
            ),
            "slots": ["concept_a", "concept_b", "dimension_1", "dimension_2", "dimension_3"],
        },
    ],
    # ------------------------------------------------------------------
    # PLANNING
    # ------------------------------------------------------------------
    "planning": [
        {
            "complexity": 3,
            "template": "Create a {plan_type} plan for {goal}. Include milestones and deliverables.",
            "slots": ["plan_type", "goal"],
        },
        {
            "complexity": 4,
            "template": (
                "Design a comprehensive {plan_type} plan for {goal}. Consider:\n"
                "- Timeline and milestones\n"
                "- Resource requirements\n"
                "- Risk mitigation\n"
                "- Success metrics"
            ),
            "slots": ["plan_type", "goal"],
        },
    ],
    # ------------------------------------------------------------------
    # CONVERSION
    # ------------------------------------------------------------------
    "conversion": [
        {
            "complexity": 2,
            "template": "Convert the following {source_format} to {target_format}:\n\n{content}",
            "slots": ["source_format", "target_format", "content"],
        },
    ],
    # ------------------------------------------------------------------
    # EDITING
    # ------------------------------------------------------------------
    "editing": [
        {
            "complexity": 2,
            "template": "Improve the following {content_type} for {improvement_goal}:\n\n{draft_text}",
            "slots": ["content_type", "improvement_goal", "draft_text"],
        },
        {
            "complexity": 3,
            "template": (
                "Edit the following {content_type} to be more {quality_1} and {quality_2}. "
                "Maintain the original meaning:\n\n{draft_text}"
            ),
            "slots": ["content_type", "quality_1", "quality_2", "draft_text"],
        },
    ],
}

# ============================================================================
# SLOT FILL POOLS — The atomic building blocks
# ============================================================================

LANGUAGES = [
    "Python", "JavaScript", "TypeScript", "Java", "C++", "C", "Rust",
    "Go", "Ruby", "Swift", "Kotlin", "Scala", "R", "Julia", "SQL",
    "Bash", "PHP", "Haskell", "Lua", "Dart",
]

SIMPLE_TASKS = [
    "computes the factorial of a number",
    "checks if a string is a palindrome",
    "reverses a linked list",
    "finds the maximum element in a list",
    "counts the number of vowels in a string",
    "converts Celsius to Fahrenheit",
    "calculates the area of a circle given its radius",
    "checks if a number is prime",
    "generates the Fibonacci sequence up to n terms",
    "removes duplicate elements from a list",
    "sorts a list of strings alphabetically",
    "calculates the GCD of two numbers",
    "flattens a nested list",
    "counts word frequency in a string",
    "validates an email address format",
    "converts a decimal number to binary",
    "finds the intersection of two lists",
    "implements a basic stack data structure",
    "rotates elements of a list by k positions",
    "checks if two strings are anagrams",
]

MEDIUM_TASKS = [
    "implements a binary search tree with insert and search operations",
    "parses a CSV file and computes column statistics",
    "implements a LRU cache with O(1) operations",
    "creates a simple HTTP request handler",
    "implements a producer-consumer queue with threading",
    "builds a trie for autocomplete suggestions",
    "implements a graph BFS and DFS traversal",
    "creates a simple command-line argument parser",
    "implements a rate limiter using the token bucket algorithm",
    "builds a simple expression evaluator for arithmetic",
    "implements merge sort with both recursive and iterative approaches",
    "creates a connection pool manager",
    "implements a simple state machine",
    "builds a priority queue using a heap",
    "implements a retry decorator with exponential backoff",
]

COMPLEX_TASKS = [
    "implements a concurrent web scraper with rate limiting and error recovery",
    "builds a simple key-value store with persistence and transactions",
    "implements the A* pathfinding algorithm with configurable heuristics",
    "creates a data pipeline that reads, transforms, and validates CSV data",
    "implements a text search engine with TF-IDF ranking",
    "builds a simple REST API with CRUD operations and input validation",
    "implements a B-tree with insert, search, and delete operations",
    "creates a job scheduler with dependency resolution",
    "implements a streaming data processor with windowed aggregation",
    "builds a simple template engine with variable substitution and loops",
]

ADVANCED_TASKS = [
    "implements a distributed consensus protocol (simplified Raft)",
    "builds a query optimizer for a simple SQL-like language",
    "implements a garbage collector using mark-and-sweep",
    "creates a compiler frontend (lexer + parser) for a simple expression language",
    "implements a database index using a B+ tree with range queries",
    "builds a reactive data-binding framework",
    "implements a lock-free concurrent hash map",
    "creates a simple virtual machine with a custom instruction set",
    "implements an incremental parser for a configuration language",
    "builds a real-time event processing engine with pattern matching",
]

EXPERT_TASKS = [
    "a distributed task queue with exactly-once delivery guarantees",
    "a query planner for join ordering optimization",
    "a type inference engine for a simple functional language",
    "a memory allocator with compaction and defragmentation",
    "a conflict-free replicated data type (CRDT) library",
]

OPTIMIZATION_TARGETS = [
    "memory efficiency", "throughput", "latency", "readability",
    "testability", "thread safety", "cache locality", "minimal allocations",
]

ARCHITECTURE_PATTERNS = [
    "microservice", "plugin architecture", "event-driven",
    "repository pattern", "CQRS", "pipeline", "observer pattern",
    "strategy pattern", "decorator pattern", "adapter pattern",
]

REQUIREMENTS = [
    "Support concurrent access from multiple threads",
    "Handle graceful shutdown with cleanup",
    "Include comprehensive error handling with custom exceptions",
    "Implement structured logging",
    "Support configuration via environment variables",
    "Include performance benchmarks",
    "Provide a clean public API with private implementation details",
    "Support serialization/deserialization",
    "Include input validation with clear error messages",
    "Implement retry logic for transient failures",
    "Support pagination for large result sets",
    "Include rate limiting",
    "Provide metrics collection hooks",
    "Support pluggable backends",
    "Implement caching with configurable TTL",
]

# Code snippets for debugging/refactoring tasks
BUGGY_SNIPPETS = {
    "Python": [
        (
            "def merge_sorted(a, b):\n"
            "    result = []\n"
            "    i = j = 0\n"
            "    while i < len(a) and j < len(b):\n"
            "        if a[i] <= b[j]:\n"
            "            result.append(a[i])\n"
            "            i += 1\n"
            "        else:\n"
            "            result.append(b[j])\n"
            "            j += 1\n"
            "    return result  # Bug: doesn't append remaining elements"
        ),
        (
            "def flatten(lst):\n"
            "    result = []\n"
            "    for item in lst:\n"
            "        if type(item) == list:\n"
            "            result.extend(flatten(item))\n"
            "        result.append(item)  # Bug: appends even when item is a list\n"
            "    return result"
        ),
        (
            "class LRUCache:\n"
            "    def __init__(self, capacity):\n"
            "        self.cache = {}\n"
            "        self.capacity = capacity\n"
            "    def get(self, key):\n"
            "        return self.cache.get(key, -1)\n"
            "    def put(self, key, value):\n"
            "        self.cache[key] = value\n"
            "        if len(self.cache) > self.capacity:\n"
            "            del self.cache[list(self.cache.keys())[0]]  # Bug: not tracking access order"
        ),
        (
            "def binary_search(arr, target):\n"
            "    left, right = 0, len(arr)\n"
            "    while left < right:\n"
            "        mid = (left + right) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            left = mid  # Bug: should be mid + 1\n"
            "        else:\n"
            "            right = mid\n"
            "    return -1"
        ),
    ],
    "JavaScript": [
        (
            "function debounce(fn, delay) {\n"
            "  let timer;\n"
            "  return function() {\n"
            "    clearTimeout(timer);\n"
            "    timer = setTimeout(fn, delay);  // Bug: loses `this` and arguments\n"
            "  };\n"
            "}"
        ),
        (
            "async function fetchAll(urls) {\n"
            "  const results = [];\n"
            "  for (const url of urls) {\n"
            "    const resp = await fetch(url);\n"
            "    results.push(resp.json());  // Bug: missing await on .json()\n"
            "  }\n"
            "  return results;\n"
            "}"
        ),
    ],
}

REFACTORING_GOALS = [
    "improve readability and follow PEP 8",
    "reduce code duplication",
    "use more idiomatic language features",
    "separate concerns into distinct functions",
    "add proper error handling",
    "make it more testable",
    "improve performance",
    "use dataclasses instead of raw dictionaries",
    "convert to async/await",
    "extract a reusable library from the script",
]

REFACTOR_SNIPPETS = {
    "Python": [
        (
            "def process(data):\n"
            "    results = []\n"
            "    for item in data:\n"
            "        if item['type'] == 'A':\n"
            "            val = item['value'] * 2 + 10\n"
            "            if val > 100:\n"
            "                val = 100\n"
            "            results.append({'name': item['name'], 'result': val})\n"
            "        elif item['type'] == 'B':\n"
            "            val = item['value'] ** 0.5\n"
            "            if val > 100:\n"
            "                val = 100\n"
            "            results.append({'name': item['name'], 'result': val})\n"
            "        elif item['type'] == 'C':\n"
            "            val = item['value'] / 3.14\n"
            "            if val > 100:\n"
            "                val = 100\n"
            "            results.append({'name': item['name'], 'result': val})\n"
            "    return results"
        ),
        (
            "import json, os, sys\n"
            "def load_and_process(path):\n"
            "    f = open(path)\n"
            "    data = json.load(f)\n"
            "    out = []\n"
            "    for d in data:\n"
            "        try:\n"
            "            x = d['x']\n"
            "            y = d['y']\n"
            "            z = (x**2 + y**2)**0.5\n"
            "            out.append(z)\n"
            "        except:\n"
            "            pass\n"
            "    f.close()\n"
            "    return out"
        ),
    ],
}

DESIGN_PATTERNS = [
    "Strategy", "Observer", "Factory", "Singleton", "Decorator",
    "Builder", "Command", "Template Method", "Iterator", "Adapter",
]

# ============================================================================
# TEXT / KNOWLEDGE DOMAIN POOLS
# ============================================================================

CONCEPTS: dict[str, list[str]] = {
    "algorithms": [
        "dynamic programming", "divide and conquer", "greedy algorithms",
        "graph traversal", "sorting algorithms", "hash tables",
        "balanced binary trees", "amortized analysis", "memoization",
    ],
    "web_development": [
        "REST APIs", "WebSocket protocol", "OAuth 2.0",
        "server-side rendering", "content delivery networks",
        "database indexing", "caching strategies", "CORS",
    ],
    "machine_learning": [
        "gradient descent", "backpropagation", "regularization",
        "cross-validation", "feature engineering", "ensemble methods",
        "transformer architecture", "attention mechanisms",
    ],
    "physics": [
        "Newton's laws of motion", "thermodynamic entropy",
        "Maxwell's equations", "quantum superposition",
        "special relativity", "wave-particle duality",
    ],
    "chemistry": [
        "chemical bonding", "reaction kinetics", "acid-base equilibrium",
        "organic synthesis", "electrochemistry", "thermochemistry",
    ],
    "biology": [
        "DNA replication", "natural selection", "cellular respiration",
        "protein folding", "CRISPR gene editing", "neural plasticity",
    ],
    "mathematics": [
        "group theory", "linear algebra", "Fourier transforms",
        "probability distributions", "differential equations",
        "topology basics", "number theory", "combinatorics",
    ],
    "finance": [
        "compound interest", "portfolio diversification",
        "options pricing", "risk-adjusted returns",
        "time value of money", "capital asset pricing model",
    ],
    "engineering": [
        "finite element analysis", "control systems",
        "signal processing", "fluid dynamics", "materials science",
        "structural load analysis", "thermodynamic cycles",
    ],
    "energy": [
        "wind turbine aerodynamics", "photovoltaic cell efficiency",
        "battery storage technology", "grid stability",
        "power curve modeling", "wake effects in wind farms",
    ],
    "general": [
        "the water cycle", "photosynthesis", "supply and demand",
        "the scientific method", "climate change", "the internet",
        "artificial intelligence", "renewable energy",
    ],
}

TOPICS = {
    "cooking": [
        "a traditional Italian pasta dish", "a plant-based protein bowl",
        "a French pastry technique", "a spice-blending guide for curries",
        "a meal prep strategy for the week", "fermentation basics",
    ],
    "travel": [
        "a weekend itinerary for a European city",
        "budget travel tips for Southeast Asia",
        "the best hiking trails in national parks",
        "how to plan a road trip across a new country",
    ],
    "fitness": [
        "a beginner strength training program",
        "the science of muscle recovery",
        "how to design a marathon training plan",
        "nutrition strategies for endurance athletes",
    ],
    "creative_writing": [
        "a day in the life of a lighthouse keeper",
        "two old friends reuniting after decades",
        "discovering a hidden room in a new house",
        "the first contact between humans and an alien species",
        "a detective solving a mystery in a small town",
        "a scientist making an unexpected discovery",
    ],
    "education": [
        "how to teach fractions to 4th graders",
        "designing an engaging lecture on climate science",
        "creating a project-based learning curriculum",
        "strategies for differentiated instruction",
    ],
    "technology": [
        "how modern CPUs execute instructions",
        "the evolution of internet protocols",
        "how GPS satellites determine position",
        "how databases maintain ACID properties",
    ],
}

AUDIENCES = [
    "a 10-year-old", "a high school student", "a college freshman",
    "a non-technical manager", "a junior developer", "a senior engineer",
    "a domain expert", "a general audience", "a technical interviewer",
]

TONES = [
    "formal", "casual", "technical", "conversational", "academic",
    "enthusiastic", "neutral", "humorous", "professional",
]

GENRES = [
    "short story", "essay", "blog post", "tutorial",
    "product description", "news article", "email",
    "technical report", "letter of recommendation",
    "executive summary", "proposal", "FAQ document",
]

PERSPECTIVES = [
    "first person", "third person limited", "third person omniscient",
    "second person instructional",
]

SUMMARY_STYLES = [
    "concise", "detailed", "executive", "technical", "narrative",
]

CONTENT_TYPES = [
    "article", "research paper abstract", "meeting transcript",
    "product review", "technical documentation", "news report",
]

FOCUS_AREAS = [
    "key findings", "practical implications", "methodology",
    "main arguments", "actionable recommendations",
]

MATH_PROBLEMS = [
    "∫(x²·sin(x))dx from 0 to π",
    "the eigenvalues of the matrix [[2,1],[1,3]]",
    "whether the series Σ(1/n·ln(n)) from n=2 to ∞ converges",
    "the general solution to y'' + 4y = cos(2x)",
    "the number of ways to partition 10 into at most 4 parts",
    "the limit of (1 + 1/n)^(n²) as n → ∞",
    "the determinant of a 3×3 matrix using cofactor expansion",
    "P(X > 2) where X ~ Poisson(λ=3)",
]

MATH_STATEMENTS = [
    "every continuous function on [a,b] is Riemann integrable",
    "the square root of 2 is irrational",
    "for every ε > 0, there exists N such that |a_n - L| < ε for all n > N implies lim a_n = L",
    "the sum of the first n odd numbers equals n²",
    "a finite group of prime order is cyclic",
]

PROOF_TECHNIQUES = [
    "mathematical induction", "proof by contradiction",
    "direct proof", "the pigeonhole principle",
    "proof by contrapositive", "a constructive proof",
]

DATA_DESCRIPTIONS = [
    "monthly sales figures for 12 months across 5 product categories",
    "daily temperature readings from 3 weather stations over a year",
    "student exam scores across 4 subjects for a class of 30",
    "website traffic data (page views, unique visitors, bounce rate) for 6 months",
    "sensor readings from an IoT device (temperature, humidity, pressure) over 24 hours",
    "stock price data (open, close, high, low, volume) for 10 companies over 30 days",
]

STATISTICS = [
    "mean, median, and standard deviation",
    "correlation coefficients between all variable pairs",
    "a linear regression model with R² and residual analysis",
    "descriptive statistics and outlier detection",
    "confidence intervals for the mean at 95% confidence",
    "a hypothesis test for the difference between two groups",
]

ANALYSIS_TECHNIQUES = [
    "correlation", "regression", "clustering", "time series",
    "principal component", "anomaly detection", "survival",
]

PLOT_TYPES = [
    "scatter plot", "histogram", "box plot", "heatmap",
    "line chart", "bar chart", "violin plot",
]

BRAINSTORM_TOPICS = [
    "reducing food waste in restaurants",
    "improving remote team collaboration",
    "making public transportation more efficient",
    "teaching programming to children",
    "reducing energy consumption in office buildings",
    "improving patient experience in hospitals",
    "making scientific research more reproducible",
    "reducing the environmental impact of software development",
]

PLAN_TYPES = [
    "project", "marketing", "migration", "testing",
    "deployment", "training", "research", "product launch",
]

GOALS = [
    "migrating a monolithic application to microservices",
    "launching a new mobile app in 3 months",
    "implementing a company-wide data governance program",
    "setting up a CI/CD pipeline for a legacy codebase",
    "creating a developer onboarding program",
    "building a data warehouse from scratch",
]

# Instruction-following step pools
INSTRUCTION_STEPS = [
    "Read the following data and extract all numeric values",
    "Sort the items by the second column in descending order",
    "Filter out any entries where the value is less than the median",
    "Group the remaining items by category",
    "Calculate the running average across the groups",
    "Format the output as a markdown table",
    "Add a summary row at the bottom with totals",
    "Validate that all values are within the expected range [0, 100]",
    "Create a brief one-paragraph interpretation of the results",
    "Export the final result as a JSON object",
]

# Translation pools
LANGUAGES_NATURAL = [
    "English", "Spanish", "French", "German", "Italian",
    "Portuguese", "Japanese", "Chinese", "Korean", "Arabic",
]

# ============================================================================
# SENSITIVITY-GRADED TOPICS
# ============================================================================
# Level 0: Completely benign
# Level 1: Neutral professional
# Level 2: Dual-use (legitimate but requires care)
# Level 3: Fictional/educational edge

SENSITIVITY_TOPICS: dict[int, list[str]] = {
    0: [
        "how to bake a chocolate cake",
        "the life cycle of butterflies",
        "how to organize a bookshelf",
        "the history of the printing press",
        "how to start a vegetable garden",
        "basic knitting patterns for beginners",
        "how to plan a birthday party",
        "the rules of chess",
        "how bicycles work",
        "favorite board games for families",
    ],
    1: [
        "how to negotiate a salary raise",
        "conflict resolution in the workplace",
        "how to write a persuasive argument",
        "the economics of supply chain management",
        "performance review best practices",
        "how to give constructive criticism",
        "managing competitive dynamics in business",
        "interview techniques for hiring managers",
    ],
    2: [
        "how encryption protects data in transit",
        "the chemistry of common household cleaning products",
        "how vulnerability scanners identify software weaknesses",
        "the pharmacology of common over-the-counter medications",
        "network security fundamentals and firewall configuration",
        "how forensic scientists analyze evidence",
        "the physics of controlled demolition",
        "how penetration testers assess system security",
    ],
    3: [
        "write a thriller scene where a detective analyzes a crime scene",
        "create a fictional heist scenario for a tabletop RPG",
        "describe a cybersecurity CTF challenge involving a simulated breach",
        "write educational material about historical code-breaking techniques",
        "create a fictional scenario for a disaster response training exercise",
        "write a scene from a spy novel involving counter-intelligence",
        "describe a fictional debate between a prosecutor and defense attorney",
        "write a training scenario for emergency medical responders",
    ],
}

# ============================================================================
# PERSONA TEMPLATES
# ============================================================================

PERSONAS: dict[str, list[str]] = {
    "none": [""],  # No persona — direct instruction
    "light": [
        "You are a helpful assistant.",
        "As a knowledgeable tutor, ",
        "Acting as a senior developer, ",
        "As an experienced data scientist, ",
    ],
    "moderate": [
        "You are a {role} at a {company_type} company. A colleague asks you to ",
        "You are a {role} preparing materials for a {audience}. ",
        "You are a {role} reviewing a junior team member's work. ",
    ],
    "deep": [
        (
            "You are Dr. Sarah Chen, a computational physicist at a national lab. "
            "You're known for clear explanations and rigorous methodology. "
            "A graduate student asks you to "
        ),
        (
            "You are Marcus, a staff software engineer with 15 years of experience "
            "in distributed systems. You value simplicity and correctness. "
            "During a code review, you need to "
        ),
        (
            "You are Professor Amara, who teaches an advanced course on {domain_context}. "
            "You are preparing lecture notes that "
        ),
    ],
}

ROLES = [
    "software engineer", "data analyst", "product manager",
    "research scientist", "technical writer", "DevOps engineer",
    "security analyst", "UX designer", "machine learning engineer",
    "database administrator",
]

COMPANY_TYPES = [
    "technology startup", "Fortune 500", "consulting",
    "fintech", "healthcare", "e-commerce", "SaaS",
]

# ============================================================================
# FORMAT CONSTRAINT TEMPLATES
# ============================================================================

FORMAT_CONSTRAINTS: dict[str, str] = {
    "json": "Output your response as a valid JSON object.",
    "csv": "Format your response as CSV with a header row.",
    "markdown_table": "Present the results in a markdown table.",
    "bullet_points": "Use bullet points for your response.",
    "numbered_list": "Provide a numbered list.",
    "code_only": "Respond with code only, no explanations.",
    "code_with_tests": "Include the implementation and comprehensive unit tests.",
    "structured_report": (
        "Structure your response with these sections: "
        "Summary, Analysis, Findings, Recommendations."
    ),
}

# ============================================================================
# PASSAGE POOL for summarization / editing tasks
# ============================================================================

PASSAGES = [
    (
        "The development of renewable energy sources has accelerated dramatically "
        "in the past decade. Solar panel costs have dropped by over 89% since 2010, "
        "while wind energy capacity has tripled globally. Despite these advances, "
        "energy storage remains a critical bottleneck. Current lithium-ion batteries "
        "can store energy for hours, but grid-scale storage for days or weeks "
        "requires new technologies such as flow batteries, compressed air storage, "
        "or green hydrogen. The intermittent nature of solar and wind means that "
        "without adequate storage, fossil fuel plants must remain on standby, "
        "undermining the emission reduction goals of the energy transition."
    ),
    (
        "Machine learning models in healthcare face unique challenges compared to "
        "other domains. First, medical data is often imbalanced — rare diseases have "
        "few positive examples. Second, interpretability is critical: clinicians need "
        "to understand why a model made a particular prediction before acting on it. "
        "Third, data privacy regulations like HIPAA strictly limit how patient data "
        "can be collected, stored, and shared. Federated learning has emerged as a "
        "promising approach that trains models across multiple hospitals without "
        "sharing raw patient data, but it introduces new challenges around model "
        "convergence and communication efficiency."
    ),
    (
        "Urban planning in the 21st century must balance competing demands: "
        "population density for efficient public transit, green spaces for mental "
        "health and biodiversity, affordable housing near employment centers, and "
        "infrastructure resilience against climate change. The concept of the "
        "'15-minute city' proposes that all essential services should be accessible "
        "within a 15-minute walk or bike ride. Critics argue this vision works "
        "mainly for wealthy, dense urban cores and fails to address the needs of "
        "suburban and rural populations."
    ),
    (
        "The debate over programming language choice in education reflects deeper "
        "pedagogical philosophies. Python's popularity in introductory courses stems "
        "from its readable syntax and vast library ecosystem, but critics argue it "
        "hides important concepts like memory management and type systems. Languages "
        "like Rust force students to confront ownership and borrowing, while "
        "functional languages like Haskell emphasize mathematical reasoning. "
        "There is growing consensus that exposure to multiple paradigms produces "
        "more versatile programmers than deep expertise in a single language."
    ),
    (
        "Wind turbine type certification validates that specific turbine variants "
        "comply with international standards through comprehensive evaluation, "
        "including model validation where computational models must be compared "
        "with measurement campaigns to demonstrate sufficient accuracy. For this "
        "purpose, fully instrumented turbines are built and set for operation in "
        "a measurement campaign. Small changes in the turbine design are allowed "
        "without additional testing campaigns being required, but the selection of "
        "these tolerance levels is not based on quantitative analysis."
    ),
]

DRAFT_TEXTS = [
    (
        "The new software is very good and works well. It has lots of features "
        "that users like. The interface is nice and easy to use. We think it will "
        "be popular because its good."
    ),
    (
        "Our quarterly results show we made more money than last quarter. Revenue "
        "went up and costs went down. We hired new people and launched new products. "
        "Everything is going great and we expect it to keep going great."
    ),
    (
        "The experiment was done and the results were obtained. The data was analyzed "
        "using statistical methods. The results show that the hypothesis was correct. "
        "More research is needed to confirm these findings."
    ),
]
