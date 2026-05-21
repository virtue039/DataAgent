"""Runtime configuration for a BIRD evaluation run."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SETTINGS = ("none", "oracle", "retrieval", "joint")


@dataclass
class Config:
    """Holds everything one evaluation run needs.

    `setting` selects the evidence provider:
      none      -> no business knowledge (lower bound)
      oracle    -> BIRD's per-question gold evidence (upper bound)
      retrieval -> flat-RAG over a KB built from BIRD train evidence (KAT-SQL-style baseline)
      joint     -> P1d joint subgraph retrieval over per-DB JointGraphs
    """

    bird_dir: Path
    setting: str = "oracle"
    limit: int | None = None
    concurrency: int = 4
    temperature: float = 0.0
    schema_sample_rows: int = 0
    sql_timeout: float = 30.0
    output_dir: Path = Path("results")
    question_ids_from: Path | None = None
    dry_run: bool = False

    # Flat-RAG retrieval setting
    kb_path: Path = Path("data/kb_train.json")
    retrieval_top_k: int = 5
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
    )

    # P1d joint retrieval setting
    joint_graphs_dir: Path = Path("data/joint_graphs")

    # LLM (DeepSeek by default; any OpenAI-compatible endpoint works)
    model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    max_tokens: int = 1024
    request_timeout: float = 120.0

    def __post_init__(self) -> None:
        self.bird_dir = Path(self.bird_dir).expanduser()
        self.output_dir = Path(self.output_dir).expanduser()
        self.kb_path = Path(self.kb_path).expanduser()
        self.joint_graphs_dir = Path(self.joint_graphs_dir).expanduser()
        if self.question_ids_from is not None:
            self.question_ids_from = Path(self.question_ids_from).expanduser()
        if self.setting not in SETTINGS:
            raise ValueError(f"setting must be one of {SETTINGS}, got {self.setting!r}")
