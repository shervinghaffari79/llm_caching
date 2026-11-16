"""
Central configuration for the semantic-caching reproduction study.

Every experiment imports its knobs from here so that the *same* hardware,
batching, encoder and threshold settings are provably shared between the
latency breakdown and the end-to-end response-time / token-savings runs.
That shared provenance is what makes the numbers in the regenerated tables
mutually consistent (see `semcache.reporting.consistency`).
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
EMB_CACHE_DIR = DATA_DIR / "emb_cache"
LLM_CACHE_DIR = DATA_DIR / "llm_cache"

for _d in (DATA_DIR, RESULTS_DIR, RAW_DIR, TABLES_DIR, FIGURES_DIR,
           EMB_CACHE_DIR, LLM_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=VALUE lines; never overrides the environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
BI_ENCODER = "sentence-transformers/all-mpnet-base-v2"
CROSS_ENCODER = "cross-encoder/stsb-roberta-large"
EMB_DIM = 768

# --------------------------------------------------------------------------
# LLM under test.
#
# The manuscript used Qwen3-4B-Instruct-2507 locally. This reproduction uses
# an API-hosted model instead, which changes the *absolute* no-cache baseline
# latency but not the internal consistency question the reviewer raised: the
# cache-side overhead is measured on the same machine in the same run, so the
# arithmetic identity in `reporting.consistency` still binds.
# --------------------------------------------------------------------------
LLM_MODEL = os.environ.get("SEMCACHE_LLM_MODEL", "gpt-5.4-nano")
LLM_REASONING_EFFORT = os.environ.get("SEMCACHE_LLM_REASONING", "none")
LLM_SYSTEM_PROMPT = "You are a helpful assistant who responds briefly."
LLM_MAX_COMPLETION_TOKENS = 256
LLM_TIMEOUT_S = 60.0
LLM_MAX_RETRIES = 3
# Concurrency for *offline* pre-generation of the answer pool only. The
# measured serving loop is strictly sequential so per-query latency is real.
LLM_PREFETCH_WORKERS = 8

# --------------------------------------------------------------------------
# Batching / hardware policy
#
# `online` (batch size 1, no pre-batching) is the only regime that a live
# cache can use: query i+1 is not known when query i must be answered.
# `offline` batching is what an evaluation script does when it scores a whole
# labelled file at once. Reporting a latency measured in `offline` mode next
# to an end-to-end average measured in `online` mode is precisely the
# apples-to-oranges comparison flagged in review, so every latency number we
# emit is tagged with its regime.
# --------------------------------------------------------------------------
BatchRegime = Literal["online", "offline"]
ONLINE_BATCH = 1
OFFLINE_BATCH = 32
TORCH_THREADS = int(os.environ.get("SEMCACHE_TORCH_THREADS", "0")) or None
# Threads for meta-classifier inference while serving (training keeps n_jobs=-1).
META_INFER_THREADS = int(os.environ.get("SEMCACHE_META_THREADS", "1"))


def preferred_torch_device() -> str:
    """Select the fastest available Torch backend, with an optional override."""
    override = os.environ.get("SEMCACHE_DEVICE")
    if override:
        return override

    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _host_cpu_name() -> str:
    """CPU model of the host; LightGBM and feature arithmetic run here even
    when the encoders run on a GPU."""
    try:
        if platform.system() == "Darwin":
            import subprocess
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


@dataclass(frozen=True)
class HardwareProfile:
    """Recorded alongside every timing result so tables cannot be mixed."""
    device: str
    torch_threads: int
    cpu: str
    platform: str
    torch_version: str
    host_cpu: str = ""
    host_cpu_count: int = 0

    @staticmethod
    def capture() -> "HardwareProfile":
        import torch
        dev = preferred_torch_device()
        if dev == "cuda":
            cpu = torch.cuda.get_device_name(0)
        elif dev == "mps":
            cpu = f"Apple MPS ({platform.processor() or platform.machine()})"
        else:
            cpu = platform.processor() or platform.machine()
        return HardwareProfile(
            device=dev,
            torch_threads=torch.get_num_threads(),
            cpu=cpu,
            platform=platform.platform(),
            torch_version=torch.__version__,
            host_cpu=_host_cpu_name(),
            host_cpu_count=os.cpu_count() or 0,
        )

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Dataset sizes (mirroring the manuscript, with leakage-safe splits)
# --------------------------------------------------------------------------
N_BIENCODER = 50_000        # 25k/25k balanced
N_CROSSENCODER = 10_000     # 5k/5k balanced
N_META = 30_000             # 15k/15k balanced  -> Table 4
N_ABLATION = 3_000          # 1800/1200 (60/40) -> Table 7, held out
N_ABLATION_BALANCED = 3_000  # 1500/1500 stability re-check
CV_FOLDS_META = 5
CV_FOLDS_CROSS = 3

# --------------------------------------------------------------------------
# Workload simulation (dynamic stream)
# --------------------------------------------------------------------------
STREAM_LENGTHS = (500, 1_000, 1_500, 2_000)
STREAM_DUPLICATE_RATE = 0.50   # fraction of stream events that are repeats
STREAM_ZIPF_S = 1.1            # popularity skew over topics
CACHE_SIZES_EVICTION = (300, 500, 800)
CACHE_SIZES_QQP = (1_500, 2_000)

# --------------------------------------------------------------------------
# Semantic Decay eviction weights (from the manuscript's grid search)
# --------------------------------------------------------------------------
SD_W_FRESHNESS = 0.28
SD_W_AGE = 0.38
SD_W_BOOST = 0.34
SD_LAMBDA = 0.01
SD_RECENT_WINDOW = 20
SD_FREQ_TOP_K = 20

# --------------------------------------------------------------------------
# Decision thresholds
# --------------------------------------------------------------------------
# Fixed operating points. The two encoder thresholds are the F1-maximising
# validation thresholds of the selected models in the individual-model
# evaluation (exp4); they are constants at serving time, and every serving
# entry point accepts an override for threshold sweeps.
META_PROB_THRESHOLD = 0.50
BI_COSINE_THRESHOLD = 0.726
CROSS_THRESHOLD = 0.557
# Selective cross-encoder gate: the C2 meta-classifier decides on its own
# outside this probability band and defers to the cross-encoder inside it.
GATE_BAND = (0.30, 0.70)


# --------------------------------------------------------------------------
# The four named serving configurations.
#
# This enumeration is the direct answer to "which components were active".
# Every latency, response-time, token-saving and hit-ratio number produced by
# this codebase is emitted per-configuration; no result is reported without
# naming the configuration that produced it.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ServingConfig:
    key: str
    label: str
    use_bi: bool
    use_cross: bool
    use_geometric: bool          # distance metrics + element-wise ops
    use_text: bool
    use_meta_clf: bool           # LightGBM meta-classifier
    decision: Literal["threshold", "meta"]
    paper_name: str = ""
    note: str = ""
    # Selective cross-encoder: the meta-classifier runs on the cheap features
    # and the cross-encoder is called only for probabilities inside GATE_BAND.
    gated: bool = False

    @property
    def feature_groups(self) -> tuple[str, ...]:
        g: list[str] = []
        if self.use_bi:
            g.append("bi_cosine")
        if self.use_cross:
            g.append("cross_score")
        if self.use_geometric:
            g += ["distance", "elementwise"]
        if self.use_text:
            g.append("text")
        return tuple(g)


CONFIGS: dict[str, ServingConfig] = {
    "C1_bi_only": ServingConfig(
        key="C1_bi_only",
        label="Bi-encoder only (cosine threshold)",
        use_bi=True, use_cross=False, use_geometric=False, use_text=False,
        use_meta_clf=False, decision="threshold",
        paper_name="bi-encoder caching",
        note="GPTCache-equivalent decision rule; single static cosine threshold.",
    ),
    "C2_bi_geo_lgb": ServingConfig(
        key="C2_bi_geo_lgb",
        label="Bi-encoder + geometric/text features + LightGBM (no cross-encoder)",
        use_bi=True, use_cross=False, use_geometric=True, use_text=True,
        use_meta_clf=True, decision="meta",
        paper_name="cheap-feature stacking",
        note="The 'cheap features' configuration the title refers to.",
    ),
    "C3_bi_cross": ServingConfig(
        key="C3_bi_cross",
        label="Bi-encoder + cross-encoder re-ranking",
        use_bi=True, use_cross=True, use_geometric=False, use_text=False,
        use_meta_clf=False, decision="threshold",
        paper_name="cross-encoder re-ranking",
    ),
    "C4_full_stacking": ServingConfig(
        key="C4_full_stacking",
        label="Full stacking ensemble (bi + cross + geometric/text + LightGBM)",
        use_bi=True, use_cross=True, use_geometric=True, use_text=True,
        use_meta_clf=True, decision="meta",
        paper_name="stacking ensemble caching",
    ),
}
CONFIGS["C5_gated"] = ServingConfig(
    key="C5_gated",
    label="Cheap-feature stacking with a selective cross-encoder",
    use_bi=True, use_cross=False, use_geometric=True, use_text=True,
    use_meta_clf=True, decision="meta", gated=True,
    paper_name="gated stacking",
    note="C2's meta-classifier; uncertain cases defer to the cross-encoder.",
)
CONFIG_ORDER = ("C1_bi_only", "C2_bi_geo_lgb", "C3_bi_cross", "C4_full_stacking")

NO_CACHE_KEY = "C0_no_cache"


@dataclass
class RunManifest:
    """Written next to every result file: what ran, where, and how."""
    experiment: str
    hardware: dict
    batch_regime: str
    llm_model: str
    seed: int
    extra: dict = field(default_factory=dict)
