"""Copy the ReuseGate method into a COPY of the SOLAR repository and register it (three registration edits, no policy of theirs touched).
Usage: python apply_patch.py /path/to/SOLAR_patched"""
import pathlib
import shutil
import sys

root = pathlib.Path(sys.argv[1])
here = pathlib.Path(__file__).parent
shutil.copy(here / "reusegate_agent.py", root / "src/agent/reusegate.py")
shutil.copy(here / "reusegate_solver.py", root / "src/solver/reusegate.py")
shutil.copy(here / "reusegate.json", root / "configs/memory_systems/reusegate.json")


def patch(path, old, new):
    p = root / path
    s = p.read_text()
    assert old in s and new not in s, (path, old)
    p.write_text(s.replace(old, new, 1))


patch("src/solver/__init__.py", '        "arc": ("src.solver.arc.ARCSolver", "src.agent.arc.ARCAgentConfig"),',
      '        "arc": ("src.solver.arc.ARCSolver", "src.agent.arc.ARCAgentConfig"),\n'
      '        "reusegate": ("src.solver.reusegate.ReuseGateSolver", "src.agent.reusegate.ReuseGateAgentConfig"),')
patch("src/utils.py", '            "arc": "configs/memory_systems/arc.json",',
      '            "arc": "configs/memory_systems/arc.json",\n            "reusegate": "configs/memory_systems/reusegate.json",')
patch("run_scripts/run_experience_eval.py", '    "arc",               # ARC (Adaptive Replacement Cache) baseline',
      '    "arc",               # ARC (Adaptive Replacement Cache) baseline\n    "reusegate",         # ReuseGate (label-free semantic-demand admission), added by us')
patch("run_scripts/run_experience_eval.py", '        elif method_name == "arc":',
      '        elif method_name == "reusegate":\n            return solver.agent.add_memory_reusegate(content, doc_id)\n        elif method_name == "arc":')
print("patched", root)
