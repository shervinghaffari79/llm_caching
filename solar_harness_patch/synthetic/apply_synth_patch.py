"""Create the patched runners module in a COPY of the SOLAR repo: appends reusegate_runners.py to synthetic_workloads/runners.py.
Usage: python apply_synth_patch.py /path/to/SOLAR_synth_patched"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
here = pathlib.Path(__file__).parent
target = root / "synthetic_workloads" / "runners.py"
target.write_text(target.read_text() + "\n\n" + (here / "reusegate_runners.py").read_text())
print("patched", target)
