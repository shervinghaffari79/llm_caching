"""One-shot launcher for a fresh Colab GPU runtime (runtimes are reclaimed after ~1 h, so everything is idempotent and results are
tarred every 2 minutes for the controller to download). Usage on the runtime: python run_all.py
Phases: clone the authors' repo at the pinned commit; build two virtualenvs; start vLLM (Qwen3-4B fp16, served as gpt-4o-mini);
patched copy with ReuseGate; run the task list with 8 parallel workers; keep /content/results_sync.tgz fresh."""
import os
import subprocess
import textwrap

SCRIPT = textwrap.dedent(r"""#!/bin/bash
set -x
cd /content
pip install -q uv
[ -d SOLAR ] || (git clone -q https://github.com/ysunbp/SOLAR.git && cd SOLAR && git checkout -q 8d01e9340eb3a9ab77761366e57a29c53dafefe7)
[ -x hvenv/bin/python ] || (uv venv --clear hvenv --python 3.12 && uv pip install --python hvenv/bin/python -q -r SOLAR/requirements.txt)
[ -x qvenv/bin/vllm ] || (uv venv --clear qvenv --python 3.12 && uv pip install --python qvenv/bin/python -q vllm openai huggingface_hub)
rm -rf SOLAR_patched && tar xzf /content/solar_patch.tgz && cp -r SOLAR SOLAR_patched && hvenv/bin/python solar_harness_patch/apply_patch.py /content/SOLAR_patched
unset UV_SYSTEM_PYTHON HF_TOKEN
nohup /content/qvenv/bin/vllm serve Qwen/Qwen3-4B --dtype float16 --served-model-name gpt-4o-mini \
  --chat-template /content/SOLAR/qwen3_nonthinking.jinja --max-model-len 8192 --max-num-seqs 32 --gpu-memory-utilization 0.90 --port 8000 > /content/serve.log 2>&1 &
for i in $(seq 1 120); do curl -s localhost:8000/v1/models | grep -q gpt-4o-mini && break; sleep 10; done
echo SERVER_READY
# sync loop: results are small json files
( while true; do tar czf /content/results_sync.tgz.tmp -C /content out logs 2>/dev/null && mv /content/results_sync.tgz.tmp /content/results_sync.tgz; sleep 120; done ) &
# task list: breadth first over seeds and sessions; K=10 before K=50; baselines in the unmodified copy, ReuseGate in the patched copy
python3 - <<'PY'
tasks = []
for K in (10, 50):
    for seed in (42, 1337, 2024):
        for i in range(10):
            base = "fifo lru lfu arc solar_a solar_e solar" + (" embedder_message" if K == 50 else "")
            tasks.append(f"/content/SOLAR {K} {seed} Locomo-{i} {base}")
            tasks.append(f"/content/SOLAR_patched {K} {seed} Locomo-{i} reusegate")
open("/content/tasks.txt", "w").write("\n".join(tasks) + "\n")
PY
cat > /content/run_one.sh <<'EOS'
#!/bin/bash
REPO=$1; K=$2; SEED=$3; DS=$4; shift 4
export OPENAI_API_KEY=x OPENAI_BASE_URL=http://localhost:8000/v1 EVALUATE_BASE_URL=http://localhost:8000/v1 EVALUATE_MODEL=gpt-4o-mini EVALUATE_API_KEY=x TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
OUT=/content/out/K${K}_s${SEED}
mkdir -p $OUT /content/logs
cd $REPO
/content/hvenv/bin/python run_scripts/run_experience_eval.py --methods "$@" --datasets $DS --capacity $K --seed $SEED --output $OUT > /content/logs/$(basename $REPO)_${DS}_K${K}_s${SEED}.log 2>&1
echo "EXIT $?" >> /content/logs/$(basename $REPO)_${DS}_K${K}_s${SEED}.log
EOS
cat /content/tasks.txt | xargs -P 8 -L 1 bash /content/run_one.sh
echo ALL_TASKS_DONE > /content/all_done.txt
""")

open("/content/run_all.sh", "w").write(SCRIPT)
subprocess.Popen(["setsid", "nohup", "bash", "/content/run_all.sh"], stdout=open("/content/run_all.log", "w"), stderr=subprocess.STDOUT,
                 stdin=subprocess.DEVNULL, close_fds=True)
print("run_all launched")
