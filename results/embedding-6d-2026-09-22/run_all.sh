#!/usr/bin/env bash
# Embed all eight native 6D Ionut cases, four at a time, then summarize.
# Run from the repository root.
set -u
cases="ionut-itg-tem-argmax-gamma ionut-itg-tem-argmax-omega ionut-itg-tem-softmax-gamma
ionut-itg-tem-softmax-omega ionut-itg-kbm-argmax-gamma ionut-itg-kbm-argmax-omega
ionut-itg-kbm-softmax-gamma ionut-itg-kbm-softmax-omega"
export NUMBA_NUM_THREADS=3 OMP_NUM_THREADS=3
echo $cases | tr ' ' '\n' | xargs -P 4 -I{} sh -c \
  './.venv/Scripts/python.exe -m scripts.embed_6d --case {} --output results/embedding-6d-2026-09-22/{} 2>&1 | grep -v -i warn > results/embedding-6d-2026-09-22/{}.log; echo "{} finished"'
./.venv/Scripts/python.exe -m scripts.embed_6d --summarize --output results/embedding-6d-2026-09-22
