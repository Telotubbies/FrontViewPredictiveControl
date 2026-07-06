# Contributing to FrontViewPredictiveControl

## Development setup

```bash
git clone https://github.com/Telotubbies/FrontViewPredictiveControl.git
cd FrontViewPredictiveControl
pip install -e ".[dev]"
```

## Running tests

```bash
# Unit tests (no CARLA required)
pytest tests/test_lane_mpc.py tests/test_reference.py tests/test_safety_override.py tests/test_stuck_recovery.py tests/test_improved_lane_fitting.py -v

# All tests (CARLA-dependent tests will error — run those locally with CARLA)
pytest tests/ --continue-on-collection-errors

# Lint
ruff check . --select=E,F,W --ignore=E501,W503
```

CI รันบน GitHub Actions ทุก push/PR — ดู `.github/workflows/ci.yml`

## Branch strategy

- `main` — stable, production-ready
- `project-ADAS` — active development branch
- `feature/<name>` — feature branch จาก `project-ADAS`
- `fix/<name>` — bugfix branch

## Commit convention

Conventional commits:

```
feat(control): add steering rate constraint to MPC
fix(perception): correct BEV homography for dashed lines
refactor(alg): merge optimized_reference into reference
test(safety): add stuck recovery full cycle test
docs(readme): update entry point to main.py
chore: update dependencies
```

Subject ≤ 72 ตัวอักษร, body อธิบาย **อะไรเปลี่ยน + ทำไม**

## Architecture

ดู `agent/README.md` สำหรับ subagent team structure และ `docs/STRUCTURE.md` สำหรับโมดูล layout

## Config

ทุก parameter อยู่ใน `config.yaml` (source of truth) — ห้าม hardcode ค่าใหม่ใน `.py` ไฟล์
