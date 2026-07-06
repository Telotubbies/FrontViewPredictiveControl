# Work Log — FrontViewPredictiveControl

> บันทึกการเปลี่ยนแปลงทุกงาน พร้อม description และผู้รับผิดชอบ
> Branch: `project-ADAS` | เริ่ม: 2026-07-06

---

## Phase 1: Cleanup + CI Setup

| # | Date | Task | Files | Agent | Description | Commit |
|---|------|------|-------|-------|-------------|--------|
| 1 | 2026-07-06 | สร้าง branch `project-ADAS` | — | orchestrator | แยก branch ใหม่จาก `main` เพื่อทำ cleanup + restructure | — |
| 2 | 2026-07-06 | ลบ `run_unet_mpc.py` | `run_unet_mpc.py` | dev-control | ลบ legacy entry point 1317 บรรทัดที่ทับซ้อนกับ `main.py` | `fbcb14a` |
| 3 | 2026-07-06 | ลบ `adas_v2/` ทั้งโฟลเดอร์ | `adas_v2/` (20 ไฟล์) | dev-control | ลบโมดูล ADAS ระดับสูง (lane change, multi-lane, mode selector) ที่แยกจาก core MPC/DSUNET | `fbcb14a` |
| 4 | 2026-07-06 | อัปเดต scripts ให้ใช้ `main.py` | `scripts/run.sh`, `scripts/run_lka.sh`, `scripts/evaluate_run.py`, `scripts/run_test_collect_and_eval.py`, `scripts/run_test_phase1_phase5.py` | dev-control | เปลี่ยนการเรียกจาก `run_unet_mpc.py` → `main.py` ทั้ง classical และ unet mode | `fbcb14a` |
| 5 | 2026-07-06 | แก้ stuck recovery bug ใน `main.py` | `main.py:204-215` | dev-control | เปลี่ยนจาก `get_recovery_control()` (ไม่มี method) เป็น `stuck_recovery.update(speed, throttle)` ที่คืน `(steer, throttle, brake, reverse)` tuple | `fbcb14a` |
| 6 | 2026-07-06 | เพิ่ม `--record-dir` + `--duration` args ใน `main.py` | `main.py:323-329` | dev-control | เพิ่ม CLI placeholder สำหรับ compatibility กับ test scripts (ยังไม่ implement fully) | `fbcb14a` |
| 7 | 2026-07-06 | ทำ `config.yaml` เป็น source of truth | `config.py` (252→82 บรรทัด) | dev-ai | ลบ hardcoded fallback 119 ค่าที่ซ้ำกับ YAML, raise `FileNotFoundError` ถ้าไม่มี config.yaml แทน silent fallback | `fbcb14a` |
| 8 | 2026-07-06 | Wire Pure Pursuit เป็น MPC fallback | `control/lane_mpc.py:31,140-142,359,362-374` | dev-control | แทน P-control แบบง่าย (`-0.5*cte - 0.3*heading_err`) ด้วย `PurePursuitController.compute_steering()` ที่ปลอดภัยกว่าที่ความเร็วสูง, status เปลี่ยนเป็น `"Fallback_PP"` | `fbcb14a` |
| 9 | 2026-07-06 | แก้ `config_clean` import error | `managers/control_manager.py:20-21,38`, `core/mpc_runner.py:28-33` | dev-control | เปลี่ยน `from config_clean import get_config` (ไฟล์ไม่มี) เป็น `from config import *` และ `import config as _config_module` | `fbcb14a` |
| 10 | 2026-07-06 | แก้ `get_recovery_control()` ใน `control_manager.py` | `managers/control_manager.py:157-176` | dev-control | เปลี่ยนจาก method ที่ไม่มี เป็น `stuck_recovery.update(speed_ms, throttle)` แล้ว unpack tuple ไป apply ลง control command | `fbcb14a` |
| 11 | 2026-07-06 | สร้าง GitHub Actions CI | `.github/workflows/ci.yml` | qa-reviewer | workflow รัน lint (ruff) + unit tests (no CARLA) + smoke imports บน Ubuntu, Python 3.11 | `fbcb14a` |
| 12 | 2026-07-06 | สร้าง `pyproject.toml` | `pyproject.toml` | tech-lead | setuptools config + pytest config (markers, testpaths) + ruff config (line-length 100, py311) | `fbcb14a` |
| 13 | 2026-07-06 | สร้าง PR template + CONTRIBUTING.md | `.github/pull_request_template.md`, `.github/CONTRIBUTING.md` | tech-lead | PR template มี safety checklist, CONTRIBUTING มี setup + test + branch strategy + commit convention | `fbcb14a` |
| 14 | 2026-07-06 | อัปเดต `.gitignore` | `.gitignore` | git-ops | เพิ่ม `*.onnx`, `*.h5`, `*.safetensors`, `*.pkl`, `.env`, `wandb/`, `mlruns/`, coverage artifacts, media files (`*.mp4`, `*.bag`) | `fbcb14a` |
| 15 | 2026-07-06 | อัปเดต `tests/requirements-test.txt` | `tests/requirements-test.txt` | qa-reviewer | เพิ่ม dependencies ที่ test จริงต้องการ: pyyaml, numpy, scipy, opencv, torch, casadi, ruff | `fbcb14a` |
| 16 | 2026-07-06 | รัน unit test suite verify | — | qa-reviewer | `pytest tests/test_lane_mpc.py tests/test_reference.py tests/test_safety_override.py tests/test_stuck_recovery.py tests/test_improved_lane_fitting.py -v` → **34 passed** | — |

---

## Phase 2: src/ Layout Restructure

| # | Date | Task | Files | Agent | Description | Commit |
|---|------|------|-------|-------|-------------|--------|
| 17 | 2026-07-06 | ลบ archive/ ทั้งหมด | `perception/archive/` (8 ไฟล์), `scripts/archive/` (28 ไฟล์), `docs/archive/` (32 ไฟล์), `visualization/` (2 ไฟล์), `scripts/debug/` (2 ไฟล์), `adas/run.py` | git-ops | ลบ legacy code ทั้งหมดที่ไม่ได้ใช้ | TBD |
| 18 | 2026-07-06 | ลบไฟล์ perception ซ้ำซ้อน | `perception/lane_tracker.py`, `edge_detection.py`, `addison_lane.py`, `lane_clustering.py`, `geometric_validation.py`, `temporal_lane_buffer.py`, `classical_lane_detector.py`, `lane_validator.py` | dev-ai | ลบไฟล์ที่ไม่ถูก import จากที่อื่น (เก็บ `ego_lane_mask.py` ไว้เพราะ `lane_detector.py` ใช้) | TBD |
| 19 | 2026-07-06 | ลบไฟล์ซ้ำซ้อน | `managers/perception_manager_3080ti.py`, `tests/test_carla_manager_unittest.py`, `tests/test_spline_fit.py` | git-ops | ลบ GPU-specific duplicate และ test ซ้ำซ้อน | TBD |
| 20 | 2026-07-06 | ย้าย modules เข้า `src/` | `control/`, `perception/`, `safety/`, `algorithms/`, `core/`, `managers/`, `gui/`, `temporal/`, `bridge/`, `utils/` → `src/` + `main.py`, `pipeline.py`, `carla_input_output.py`, `state.py`, `config.py` → `src/` | dev-control | ย้ายทั้งหมดเข้า `src/` เป็น Python package มาตรฐาน + สร้าง `src/__init__.py` | TBD |
| 21 | 2026-07-06 | ย้าย `config.yaml` → `config/` | `config.yaml` → `config/default.yaml`, สร้าง `config/__init__.py` loader | dev-ai | สร้าง config package พร้อม loader ที่อ่านจาก `config/default.yaml`, flatten nested sections เป็น module-level constants | TBD |
| 22 | 2026-07-06 | สร้าง entry point wrapper | `main.py` (ใหม่ที่ root) | dev-control | thin wrapper ที่ add `src/` ลง sys.path แล้ว import `main` จาก `src/main.py` — ทำให้ `python main.py` ยังทำงานได้ | TBD |
| 23 | 2026-07-06 | จัด `tests/` เป็น `unit/` + `integration/` | 8 unit tests → `tests/unit/`, 6 integration tests → `tests/integration/`, สร้าง `tests/conftest.py` | qa-reviewer | แยก unit test (no CARLA) จาก integration test (CARLA required) + conftest เพิ่ม src/ ลง sys.path | TBD |
| 24 | 2026-07-06 | อัปเดต `pyproject.toml` สำหรับ src/ layout | `pyproject.toml` | tech-lead | `package-dir = {"" = "src"}`, `packages.find.where = ["src"]`, เพิ่ม `[project.scripts] fpvc = "main:main"`, อัปเดต `testpaths` เป็น `tests/unit, tests/integration` | TBD |
| 25 | 2026-07-06 | อัปเดต CI workflow สำหรับ src/ | `.github/workflows/ci.yml` | qa-reviewer | เปลี่ยน `PYTHONPATH: src:.` ทุก step, อัปเดต test paths เป็น `tests/unit/` | TBD |
| 26 | 2026-07-06 | Verify imports + รัน test suite | — | qa-reviewer | smoke imports ผ่านทั้งหมด, `pytest tests/unit/` → **34 passed** | TBD |
| 27 | 2026-07-06 | Commit + push trigger CI | — | git-ops | commit phase 2 + push เพื่อ trigger GitHub Actions | `a34b247` |

---

## Phase 3: File Naming Refactor

| # | Date | Task | Files | Agent | Description | Commit |
|---|------|------|-------|-------|-------------|--------|
| 28 | 2026-07-06 | ลบ dead code (6 ไฟล์, 1572 บรรทัด) | `algorithms/optimized_reference.py` (180), `gui/apollo_dashboard.py` (642), `gui/dsunet_dashboard.py` (312), `perception/vectorized_sliding_window.py` (277), `perception/gpu_bev_transform.py` (171), `utils/cuda_optimizations.py` (342) | git-ops | ลบไฟล์ที่ไม่ถูก import จากที่ไหนเลย — audit ด้วย grep ทั้ง repo | TBD |
| 29 | 2026-07-06 | Rename `improved_lane_fitting.py` → `spline_lane_fitting.py` | `src/perception/spline_lane_fitting.py`, `src/perception/lane_trajectory.py:31`, `tests/unit/test_spline_lane_fitting.py` | dev-ai | "improved" prefix เป็น subjective naming — เปลี่ยนเป็น "spline" ที่บอก implementation จริง (B-spline) | TBD |
| 30 | 2026-07-06 | Rename `safety_override.py` → `override.py` | `src/safety/override.py`, `src/safety/__init__.py:7`, `src/main.py:33`, `src/pipeline.py:28`, `src/managers/control_manager.py:110`, `src/algorithms/lane_keep_assist_step.py:61`, `tests/unit/test_override.py` | dev-control | ลบ redundant "safety_" prefix — ไฟล์อยู่ใน `safety/` อยู่แล้ว | TBD |
| 31 | 2026-07-06 | Rename `algorithms/step.py` → `algorithms/lane_keep_assist_step.py` | `src/algorithms/lane_keep_assist_step.py`, `src/algorithms/__init__.py:5` | dev-control | "step" กว้างเกินไป — เปลี่ยนเป็น "lka_step" ที่บอกว่าเป็น Lane Keeping Assist step | TBD |
| 32 | 2026-07-06 | Rename test files ให้ตรงโมดูล | `tests/unit/test_improved_lane_fitting.py` → `test_spline_lane_fitting.py`, `tests/unit/test_safety_override.py` → `test_override.py` | qa-reviewer | ชื่อ test ตรงกับชื่อโมดูลที่ test หลัง rename | TBD |
| 33 | 2026-07-06 | อัปเดต CI workflow + CONTRIBUTING | `.github/workflows/ci.yml`, `.github/CONTRIBUTING.md` | qa-reviewer | อัปเดต smoke import + test paths ให้ใช้ชื่อใหม่ | TBD |
| 34 | 2026-07-06 | Verify imports + รัน test suite | — | qa-reviewer | smoke imports ผ่าน, `pytest tests/unit/` → **34 passed** | TBD |

---

## Subagent Reports (initial assessment)

| Agent | ID | งาน | ผล |
|-------|----|------|-----|
| orchestrator | `2f10495a` | repo scope analysis | พบ tech debt: config fallback, duplicate entry points, ขาด unit test control layer |
| dev-control | `32bf6115` | control stack review | พบ blocking: config_clean import, get_recovery_control missing, PP ไม่ได้ใช้, MPC weights ไม่ตรง |
| dev-ai | `f6745049` | perception stack review | pipeline ตรงตาม context, DSUNet IoU 0.861/Dice 0.916 |
| tech-lead | `6e672015` | architecture review | Request Changes: 2 blocking (run_unet_mpc ref, get_recovery_control) — แก้แล้วทั้งคู่ |
| qa-reviewer | `39690658` | test suite run | FAIL: 50 passed/12 failed/5 errors — ส่วนใหญ่เป็น test bug เดิม (config_clean, UnicodeDecodeError) |
| git-ops | `6831edf4` | repo state audit | repo สะอาด 1.48 MiB, .gitignore ขาด model weights formats |

---

## Subagent Reports (phase 1 execution)

| Agent | ID | งาน | ผล |
|-------|----|------|-----|
| dev-control | `90eb414c` | wire PP fallback + fix imports + fix get_recovery_control | สำเร็จ: 4 edits ใน lane_mpc.py, 2 edits ใน control_manager.py, 1 edit ใน mpc_runner.py |
| dev-ai | `b34b7bb0` | config.yaml source of truth | สำเร็จ: config.py 252→82 บรรทัด, ลบ fallback 119 ค่า, raise error ถ้าไม่มี YAML |
| tech-lead | `932aefed` | review changes | Request Changes → แก้ครบแล้วทั้ง 2 blocking |
| qa-reviewer | `e1ba1b71` | run tests after changes | 34/34 unit tests ผ่าน |
| orchestrator | `cf2d15f8` | plan next phase | สร้าง task plan 5 priorities (safety, MPC tuning, perception, testing, GUI) |
| git-ops | `75959b1b` | audit branch state | diff สะอาด ไม่มี debug/credential/binary, เตรียม commit message พร้อม |

---

## Sprint 6: Subagent Integration Workflow (2025-07-12)

| # | Agent | Task | Files | Description | Result |
|---|-------|------|-------|-------------|--------|
| 1 | orchestrator | Assess project state, run baseline tests | — | 611 passed, 2 skipped, 0 failed | baseline established |
| 2 | dev-control | Fix UXColors missing aliases | `config/__init__.py` | Added TEXT, SUCCESS, WARNING, ERROR, BACKGROUND to UXColors class | dashboard import fixed |
| 3 | dev-control | Fix config_clean imports | `managers/carla_manager.py`, `managers/perception_manager.py`, `managers/display_manager.py` | Replaced `from config_clean import get_config` with `from config import *` + `get_config()` shim | 3 files fixed |
| 4 | dev-control | Fix Qt fallback | `gui/qt_compatibility.py`, `gui/panels/__init__.py` | Added PySide6 fallback + dummy stubs when no Qt installed; guard panels import | worker.py imports OK |
| 5 | dev-control | Fix BOM in panels/__init__.py | `gui/panels/__init__.py` | Removed UTF-8 BOM that caused SyntaxError | syntax check passes |
| 6 | dev-control | Fix test_carla_manager_import | `tests/unit/test_basic_functionality.py` | Changed `callable(manager.camera_callback)` to `assertIsNone` (camera_callback is None until registered) | test passes |
| 7 | dev-ai | Fix ego_lane_tracker broken import | `perception/ego_lane_tracker.py` | Moved logger before try/except; added guard for `cluster_lane_candidates is None` in `update()` | graceful degradation |
| 8 | dev-ai | Fix classical_lane_detector import | `perception/lane_trajectory.py` | Changed `from .classical_lane_detector import create_classical_detector` to `from .classical.detector import ClassicalLane` | correct module path |
| 9 | dev-ai | Inline edge_detection functions | `perception/classical/detector.py` | Replaced deleted `edge_detection` module with inline `threshold`, `blur_gaussian`, `mag_thresh` | classical detector works |
| 10 | tech-lead | Review all changes | — | Reviewed diff: minimal, focused, follows existing style, no architectural concerns | approved |
| 11 | qa-reviewer | Write regression tests | `tests/unit/test_regression_fixes.py` | 17 regression tests covering all 6 bug categories | 17/17 passed |
| 12 | qa-reviewer | Full test suite | — | 630 passed, 0 failed, lint clean | ready for delivery |
| 13 | git-ops | Commit + update WORKLOG | — | Commit all changes with descriptive message | — |

---

## Sprint 7: Measurement & Evaluation System (2025-07-12)

| # | Agent | Task | Files | Description | Result |
|---|-------|------|-------|-------------|--------|
| 1 | orchestrator | Plan metrics system | — | Audit existing telemetry (InfluxDB exporter, get_metrics, MPCResult) and design comprehensive MetricsCollector | plan approved |
| 2 | dev-control | Create MetricsCollector | `src/telemetry/metrics_collector.py` | New module: FrameMetrics dataclass + MetricsCollector class — records vehicle, lane, MPC, safety, ADAS, performance metrics per frame; saves CSV + JSON summary | module created |
| 3 | dev-control | Add MPC solve_time to pipeline | `src/pipeline.py`, `src/state.py` | Capture `mpc_solve_time_ms` and `solver_status` in pipeline.step(); add field to FrameState | pipeline records MPC timing |
| 4 | dev-control | Integrate MetricsCollector into main loop | `src/main.py` | Import, init, record every frame, save on cleanup; add `--metrics-dir` CLI arg | main loop records all metrics |
| 5 | dev-ai | Perception metrics recording | — | MetricsCollector already captures cte, heading, curvature, confidence, geometry_valid, phase_p1-p5 rates | perception metrics covered |
| 6 | dev-ai | DSUNet evaluation script | `scripts/run_evaluation.py` | Standalone script wrapping existing evaluate.py — outputs IoU/Dice/Precision/Recall/F1 + baseline comparison | script created |
| 7 | dev-control | Run metrics analysis script | `scripts/run_metrics.py` | Standalone CSV analyzer — prints summary + saves JSON; works without CARLA | script created |
| 8 | tech-lead | Review metrics system | — | Reviewed diff: clean module, no cross-layer violations, backward compatible, no blocking calls in main loop | approved |
| 9 | qa-reviewer | Write MetricsCollector tests | `tests/unit/test_metrics_collector.py` | 14 tests: FrameMetrics defaults, record/save/summary/reset, fallback counting, phase rates, safety metrics, run_metrics analysis | 14/14 passed |
| 10 | qa-reviewer | Full test suite + lint | — | 644 passed, 0 failed, ruff clean | ready for delivery |
| 11 | git-ops | Commit + push + WORKLOG | `4be6b66` | Commit all Sprint 7 changes, pushed to project-ADAS | done |

### Metrics Captured (per frame, saved to CSV + JSON)

**Perception:**
- CTE (mean, std, max_abs, RMSE)
- Heading error (mean, std, max_abs)
- Lane confidence (mean, std, min)
- Geometry valid rate
- Phase P1–P5 success rates

**Control / MPC:**
- Solve time (mean, p95, max)
- Fallback count + rate
- Steering (mean, std, max_abs)
- Speed (mean, std, max)

**Performance:**
- FPS (mean, min, std)
- Loop time (mean, p95, max)

**Safety:**
- Safety active count + rate
- Stuck recovery count
- AEB trigger count
- ACC active count
- LDW warning count

**DSUNet Model Evaluation (offline):**
- IoU, Dice, Precision, Recall, F1
- Baseline comparison (IoU ≥ 0.861, Dice ≥ 0.916)

---

## Notes

- ทุกงานที่ทำจะถูกบันทึกในไฟล์นี้ พร้อม commit hash
- ถ้างานยังไม่ commit ให้ใส่ `—` ในคอลัมน์ Commit
- ถ้างานยังไม่เริ่ม ให้ใส่ `TBD` ในคอลัมน์ Date และ Commit
- หลัง commit ทุกครั้ง ให้อัปเดตตารางนี้พร้อม commit hash
