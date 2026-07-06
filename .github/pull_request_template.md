## Summary

<!-- 1-2 บรรทัด อธิบายสิ่งที่ PR นี้ทำ -->

## Changes

<!-- รายการไฟล์/โมดูลที่เปลี่ยน + สิ่งที่เปลี่ยน -->

- `path/to/file.py` — <description>
- `config.yaml` — <parameter changed + reason>

## Testing

<!-- ผล test ที่รัน -->

- [ ] Unit tests: `pytest tests/test_lane_mpc.py tests/test_reference.py tests/test_safety_override.py tests/test_stuck_recovery.py tests/test_improved_lane_fitting.py -v`
- [ ] CI: GitHub Actions ผ่าน (lint + unit-tests job)
- [ ] Integration test (ถ้ากระทบ control/perception): รันบน CARLA จริง

## Related modules

<!-- กระทบโมดูลไหน: control / perception / safety / alg / core / gui -->

## Safety checklist (ถ้ากระทบ control/safety)

- [ ] MPC fallback (Pure Pursuit) ทำงานเมื่อ solver fail
- [ ] `|δ| ≤ MPC_MAX_STEER` และ rate limit ไม่ถูกละเมิด
- [ ] Safety override เบรกเมื่อ CTE > threshold
- [ ] Stuck recovery กลับสู่ idle ได้ ไม่ loop ค้าง
