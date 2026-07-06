# Subagent Team — FrontViewPredictiveControl (CARLA MPC)

ทีม subagent 6 ตัวสำหรับทำงานต่อบน repo:
https://github.com/Telotubbies/FrontViewPredictiveControl

Claude Code จะ auto-detect และเรียกใช้งานตาม `description` ของแต่ละตัว

## ทีมงาน

| Agent | ไฟล์ | บทบาท |
|---|---|---|
| `orchestrator` | orchestrator.md | รับ task, แตกงาน, สั่งงาน agent อื่นตามลำดับ, gate ก่อน merge |
| `dev-control` | dev-control.md | แก้/เขียนโค้ด control logic: MPC, Pure Pursuit, safety, alg/ |
| `dev-ai` | dev-ai.md | แก้/เขียนโค้ด perception & AI: DSUNet, BEV, lane fitting, Kalman/LSTM |
| `tech-lead` | tech-lead.md | รีวิวสถาปัตยกรรม, ความสะอาดโค้ด, consistency ทั้ง repo |
| `qa-reviewer` | qa-reviewer.md | เขียน/รัน pytest, เช็ค safety edge case, sign-off ก่อน commit |
| `git-ops` | git-ops.md | git add/commit/push, เปิด PR, ไม่ merge เอง |

## ลำดับการทำงานที่แนะนำ

```
User request
   │
   ▼
orchestrator  ── วิเคราะห์ว่ากระทบโมดูลไหน (control / perception / ทั้งคู่)
   │
   ├──► dev-control  (ถ้ากระทบ src/control/, src/alg/, src/safety/, src/managers/control_manager.py)
   │
   └──► dev-ai       (ถ้ากระทบ src/perception/, dsunet_training/, src/temporal/)
                │
                ▼
          tech-lead   ── ตรวจ style / architecture / duplication → approve หรือ request changes
                │
                ▼
          qa-reviewer ── pytest tests/ + safety edge case → PASS/FAIL report
                │
                ▼ (เฉพาะเมื่อ PASS)
          git-ops     ── branch → commit (conventional commits) → push → เปิด PR
```

**Gate สำคัญ:** `git-ops` จะไม่ commit/push ถ้า `tech-lead` ยังไม่ approve หรือ
`qa-reviewer` ยังไม่ PASS — ถ้า user สั่งให้ git-ops ทำงานตรงๆ โดยข้ามขั้นตอน
ให้ git-ops เตือนก่อนว่าข้ามการตรวจสอบ

## วิธีใช้

ถ้าใช้ Claude Code จะ auto-detect ไฟล์ใน `.claude/agents/` แล้วเรียกใช้ตรงๆ:

- "ใช้ orchestrator วางแผนแก้บั๊ก MPC เลี้ยวโค้งช้าเกินไป"
- "ให้ dev-ai ปรับ threshold การ segment lane ใน lane_detector.py"
- "ให้ qa-reviewer รัน test แล้วสรุปผล"

ถ้าใช้ Claude ผ่านหน้าเว็บ/แอปธรรมดา (ไม่ใช่ Claude Code) ก็ยังใช้ system prompt
พวกนี้ได้ — แค่ copy เนื้อหาในแต่ละไฟล์ (ส่วนหลัง `---` ที่สอง) ไปตั้งเป็น
custom instructions/system prompt ของแต่ละ session แทน

## อ้างอิง

| ไฟล์ | หน้าที่ |
|------|--------|
| `STANDARDS.md` | coding standards + naming conventions + architecture rules |
| `WORKLOG.md` | บันทึกการเปลี่ยนแปลงทุกงาน |
| `.github/CONTRIBUTING.md` | วิธี setup + รัน test + branch strategy |
| `.github/pull_request_template.md` | PR template พร้อม safety checklist |
| `config/default.yaml` | ทุก parameter (source of truth) |
