# 08 — Human-Machine Interface (HMI) & Dashboard Design

> **Scope:** Tesla-like prototype running **MPC + ADAS** (no reinforcement learning).
> **Purpose:** Define the display architecture, ADAS status visualization, alert
> strategy, autopilot interaction model, driver monitoring, voice control, and
> mobile companion app for the FrontViewPredictiveControl platform.

---

## 1. Overview

The HMI is the single communication channel between the vehicle's autonomous
stack (MPC controller, ADAS features, perception, planner) and the human
occupant. A Tesla-like prototype demands an interface that is:

1. **Glance-efficient** — the driver must never spend more than 2 seconds
   looking away from the road (ISO 15005 / NHTSA visual-manual guidelines).
2. **State-transparent** — at any moment the driver can answer "what is the car
   doing right now and why?"
3. **Escalating** — warnings progress from informational → cautionary →
   imminent, with matching modality (visual → haptic → auditory).
4. **Non-RL** — all UI state transitions are rule-based, deterministic, and
   auditable. No learned policies drive the interface.

The existing implementation (`src/gui/dashboard.py`) uses **Pygame** for the
real-time in-vehicle display and **Grafana** for offline telemetry. This
document formalizes that architecture and extends it toward a production-style
multi-display HMI.

---

## 2. Display Architecture

The prototype targets a **three-surface** display stack, mirroring modern
Tesla / premium EV layouts.

### 2.1 Central 15" Touchscreen (Primary)

| Property | Value |
|---|---|
| Resolution | 1920 × 1200 (landscape) |
| Refresh | 60 Hz render, 30 Hz data |
| Framework | Pygame (prototype) → Qt/QML (production path) |
| Role | Speed, energy, ADAS status, 3D surroundings, media, climate |

The central screen is the **single source of truth** for vehicle state. It is
divided into three persistent regions plus a collapsible app dock:

```
+--------------------------------------------------------------------+
|  [ Speed | Energy | Gear | Autopilot badge ]      [ Time | Temp ]  |  <- Top bar (h=80px)
+--------------------------------+-----------------------------------+
|                                |                                   |
|     3D Surround View           |       ADAS Status Panel           |
|     (ego + detected actors)    |   (AEB/ACC/LDW/BSW/TJA/TLight)    |
|     720 × 600                  |       480 × 600                   |
|                                |                                   |
+--------------------------------+-----------------------------------+
|  [ Media ] [ Climate ] [ Nav ] [ Camera ] [ Phone ]   (app dock)   |  <- Bottom bar (h=120px)
+--------------------------------------------------------------------+
```

> **Mockup note:** The 3D view occupies the left two-thirds; the ADAS panel is
> a right-aligned vertical stack of status tiles. Each tile is 480 × 90 px,
> color-coded by state (green = active, grey = standby, amber = caution,
> red = alert).

### 2.2 Driver Display (Instrument Cluster)

A compact 5"–8" screen behind the steering wheel showing only the
**minimum-viable** set: digital speed, speed limit, gear selector, and the
autopilot engagement state. This mirrors Tesla's post-2021 "no instrument
cluster" simplification but keeps a fallback display for the prototype so that
critical state is visible even if the central screen is busy.

### 2.3 Head-Up Display (HUD)

Projected onto the windshield, the HUD mirrors only three elements from the
driver display:

1. Current speed (large, centered).
2. Speed-limit sign with active ACC set-speed.
3. Lane-outline glyph that turns blue when LKA/TJA is engaged and flashes red
   on lane departure.

The HUD exists to keep the driver's eyes **on the road**; it intentionally
shows no more than three items to avoid cognitive overload.

### 2.4 Existing Dashboard (Pygame + Grafana)

The current `Dashboard` class (`src/gui/dashboard.py`) initializes a Pygame
surface set with three blittable regions:

- `panel_surface` — the main status panel (`PANEL_W × PANEL_H`).
- `bev_surface` — the bird's-eye-view window (`BEV_WIN_W × BEV_WIN_H`).
- `view3d_surface` — the 3D perspective view (`VIEW3D_W × VIEW3D_H`).

Text is rendered via `render_text()` using `UXColors.TEXT`, and numpy images
are blitted via `render_image()`. This triple-surface layout maps directly onto
the production architecture: `panel_surface` → top bar + ADAS panel,
`bev_surface` → HUD/minimap, `view3d_surface` → central 3D surround view.

**Grafana** runs alongside as an **offline telemetry browser** (not in-cabin).
Engineers query CAN-bus logs, MPC solver traces, and ADAS event timelines
through dashboards; it is explicitly **not** a driver-facing surface.

---

## 3. 3D Visualization

The 3D surround view renders the ego vehicle and all detected actors in a
perspective scene fed by perception outputs.

### 3.1 Data Sources

- **Ego pose** — from the MPC state vector (x, y, yaw, v).
- **Detected actors** — bounding boxes from the perception pipeline
  (vehicles, pedestrians, cyclists).
- **Lane geometry** — left/right lane polynomials from lane detection.
- **Traffic lights / signs** — from `TrafficSignRecognizer` /
  `TrafficLightController` (imported by `ADASManager`).

### 3.2 Rendering Layers (back → front)

1. **Ground plane** — dark asphalt with subtle grid for motion cueing.
2. **Lane ribbons** — green when centered, amber when drifting, red on
   departure. The ribbon width reflects lane-confidence.
3. **Actor boxes** — color by classification:
   - Vehicle → light blue
   - Pedestrian → orange
   - Cyclist → purple
   - Unknown → grey
4. **Path preview** — the MPC planned trajectory drawn as a translucent cyan
   spline extending 5 s ahead.
5. **ADAS overlays** — ACC target vehicle highlighted with a bracket; AEB
   threat vehicle pulsing red; blind-spot zone cones on either side.

### 3.3 Performance Budget

The 3D view must render at **≥ 30 FPS** on the target hardware. The Pygame
prototype uses `view3d_surface` for software rasterization; the production
path offloads to OpenGL / Vulkan. Perception bounding boxes are throttled to
the sensor frame rate (20–30 Hz) to avoid overdraw.

---

## 4. ADAS Status Display

The `ADASManager` (`src/adas/adas_manager.py`) aggregates seven features with a
strict priority order. Each feature maps to one status tile in the right-hand
ADAS panel. The tile shows: feature icon, state label, and a one-line context
value.

| Priority | Feature | Tile State Machine | Context Value |
|---|---|---|---|
| 1 | **AEB** | standby → armed → **BRAKE!** | TTC (s) to lead vehicle |
| 2 | **Traffic Light** | idle → green → yellow → **RED STOP** | light color + distance |
| 3 | **TJA + Stop & Go** | off → engaged → hold | gap time (s) |
| 4 | **ACC** | off → active → override | set-speed / actual speed |
| 5 | **LKA Pro** | off → standby → active | steering torque % |
| 6 | **LDW** | off → standby → **LANE DEPARTURE** | drift direction (L/R) |
| 7 | **BSW + LCA** | clear → **BLIND SPOT** → LCA alert | side (L/R) + closing rate |

### 4.1 Tile Visual States

Each tile uses the `UXColors` palette already imported in `dashboard.py`:

- **Active / healthy** — green left border, white text.
- **Standby** — grey left border, dimmed text.
- **Caution** (e.g., TTC < 3 s) — amber left border, amber icon, pulsing at
  2 Hz.
- **Alert** (e.g., AEB firing) — red full background, white bold text, pulsing
  at 4 Hz, accompanied by haptic + auditory cues (see §5).

### 4.2 Mockup — AEB Alert Tile

```
+----------------------------------------------------+
|| BRAKE!  AEB ACTIVE                                |
||  TTC 1.4 s   decel -6.2 m/s²                      |
+----------------------------------------------------+
```

The double-bar left border is red and pulsing; the tile expands to 480 × 120
px (from 90 px) for the duration of the event, shrinking back once the threat
clears.

### 4.3 Mockup — ACC Active Tile

```
+----------------------------------------------------+
|  ACC   72 km/h  ▼                                  |
|  set 80   gap 1.8 s   lead 64 km/h                 |
+----------------------------------------------------+
```

Green left border, a small chevron indicating the set-speed vs. actual-speed
relationship, and the lead-vehicle speed when a target is tracked.

---

## 5. Alerts — Visual, Haptic, Auditory

Alerts are the most safety-critical HMI element. They follow a three-tier
escalation model aligned with ISO 15623 and Euro NCAP HMI recommendations.

### 5.1 Escalation Tiers

| Tier | Meaning | Visual | Haptic | Auditory |
|---|---|---|---|---|
| **L1 Information** | Status change, no urgency | Tile state change, green/grey | — | — |
| **L2 Caution** | Driver should act soon | Amber pulse 2 Hz, HUD flash | Steering wheel pulse (low) | Single soft chime |
| **L3 Imminent** | Driver must act now | Red full-tile pulse 4 Hz, HUD red | Seat-belt/steering double pulse | Repeating urgent tone |

### 5.2 Feature → Tier Mapping

- **AEB** firing → L3 (visual + haptic + auditory).
- **Traffic Light RED** detected → L2 → L3 as distance closes.
- **LDW** departure → L2 (visual + haptic on the departing side).
- **BSW** with turn-signal on → L2 (auditory + side-mirror icon).
- **ACC gap shrinking** → L1 → L2 if gap < 0.8 s.
- **TJA hold release** (lead moves) → L1 chime.

### 5.3 Haptic Channel

The prototype uses a **steering-wheel motor** for haptic pulses. Pulse patterns
are encoded as `(count, duration_ms, gap_ms)`:

- LDW left drift: `(2, 80, 60)` on the left grip.
- BSW right + turn signal: `(3, 60, 40)` on the right grip.
- AEB: `(4, 120, 50)` both grips.

### 5.4 Auditory Channel

Tones are synthesized via Pygame mixer (prototype) or a dedicated audio bus
(production). All tones are **distinct in pitch and cadence** so that a driver
can identify the alert type without looking:

- AEB: 880 Hz square, 3 repeats, 150 ms on / 100 ms off.
- LDW: 500 Hz sine, single 200 ms burst.
- BSW: 660 Hz triangle, 2 repeats.
- Autopilot disengage: descending two-tone (740 → 480 Hz).
- Autopilot engage: ascending two-tone (480 → 740 Hz).

---

## 6. Autopilot UI — Engagement, Disengagement, Takeover

The prototype's "autopilot" is the **MPC + ADAS stack** (no RL policy). The UI
must make the engagement boundary unambiguous.

### 6.1 Engagement

- Triggered by the driver pulling the right stalk twice or tapping the
  steering-wheel button.
- Preconditions displayed on the central screen: lane visible, speed within
  range, no AEB active, driver hands detected.
- On success: the top-bar autopilot badge transitions grey → **blue** with an
  ascending two-tone chime. The 3D view lane ribbons turn blue. The ADAS panel
  shows ACC + LKA Pro + TJA as active.

### 6.2 Disengagement

Disengagement can be **manual** (driver brakes/steers) or **automatic** (system
limit reached). Both paths produce:

1. A descending two-tone chime.
2. The autopilot badge transitions blue → grey (manual) or blue → **amber**
   (automatic with a reason code).
3. A banner under the top bar: *"Autopilot unavailable — &lt;reason&gt;"*
   (e.g., "lane lost", "speed exceeded", "AEB intervened").
4. Control authority returns to the driver; the 3D path-preview spline fades
   out over 0.5 s.

### 6.3 Takeover Request (TOR)

When the stack detects a situation it cannot handle (e.g., construction zone,
heavy rain degrading perception, planner confidence below threshold), it issues
a **takeover request**:

- **Phase 1 (caution, 4 s budget):** amber badge pulse, HUD "Take over soon"
  text, single chime, steering pulse.
- **Phase 2 (imminent, 2 s budget):** red badge pulse, HUD "TAKE OVER NOW",
  repeating urgent tone, strong haptic pulse, seat-belt tug if equipped.
- **Phase 3 (emergency fallback):** if the driver does not respond, the MPC
  performs a **minimal-risk maneuver** (in-lane stop with hazard lights) and
  the UI shows a full-screen red "VEHICLE STOPPED — TAKE CONTROL" overlay.

The TOR timer and phase are driven by the planner's confidence output, not by
a learned model.

---

## 7. Driver Monitoring System (DMS)

A cabin-facing infrared camera tracks driver attentiveness. The DMS feeds two
HMI behaviors:

1. **Hands-on-wheel detection** — required for autopilot engagement. If hands
   are absent for > 15 s while autopilot is on, a cascade of warnings
   (visual → haptic → auditory → TOR) begins.
2. **Gaze tracking** — if the driver's gaze leaves the forward road for > 2 s
   while autopilot is off, a gentle "eyes on road" reminder appears on the HUD.

DMS states shown on the central screen (small, top-right):

- 👀 Eyes on road (green)
- ⚠️ Looking away (amber)
- ✋ Hands detected (green) / No hands (red)

The DMS is **not** used to train or trigger any RL policy; it is a pure
rule-based state machine with fixed thresholds.

---

## 8. Voice Control

A offline speech-to-text module (e.g., Vosk/Whisper-tiny) allows hands-free
control of non-safety functions:

- **Climate:** "Set temperature to 22" → climate tile update.
- **Media:** "Play next song" → media dock action.
- **Navigation:** "Navigate to &lt;POI&gt;" → nav panel route preview.
- **Autopilot:** "Increase speed" / "Cancel autopilot" → ACC set-speed change
  or disengage.

Voice is **never** the sole channel for safety-critical actions; every
autopilot voice command is echoed visually and requires a steering-wheel
button confirm for engage/disengage. This keeps the deterministic control
path intact.

---

## 9. Mobile Companion App

A lightweight Flutter/React-Native app provides **remote, non-driving**
interaction:

- **Pre-trip:** climate preconditioning, charge status, route planning.
- **During trip (read-only):** live location, ETA, energy remaining — no
  control inputs while the vehicle is in motion.
- **Post-trip:** trip log, ADAS event replay (pulls from the Grafana telemetry
  store), energy report.

The app communicates with the vehicle over a TLS WebSocket; all control
commands are gated by a geofenced "parked" check on the vehicle side.

---

## 10. UX Principles & Compliance

### 10.1 ISO 15005 / NHTSA Glance Budget

- **Single-glance time ≤ 2 s** for any in-cabin task. The 3D view and ADAS
  tiles are designed so that the most critical state (speed, autopilot badge,
  top-priority alert) is readable in the foveal region within 0.5 s.
- **Total glance time ≤ 12 s** for a multi-step task (e.g., entering a
  navigation destination). The UI breaks long tasks into atomic steps with
  confirmations so the driver can resume driving between steps.
- No scrolling lists longer than 6 items are shown while the vehicle is in
  motion; longer lists are locked out with a "Park to browse" overlay.

### 10.2 Color & Contrast

- The `UXColors` palette (already referenced in `dashboard.py`) defines a
  fixed semantic set: TEXT, ACCENT, WARN, ALERT, OK. No ad-hoc RGB values.
- Minimum contrast ratio 4.5:1 for text (WCAG AA).
- Red is reserved exclusively for L3 imminent alerts; amber for L2 caution;
  green for healthy/active. This prevents color-meaning dilution.

### 10.3 Determinism (No RL)

Every HMI state transition is expressible as a finite state machine with
explicit guards on vehicle signals (speed, TTC, lane confidence, DMS state).
There is no learned component in the alert, autopilot-badge, or TOR logic.
This makes the HMI **auditable**: given the same sensor inputs, the interface
renders the same output every time, which is a prerequisite for safety
certification.

### 10.4 Fail-Safe Behavior

- If the 3D perception feed drops, the 3D view shows a static "Perception
  unavailable" placeholder and the ADAS tiles for AEB/ACC/LDW fall back to
  standby with an amber wrench icon.
- If the central screen process crashes, the driver display + HUD retain
  speed, gear, and autopilot badge (rendered by a separate watchdog process).
- If the DMS camera fails, autopilot engagement is blocked and a "Driver
  monitoring unavailable" banner appears.

---

## 11. Mapping to Existing Code

| HMI Element | Current Code Location | Notes |
|---|---|---|
| Main panel | `src/gui/dashboard.py` → `Dashboard.panel_surface` | Becomes top bar + ADAS panel |
| BEV view | `src/gui/dashboard.py` → `Dashboard.bev_surface` | Becomes HUD / minimap source |
| 3D view | `src/gui/dashboard.py` → `Dashboard.view3d_surface` | Becomes central surround view |
| ADAS aggregation | `src/adas/adas_manager.py` → `ADASManager` | Feeds all seven status tiles |
| ADAS output | `ADASControlOutput` dataclass | steer/brake/throttle/target_speed overrides + steer_assist additive |
| Telemetry | Grafana dashboards (offline) | Not driver-facing |
| Colors | `config.UXColors` | Single semantic palette |

The `ADASControlOutput` fields (`steer_override`, `brake_override`,
`throttle_override`, `target_speed_override`, `steer_assist`) are the canonical
signals the HMI inspects to render the ADAS tiles. Override values of `-1.0`
mean "MPC in control" (tile = standby/active); any other value means the ADAS
feature is actively intervening (tile = caution/alert with the magnitude shown
as context).

---

## 12. Open Items

1. Decide whether the driver display runs as a second Pygame window or a
   separate Qt process for fault isolation.
2. Specify the DMS camera hardware (global-shutter IR vs. rolling-shutter
   RGB-IR) and its CAN message layout.
3. Define the exact takeover-request timing thresholds per scenario class
   (highway vs. urban vs. traffic jam).
4. Prototype the haptic pulse encoder on the target steering-wheel motor and
   validate against the ISO 15005 glance budget.
5. Finalize the Grafana dashboard JSON templates for ADAS event replay in the
   mobile app.

---

*This document is a design specification for the FrontViewPredictiveControl
prototype. It intentionally excludes any reinforcement-learning component; all
HMI behavior is rule-based and deterministic.*
