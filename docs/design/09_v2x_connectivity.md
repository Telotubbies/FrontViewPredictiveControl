# 09 — V2X & Connectivity (Classical, No RL)

> Tesla-like prototype. **No reinforcement learning.** All cooperative-driving
> behaviors are rule-based fusion of V2X messages with the onboard perception
> stack. This document covers V2X protocols (DSRC, C-V2X), the four V2X
> communication modes (V2V, V2I, V2N, V2P), the network architecture (edge
> nodes, 5G), security (PKI, SCMS), and a **non-RL fleet-learning** data
> pipeline inspired by Tesla's shadow-mode approach.

---

## 0. Architecture Overview

```
                         ┌──────────────────────────────────────────────┐
                         │              CLOUD / FLEET NETWORK            │
                         │  (Fleet Learning, OTA, HD Map Refresh, SCMS) │
                         └───────────────┬──────────────────────────────┘
                                         │  V2N (LTE / 5G NR / HTTP(S))
                         ┌───────────────▼──────────────────────────────┐
                         │            ROADSIDE UNIT (RSU)               │
                         │  Traffic-light phase, work-zone, SPaT/MAP    │
                         └───────┬───────────────────────┬──────────────┘
                 V2I (DSRC/C-V2X)│                       │ V2I
            ┌────────────────────┘                       └─────────────────────┐
            ▼                                                                  ▼
┌───────────────────────┐  V2V (BSM/CAM)  ┌───────────────────────┐   V2P (PSM)
│  Ego Vehicle           │←──────────────→ │  Surrounding Vehicles  │←──────────→ Pedestrian
│  ┌─────────────────┐   │                 │  ┌─────────────────┐   │              Device
│  │ V2X Stack       │   │                 │  │ V2X Stack       │   │
│  │ (DSRC / C-V2X)  │   │                 │  │ (DSRC / C-V2X)  │   │
│  ├─────────────────┤   │                 │  ├─────────────────┤   │
│  │ Fusion / Tracker│   │                 │  │ Fusion / Tracker│   │
│  ├─────────────────┤   │                 │  ├─────────────────┤   │
│  │ Planner (FSM)   │   │                 │  │ Planner (FSM)   │   │
│  └─────────────────┘   │                 │  └─────────────────┘   │
└───────────────────────┘                 └───────────────────────┘
```

The V2X stack is a **side-channel** to the vision/perception stack. It does not
replace cameras; it augments them with information beyond line-of-sight
(traffic-light phase around a curve, a hard-braking vehicle two cars ahead,
a pedestrian occluded by a parked truck). All V2X inputs are fused into the
existing object list and behavioral FSM described in `05_path_planning.md`.

---

## 1. V2X Protocols

Two physical-layer standards compete; the prototype supports **both** via an
abstract `V2XTransport` interface so the hardware can be swapped without
touching the application layer.

### 1.1 DSRC (IEEE 802.11p / WAVE)

| Layer | Standard | Notes |
|-------|----------|-------|
| Physical | IEEE 802.11p OFDM | 10 MHz channels in 5.9 GHz ITS band |
| MAC | IEEE 802.11p (CSMA/CA) | Low-overhead BSS, no association |
| Network | IPv6 (WAVE) | GeoNetworking for geographic addressing |
| Transport | UDP / WSMP (IEEE 1609.3) | WSMP = low-latency single-hop |
| Security | IEEE 1609.2 | ECDSA over NIST P-256, SCMS certs |
| Application | SAE J2735 | BSM, SPaT, MAP, PSM, TIM |

- **Latency:** ~5–50 ms single-hop.
- **Range:** ~300 m line-of-sight, degrades with occlusion.
- **Throughput:** 3–27 Mbps per channel.
- **Pros:** Mature, deterministic, no cellular subscription.
- **Cons:** Dedicated spectrum under regulatory pressure; limited scalability
  in dense traffic (CSMA/CA collisions).

### 1.2 C-V2X (3GPP Release 14+)

| Layer | Standard | Notes |
|-------|----------|-------|
| Physical | 3GPP NR PC5 (sidelink) / Uu (cellular) | PC5 = direct V2V; Uu = via base station |
| MAC/RLC | 3GPP NR | Scheduled (mode 1) or autonomous (mode 2) |
| Network | IPv6 | PC5 uses ProSe; Uu uses standard IMS/EPC |
| Transport | UDP / MQTT-SN | MQTT-SN for V2N telemetry |
| Security | 3GPP TS 33.536 + SCMS overlay | Same PKI as DSRC for app certs |
| Application | SAE J2735 / ETSI CAM-DENM | Interoperable message set |

- **Latency:** PC5 ~3–20 ms; Uu ~20–100 ms.
- **Range:** PC5 ~500 m+ with relaying; Uu = cellular coverage.
- **Pros:** Scales with 5G rollout, reuses cellular infrastructure, supports
  network slicing for low-latency URLLC.
- **Cons:** Cellular subscription cost; PC5 performance depends on resource
  scheduling by the operator.

### 1.3 Transport Abstraction

```python
class V2XTransport(ABC):
    @abstractmethod
    def send(self, msg: J2735Message, channel: Channel) -> None: ...
    @abstractmethod
    def recv(self, timeout_ms: int) -> J2735Message | None: ...
    @abstractmethod
    def neighbors(self) -> list[PeerId]: ...

class DSRCTransport(V2XTransport): ...   # 802.11p + WAVE
class CV2XTransport(V2XTransport): ...   # NR PC5 + Uu fallback
```

The fusion layer never sees the PHY. A runtime config flag selects the active
transport; both can run concurrently for redundancy.

---

## 2. Protocol Stack Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  Application Layer  (SAE J2735 message set)                 │
│  BSM | SPaT | MAP | PSM | TIM | RSA | EVA                   │
├─────────────────────────────────────────────────────────────┤
│  Facility Layer                                            │
│  Geo-broadcast | Congestion control (DCC) | LDM (local DB) │
├─────────────────────────────────────────────────────────────┤
│  Security Layer  (IEEE 1609.2 / 3GPP TS 33.536)            │
│  Sign + verify | SCMS cert chain | Misbehavior reporting   │
├─────────────────────────────────────────────────────────────┤
│  Network Layer   (IPv6 / GeoNetworking)                    │
│  Geo-unicast | Geo-broadcast | Topology broadcast          │
├─────────────────────────────────────────────────────────────┤
│  Transport Layer  (UDP / WSMP / MQTT-SN)                   │
├──────────────────────┬──────────────────────────────────────┤
│  DSRC: IEEE 802.11p   │  C-V2X: 3GPP NR PC5 + Uu            │
│  PHY (OFDM 5.9 GHz)   │  PHY (NR sidelink / cellular)        │
└──────────────────────┴──────────────────────────────────────┘
```

The **Local Dynamic Map (LDM)** is an in-vehicle spatio-temporal database that
stores all received V2X objects with TTLs and confidence metadata. It is the
bridge between the V2X stack and the perception fusion module.

---

## 3. V2V — Vehicle-to-Vehicle

### 3.1 Basic Safety Message (BSM)

Broadcast at **10 Hz** by every connected vehicle. Core fields:

```
BSM {
  msg_count        : uint8        # rolling counter
  temp_id          : uint32       # rotates every 5 min (privacy)
  position         : lat, lon, elev
  accuracy         : semi_major, semi_minor, orientation
  speed, heading   : float
  acceleration     : float        # longitudinal
  yaw_rate         : float
  brake_status     : bitmask      # ABS, TCS, SCB, engaged
  turn_signal      : enum
  vehicle_size     : length, width
  path_history     : list[waypoint]   # last ~10 s, downsampled
  path_prediction  : list[waypoint]   # next ~3 s
  event_flags      : { hazard_lights, hard_brake, disabled_vehicle }
}
```

### 3.2 Collision Warning (rule-based, no RL)

The ego vehicle maintains a **time-to-collision (TTC)** estimate for every
tracked object, including V2V peers. The warning hierarchy is a deterministic
threshold ladder:

```
function V2V_COLLISION_WARNING(ego, peers, perception):
    threats ← []
    for peer in peers ∪ perception.objects:
        ttc ← COMPUTE_TTC(ego, peer)        # longitudinal + lateral
        if ttc < TTC_HARD and lateral_overlap(ego, peer):
            threats.append((peer, ttc, "HARD_BRAKE"))
        elif ttc < TTC_SOFT:
            threats.append((peer, ttc, "CAUTION"))
    threats.sort(by=ttc)
    if threats and threats[0].ttc < TTC_EMERGENCY:
        EMERGENCY_BRAKE(threats[0].peer)    # control-level override
    return threats
```

Thresholds (tunable, not learned):

| Level | TTC (s) | Action |
|-------|---------|--------|
| Info | 4.0–6.0 | Log + HUD icon |
| Caution | 2.5–4.0 | Acoustic chime, pre-charge brakes |
| Warning | 1.5–2.5 | Partial brake (~3 m/s²) |
| Emergency | < 1.5 | Full AEB, max decel |

### 3.3 Cooperative Platooning (CACC)

Cooperative Adaptive Cruise Control uses V2V to receive the leader's
acceleration **before** the ego's radar sees the gap change, cutting the
string-stability delay from ~0.5 s (camera) to ~0.05 s (V2V).

```
┌──────┐  BSM (acc, speed)  ┌──────┐  BSM  ┌──────┐
│Leader│ ──────────────────→ │Foll-1│ ────→ │Foll-2│
└──────┘                     └──────┘       └──────┘
   │                            │              │
   └──── gap controller ◄──────┴──────────────┘
        CACC: a_des = k1*(gap - gap_ref) + k2*(v_lead - v_ego)
```

- **Spacing policy:** constant time-gap (`gap_ref = v_ego * τ + L`, `τ = 0.6 s`).
- **String stability** guaranteed when `k1, k2` satisfy the headway-transfer
  function bound (Rajamani / Swaroop classical result — no learning).
- **Join/leave protocol:** a 4-way handshake (`JOIN_REQ → JOIN_ACK → GAP_ADJUST
  → LOCKED`) governed by an FSM; rejection if speed delta > 3 m/s or lane
  mismatch.

---

## 4. V2I — Vehicle-to-Infrastructure

### 4.1 Signal Phase and Timing (SPaT) + MAP

The RSU broadcasts the current and **predicted** phase of every signal in an
intersection, plus a geometric MAP of lanes and stop bars.

```
SPaT {
  intersection_id : uint16
  states : [
    { signal_group, phase: GREEN|YELLOW|RED|PED,
      time_to_change_ms, confidence }
  ]
  timestamp
}
MAP {
  intersection_id
  lanes : [ { lane_id, type, connects_to[], stop_line_xy[] } ]
}
```

**GLOSA (Green Light Optimal Speed Advisory):** the planner computes a
target speed band that lets the ego pass the stop bar on green without
stopping:

```
v_advisory = clamp(d_to_stop / t_to_green, v_min, v_limit)
```

If `d_to_stop / t_to_green > v_limit`, the planner transitions to the
`STOP_AT_SIGNAL` behavioral state and begins a smooth deceleration profile.

### 4.2 Work-Zone and Hazard Alerts (TIM)

Traveler Information Messages carry geo-fenced advisories:

- Lane closures, reduced speed limits, lane shifts.
- Roadwork start/end coordinates with a validity window.
- Weather hazards (ice, hydroplaning risk) reported by upstream vehicles.

The planner inserts a **temporary edge cost** into the lane graph (see
`05_path_planning.md` §1) so routing can detour around long closures, while
short closures only trigger a local speed reduction.

---

## 5. V2N — Vehicle-to-Network

### 5.1 Cloud Services

```
┌─────────────┐   TLS / HTTP(S) over LTE-5G   ┌──────────────────────┐
│  Vehicle    │ ────────────────────────────→ │  Fleet Cloud         │
│  Telemetry  │ ←──────────────────────────── │  - OTA orchestrator  │
│  Agent      │   MQTT / gRPC                 │  - HD map diff store │
└─────────────┘                               │  - Fleet learning DB │
                                              │  - SCMS certificate  │
                                              │    authority proxy   │
                                              └──────────────────────┘
```

V2N channels:

| Channel | Direction | Purpose | Cadence |
|---------|-----------|---------|---------|
| Telemetry | up | health, GNSS, disengagements | 1 Hz / event |
| Map diff | down | lane additions, closures | on-demand |
| OTA | down | firmware, model weights | campaign-based |
| Fleet learn | up | shadow-mode clips | event-triggered |
| SCMS | bidir | cert provisioning, revocation | daily |

### 5.2 Over-the-Air (OTA) Updates

OTA is staged to minimize risk:

1. **Manifest** — signed JSON listing artifacts + checksums + target ECU.
2. **Download** — chunked, resumable, verified with SHA-256 + ECDSA.
3. **Staging** — written to inactive A/B partition; never the running one.
4. **Pre-install checks** — battery SoC > 50%, parked, signed rollback token.
5. **Commit** — atomic boot-flag switch; watchdog auto-rolls back on boot
   failure within 30 s.
6. **Canary** — 1% → 10% → 100% fleet rollout with telemetry gate at each
   tier (regression in disengagement rate aborts the campaign).

---

## 6. V2P — Vehicle-to-Pedestrian

### 6.1 Personal Safety Message (PSM)

Pedestrian devices (phones, wearables) broadcast a stripped-down BSM:

```
PSM {
  temp_id, position, accuracy, speed, heading,
  type : PEDESTRIAN | BICYCLIST | WHEELCHAIR,
  crossing_intent : bool,   # derived from device motion + map crosswalks
}
```

### 6.2 Pedestrian Alert Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌────────────────┐    ┌──────────┐
│ PSM received │ →  │ LDM insert   │ →  │ Fusion with    │ →  │ Planner  │
│ (V2P)        │    │ + confidence │    │ vision tracks  │    │ yield/   │
└──────────────┘    └──────────────┘    └────────────────┘    │ brake    │
                                                                └──────────┘
```

Rules (deterministic):

- If a PSM source is within `d_cross = 15 m` of the ego path and
  `crossing_intent == true` → planner enters `YIELD_PEDESTRIAN`.
- If vision confirms a pedestrian in the predicted path → escalate to
  `EMERGENCY_BRAKE` regardless of PSM (vision is authoritative).
- V2P-only detections (no vision corroboration) are **soft** alerts: speed
  reduction + acoustic warning, never full AEB, to mitigate spoofing.

---

## 7. Network Architecture — Edge & 5G

### 7.1 Edge Computing Topology

```
        ┌────────────────────────────────────────────────────┐
        │              5G Core (AMF / SMF / UPF)             │
        └───────────┬──────────────────────────┬─────────────┘
                    │                          │
        ┌───────────▼──────────┐   ┌───────────▼──────────┐
        │  Edge Node (MEC) #1  │   │  Edge Node (MEC) #2  │
        │  - Regional LDM      │   │  - Regional LDM      │
        │  - Aggregated hazard │   │  - Aggregated hazard │
        │    map (rolling 5m)  │   │    map (rolling 5m)  │
        │  - SCMS RA proxy     │   │  - SCMS RA proxy     │
        └───────┬──────────────┘   └──────────────────────┘
                │  N3/N6
       ┌────────┴────────┐
       │  gNB (5G NR)    │
       └────────┬────────┘
                │  Uu (V2N) + PC5 (V2V/V2I sidelink)
   ┌────────────┴───────────────┐
   │  Vehicles / RSUs / Ped-dev │
   └────────────────────────────┘
```

**Edge responsibilities:**

- **Regional LDM aggregation** — fuses BSMs from many vehicles into a
  coarse regional traffic + hazard picture, pushed back to vehicles as TIM.
- **Low-latency V2N relay** — when PC5 range is exceeded, the edge relays
  safety-critical messages over Uu with < 50 ms target.
- **5G network slicing** — a dedicated URLLC slice (3GPP TS 23.501) is
  reserved for safety messages; best-effort telemetry uses an eMBB slice.

### 7.2 Latency Budget

| Path | Budget | Typical |
|------|--------|---------|
| V2V PC5 direct | 20 ms | 3–10 ms |
| V2I RSU → vehicle | 50 ms | 10–30 ms |
| V2N vehicle → edge MEC | 50 ms | 15–40 ms |
| V2N vehicle → cloud | 500 ms | 100–300 ms |
| OTA campaign (GB-class) | hours | scheduled overnight |

Safety-critical loops (collision warning, GLOSA) must close within the
**50 ms edge budget**; fleet learning and OTA tolerate seconds-to-hours.

---

## 8. Security — PKI & SCMS

### 8.1 SCMS (Security Credential Management System)

SCMS is the PKI specified for V2X in the US (and analogous to the EU CCMS).
It provides **pseudonymity**: every vehicle holds a rotating set of
short-term certificates that cannot be linked back to the VIN by an
observer, while a trusted authority can revoke them.

```
                        ┌─────────────────────────────┐
                        │   Root CA (offline, HSM)    │
                        └─────────────┬───────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │ Enrollment CA    │   │ Pseudonym CA     │   │ Misbehavior Auth │
   │ (long-term LDEV) │   │ (batch of PCerts)│   │ (MA, detection)  │
   └────────┬─────────┘   └────────┬─────────┘   └──────────────────┘
            │                      │
            ▼                      ▼
   ┌──────────────────┐   ┌──────────────────┐
   │ Vehicle OBE      │   │ 20 rotating      │
   │ long-term cert   │   │ pseudonym certs  │
   └──────────────────┘   └──────────────────┘
```

- **Long-term cert (LDEV)** — provisioned at manufacture / first activation;
  used only to request batches of pseudonym certs.
- **Pseudonym certs (PCert)** — each valid ~1 week, rotated every 5 min
  during driving. A batch is pre-provisioned so the vehicle can sign BSMs
  without online connectivity.
- **Revocation** — via a global CRL / RCL pushed in TIM; vehicles drop
  messages signed by revoked certs.
- **Misbehavior detection** — local heuristics (implausible position jumps,
  cert reuse across rotations) are reported to the MA; the MA can quarantine
  a device's entire cert pool.

### 8.2 Message Signing & Verification

Every outbound J2735 message is signed:

```
signed_msg = {
  payload : J2735Message,
  signer  : pcert_id,
  signature : ECDSA-P256(payload ‖ t),
  cert_chain : [PCert, PCA-cert]
}
```

Inbound verification pipeline:

1. **Parse** — reject malformed / oversized frames.
2. **Cert check** — verify PCert signature against PCA, check not in RCL,
   check validity window.
3. **Signature check** — verify ECDSA; reject on failure.
4. **Plausibility** — position within radio horizon? speed < 100 m/s?
   Reject implausible messages (do not feed to fusion).
5. **Rate limit** — per-source cap to mitigate spam.

Verification cost (~1 ms ECDSA on a hardware accelerator) is the main CPU
budget item; the stack pipelines verification across cores.

---

## 9. Fleet Learning — Non-RL (Shadow Mode)

### 9.1 Philosophy

Tesla's "shadow mode" does **not** train a policy with reinforcement
learning. Instead it collects vast quantities of **human-driven trajectories**
and aggregates them into:

- **HD map corrections** (lane geometry, sign positions).
- **Behavioral statistics** (what speed do humans take this curve at? how
  often do they yield here?).
- **Disengagement / conflict clips** for offline analysis and rule tuning.

The prototype mirrors this with a **rule-based** learning loop: statistics
are computed by classical aggregation (means, quantiles, histograms), not by
gradient descent on a policy.

### 9.2 Shadow-Mode Loop

```
┌──────────────┐   1. trigger event    ┌──────────────────┐
│  Ego vehicle │ ────────────────────→ │  Clip assembler  │
│  (driving)   │                       │  - 20 s context  │
│              │ ←──────────────────── │  - sensor frames │
│              │   2. shadow decision  │  - human action  │
│              │   (rule-based plan)   │  - shadow plan   │
└──────────────┘                       └────────┬─────────┘
                                                │ 3. upload (V2N)
                                                ▼
                                ┌──────────────────────────────┐
                                │  Fleet Cloud — Aggregator    │
                                │  - segment by geo-tile       │
                                │  - compute per-tile stats:   │
                                │      v_quantile, yield_rate, │
                                │      conflict_count          │
                                │  - diff against current map  │
                                └────────┬─────────────────────┘
                                         │ 4. curated update
                                         ▼
                                ┌──────────────────────────────┐
                                │  Map / Rule Update Store     │
                                │  - HD map diffs              │
                                │  - planner parameter packs   │
                                │  - OTA campaign manifest     │
                                └──────────────────────────────┘
                                         │ 5. OTA downlink
                                         ▼
                                     All vehicles
```

### 9.3 Trigger Conditions (event-based, not continuous)

A clip is uploaded only when one of these deterministic triggers fires,
keeping bandwidth bounded:

| Trigger | Condition | Clip length |
|---------|-----------|-------------|
| Disagreement | `shadow_plan != human_action` for > 1.5 s | 20 s |
| Planner low-confidence | `path_cost_variance > τ` | 20 s |
| Hard event | AEB, airbag, ABS > 0.5 s | 30 s |
| Map mismatch | vision lane vs HD map lane > 0.3 m | 15 s |
| New geo-tile | first vehicle in a tile this week | 30 s |

### 9.4 Aggregation (classical statistics, no gradient descent)

For each geo-tile (e.g., 10 m × 10 m) and lane:

```
stats[tile, lane] = {
  n_samples          : int,
  speed_quantiles    : [0.1, 0.5, 0.9],   # m/s
  yaw_rate_quantiles : [0.1, 0.5, 0.9],
  yield_rate         : float,              # fraction of clips where human yielded
  conflict_count     : int,                # disagreements with shadow plan
  lane_offset_mean   : float,
  lane_offset_std    : float,
}
```

These statistics feed back into the planner as **tuned parameters**, not as a
learned policy:

- `curve_speed_advisory[tile] ← speed_quantile[0.5]` (median human speed).
- `yield_probability_prior[tile] ← yield_rate` (used by the intersection FSM
  to set a conservative gap threshold).
- `lane_offset_bias[tile] ← lane_offset_mean` (corrects systematic HD map
  shift).

This is **imitation-of-the-median**, not RL: we take the central tendency of
human behavior as a prior and clamp it within hard safety constraints from
`05_path_planning.md`.

### 9.5 Privacy & Data Minimization

- Clips are stripped of GNSS precision to ~10 m before upload unless the
  clip is a map-mismatch (which needs full precision).
- No persistent driver identifiers; clips keyed by `session_hash` that
  rotates per trip.
- Faces and license plates are blurred at the edge (in-vehicle) before any
  frame leaves the car — privacy by design.

---

## 10. Data Pipeline (End-to-End)

```
┌─────────────┐  raw clips   ┌──────────────┐  curated   ┌──────────────┐
│ In-vehicle  │ ───────────→ │ Ingestion    │ ─────────→ │ Validation   │
│ shadow agent│              │ (Kafka-like) │            │ (schema,     │
└─────────────┘              └──────┬───────┘            │  plausibility)│
                                    │                    └──────┬───────┘
                                    │                           │
                                    ▼                           ▼
                            ┌──────────────┐          ┌──────────────┐
                            │ Dedup /      │          │ Aggregation  │
                            │ geo-segment  │          │ (per-tile    │
                            │ store        │          │  stats)      │
                            └──────────────┘          └──────┬───────┘
                                                              │
                                                ┌─────────────▼─────────────┐
                                                │  Update Authoring         │
                                                │  - map diff (vector tiles)│
                                                │  - param pack (JSON)      │
                                                │  - signed manifest        │
                                                └─────────────┬─────────────┘
                                                              │
                                                ┌─────────────▼─────────────┐
                                                │  OTA Campaign Manager     │
                                                │  - canary tiers           │
                                                │  - telemetry gate         │
                                                │  - auto-rollback          │
                                                └───────────────────────────┘
```

**Stages:**

1. **Ingestion** — durable queue, ordered by timestamp, deduped by
   `clip_hash`.
2. **Validation** — schema check (protobuf), plausibility check (speed <
   100 m/s, geo within fleet region), reject + quarantine on failure.
3. **Aggregation** — streaming job (e.g., Flink/Spark Structured Streaming)
   computing per-tile statistics on a rolling 7-day window.
4. **Authoring** — a human-in-the-loop review step signs off on map diffs
   and parameter packs before they become an OTA campaign (safety gate).
5. **Campaign** — staged rollout with telemetry gate; any regression in
   disengagement rate per 1k miles aborts and rolls back.

---

## 11. Mapping to the Existing Codebase

The prototype currently has no V2X hardware; this document defines the
**software integration points** so that adding a V2X dongle later requires
no planner changes.

| Component | Interface | File (planned) |
|-----------|-----------|----------------|
| V2X transport | `V2XTransport` ABC | `src/v2x/transport.py` |
| J2735 codec | `encode/decode` | `src/v2x/j2735.py` |
| LDM | `insert/query/expire` | `src/v2x/ldm.py` |
| Fusion adapter | `LDM → ObjectList` | `src/v2x/fusion_adapter.py` |
| SPaT consumer | `→ behavioral FSM` | `src/v2x/spat_consumer.py` |
| Shadow agent | `trigger → clip` | `src/fleet/shadow_agent.py` |
| Fleet uploader | `clip → MQTT` | `src/fleet/uploader.py` |

The fusion adapter converts V2V peers and V2P sources into the same
`TrackedObject` schema used by `src/alg/reference.py` and `src/alg/lka_step.py`,
so the MPC and safety override layers are unchanged.

---

## 12. Safety & Failure Modes

| Failure | Detection | Fallback |
|---------|-----------|----------|
| V2X radio loss | no BSM from peers for > 2 s | rely on vision/radar only; disable GLOSA |
| Cert expired / revoked | SCMS verification fails | drop peer; log; alert driver |
| RSU offline | no SPaT for known intersection | treat as unsignalized (yield FSM) |
| Cloud / V2N loss | MQTT reconnect timeout | queue clips locally; resume on reconnect |
| GPS spoofing via V2X | position implausible vs vision | reject peer; do not fuse |
| OTA rollback triggered | watchdog boot failure | revert to A partition; notify cloud |

**Core invariant:** V2X is always **augmentative, never authoritative** for
safety-critical decisions. Vision remains the primary sensor; V2X only adds
range and phase information that vision cannot see. AEB is never triggered
on V2X alone.

---

## 13. Summary

- Two PHY options (DSRC 802.11p, C-V2X NR) behind one `V2XTransport` ABC.
- Four modes: V2V (BSM/CACC), V2I (SPaT/MAP/TIM), V2N (cloud/OTA), V2P (PSM).
- Edge + 5G MEC closes safety loops in < 50 ms; cloud handles fleet learning
  and OTA on a seconds-to-hours budget.
- SCMS PKI gives pseudonymous, revocable, signed messages with ECDSA-P256.
- **Fleet learning is non-RL:** shadow mode collects human-driven clips,
  classical aggregation computes per-tile statistics, and a human-in-the-loop
  authoring step ships signed map/parameter updates via staged OTA.
- The planner (`05_path_planning.md`) is unchanged; V2X feeds the same
  `TrackedObject` list and behavioral FSM, preserving the classical,
  rule-based, no-RL design philosophy of the prototype.
