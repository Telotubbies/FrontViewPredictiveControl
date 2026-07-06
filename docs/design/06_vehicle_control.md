# 06 — Vehicle Control Architecture

> **Scope:** Lateral + longitudinal control stack for a Tesla-like drive-by-wire
> prototype. **No reinforcement learning** is used anywhere in the stack. All
> controllers are model-based (MPC, PID, geometric) and tuned via system
> identification + numerical optimization.

---

## 1. Control Hierarchy

The control stack is organized as a layered hierarchy. Each layer runs at a
fixed cadence and communicates with the layer above and below through
well-defined interfaces.

```
┌──────────────────────────────────────────────────────────────┐
│  Planning Layer  (trajectory + speed profile)                │
│    • Path polynomial  (3rd order, Frenet)                     │
│    • Reference speed v_ref(s)                                 │
└───────────────────────────┬──────────────────────────────────┘
                            │  trajectory + v_ref + curvature
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  Supervisory Arbitration  (10 Hz)                            │
│    • Mode: NORMAL / ACC / AEB / ESC / MANUAL_OVERRIDE         │
│    • Selects active lateral & longitudinal controllers        │
└───────────────────────────┬──────────────────────────────────┘
                            │  mode + active setpoints
              ┌─────────────┴──────────────┐
              ▼                            ▼
┌─────────────────────────────┐  ┌──────────────────────────────┐
│  Lateral Controller (50 Hz) │  │  Longitudinal Controller     │
│                             │  │  (50 Hz)                     │
│  ┌───────────────────────┐  │  │  ┌────────────────────────┐  │
│  │ MPC (primary)         │  │  │  │ PID + Feedforward      │  │
│  │  CasADi / IPOPT       │  │  │  │  + ACC distance loop   │  │
│  ├───────────────────────┤  │  │  ├────────────────────────┤  │
│  │ Pure Pursuit (fallback)│ │  │  │ AEB override (safety)  │  │
│  ├───────────────────────┤  │  │  └────────────────────────┘  │
│  │ Stanley (low-speed)   │  │  │                              │
│  └───────────────────────┘  │  │                              │
└──────────────┬──────────────┘  └──────────────┬───────────────┘
               │ delta_cmd (rad)                 │ a_cmd (m/s²)
               ▼                                 ▼
┌──────────────────────────────────────────────────────────────┐
│  Actuator Interface  (100 Hz)                                │
│    • Jerk limiting & rate saturation                         │
│    • CAN frame packing (steering, throttle, brake)           │
│    • Drive-by-wire ECU handshake                             │
└───────────────────────────┬──────────────────────────────────┘
                            ▼
                   Vehicle Plant (bicycle model)
```

### 1.1 Timing budget

| Layer             | Rate   | Latency budget | Jitter max |
|-------------------|--------|----------------|------------|
| Supervisory       | 10 Hz  | 50 ms          | 10 ms      |
| Lateral MPC       | 50 Hz  | 15 ms          |  3 ms      |
| Longitudinal PID  | 50 Hz  |  5 ms          |  2 ms      |
| Actuator CAN TX   | 100 Hz |  2 ms          |  1 ms      |

The MPC solve must complete within 15 ms. If IPOPT fails to converge within
the budget, the supervisor falls back to Pure Pursuit for that tick and
increments a `mpc_fallback` counter. Three consecutive fallbacks trigger a
controlled deceleration and a fault report.

---

## 2. Vehicle Plant — Kinematic Bicycle Model

The control stack uses a kinematic bicycle model as the prediction model. It
is accurate for highway speeds on dry pavement and keeps the MPC QP small
enough for real-time IPOPT solves. A dynamic (lateral-slip) extension is
available for the ESC module (Section 6).

```
        │ lf │      │ lr │
   ◄────┤    ├───────┤    ├────►
        │    │       │    │
        ▼    ▼       ▼    ▼
       front axle   rear axle
        delta (steer)
```

### 2.1 Continuous-time equations

```
ẋ   = v · cos(psi + β)
ẏ   = v · sin(psi + β)
ψ̇  = (v / L) · sin(β)
v̇  = a

β  = arctan( (lr / L) · tan(delta) )
L  = lf + lr          (wheelbase)
```

For the prototype (Tesla Model 3 geometry, `src/control/pure_pursuit.py`):

```
L   = 2.875 m
lr  = 1.40  m
lf  = 1.475 m
```

### 2.2 Discrete-time integration (for MPC)

Forward Euler is used inside CasADi for speed; a Runge–Kutta 4 integrator is
available but doubles solve time and is reserved for validation:

```
x_{k+1} = x_k + dt · v_k · cos(psi_k + β_k)
y_{k+1} = y_k + dt · v_k · sin(psi_k + β_k)
psi_{k+1} = psi_k + dt · (v_k / L) · sin(β_k)
v_{k+1} = v_k + dt · a_k
```

State vector:  **z = [x, y, psi, v]ᵀ**
Control vector: **u = [delta, a]ᵀ**

This matches the implementation in `src/control/lane_mpc.py` lines 7–8.

---

## 3. Lateral Control

Three lateral controllers are implemented and arbitrated by the supervisor:

| Controller    | Role                        | Best regime               |
|---------------|-----------------------------|---------------------------|
| MPC           | Primary, high-fidelity      | All speeds, curves        |
| Pure Pursuit  | Fallback / safety           | Straight + gentle curves  |
| Stanley       | Low-speed parking / creep   | v < 3 m/s                 |

### 3.1 MPC (primary)

See `src/control/lane_mpc.py`. The MPC is the primary lateral controller
because it jointly optimizes tracking, smoothness, and actuator limits.

#### 3.1.1 Cost function

```
J = Σ_{k=0}^{N-1}  [
        w_cte   · (CTE_k)²                          # cross-track error
      + w_head  · (e_psi_k)²                        # heading error
      + w_v    · (v_k - v_ref_k)²                   # speed tracking
      + w_steer· (delta_k)²                         # control effort
      + w_acc  · (a_k)²                             # control effort
      + w_dsteer· (delta_k - delta_{k-1})²          # steering rate
      + w_jerk · (a_k - a_{k-1})²                   # longitudinal jerk
    ]
  + terminal weights on (CTE_N, e_psi_N, v_N - v_ref_N)
```

Weights are **gain-scheduled** (see `get_mpc_weights` in `lane_mpc.py`):

```
scale_cte        = 1.0 + 1.2·curve_factor + 0.5·speed_norm
scale_heading    = 1.0 + 0.8·curve_factor + 0.3·speed_norm
scale_steer_rate = max(0.4, 1.0 + 0.4·(1-conf) - 0.4·curve_factor)
scale_jerk       = 1.0 + 0.3·(1-conf)
```

Rationale (from research notes in the source):
- **Curves:** increase CTE weight for tracking accuracy.
- **High speed:** increase CTE weight to compensate for inertia lag.
- **Straight roads:** reduce weights for comfort.
- **Low confidence:** increase steer-rate penalty to suppress jitter.

#### 3.1.2 Constraints

```
|delta_k|          ≤ 0.7 rad            (~40°, max_steer)
|delta_k - delta_{k-1}| / dt ≤ max_steer_rate
a_min              ≤ a_k ≤ a_max
v_min              ≤ v_k ≤ v_max
soft slack on CTE  ≤ slack_max  (penalized quadratically)
```

#### 3.1.3 Horizon

Horizon `N` is dynamic (`compute_mpc_horizon`):

```
if |curvature| ≥ 0.025 1/m        → N = 8   (sharp curve, short horizon)
elif speed ≥ 5.6 m/s              → N = 12  (high speed, long horizon)
else                              → N = 8
dt = 0.1 s  →  horizon 0.8–1.2 s
```

Short horizons in sharp curves keep the QP small and avoid the optimizer
"cutting" the curve. Long horizons at high speed give preview for smooth
steer-in.

#### 3.1.4 Solver

- **Tool:** CasADi symbolic framework.
- **NLP solver:** IPOPT with MA57 linear solver.
- **Warm start:** previous solution shifted by one step.
- **Tolerance:** `tol = 1e-4`, `acceptable_tol = 1e-3`.
- **Max iterations:** 50 (hard cap to bound worst-case latency).
- **Failure policy:** if not converged, reuse previous `delta_cmd` and
  decrement a health counter; after 3 misses, switch to Pure Pursuit.

### 3.2 Pure Pursuit (fallback)

Geometric controller (`src/control/pure_pursuit.py`). Steering law:

```
delta = arctan( 2 · L · sin(alpha) / L_d )
```

where `alpha` is the angle from the vehicle heading to the lookahead point
and `L_d` is the lookahead distance:

```
L_d = clip( k · v,  L_min,  L_max )
L_min = 4.0 m,  L_max = 20.0 m,  k = 0.6
```

Curvature-adaptive lookahead: when `|κ| > 0.02`, `L_max` is reduced by 30%
so sharper curves use a shorter lookahead.

Output is smoothed with an exponential moving average:

```
delta_smooth = alpha · delta_raw + (1 - alpha) · delta_prev
```

with speed-dependent `alpha` (low speed → responsive, high speed → smooth).

### 3.3 Stanley (low-speed)

Used below 3 m/s (parking, creep, queue discharge). The Stanley law:

```
delta = e_psi + arctan( k_e · e_y / v )
```

- `e_psi`: heading error to the nearest path segment.
- `e_y`: cross-track error (lateral).
- `k_e`: gain (tuned ~1.0).

At very low speed the `arctan` term saturates, giving full corrective steer
without the oscillation that Pure Pursuit exhibits when `L_d` collapses.

### 3.4 Lateral transfer function (small-signal)

Linearizing the bicycle model about a straight-line operating point
(`v = v0`, `delta = 0`, `psi = 0`) yields the lateral dynamics:

```
G_yδ(s) = Y(s)/Δ(s) = (v0²·lr/L) / ( s²·(v0·L) + s·(v0²) )
```

i.e. a double integrator with a velocity-dependent damping term. This is why
lateral control gets harder at low speed (damping → 0) and why Stanley is
preferred in that regime.

---

## 4. Longitudinal Control

### 4.1 PID + Feedforward

The longitudinal controller tracks a reference speed `v_ref(s)` using a PI
controller with feedforward acceleration and a derivative-on-measurement term:

```
a_cmd = a_ff + Kp · e_v + Ki · ∫e_v dt − Kd · dv/dt
e_v   = v_ref − v
a_ff  = v_ref · dv_ref/ds          (centripetal + grade feedforward)
```

Discrete form (50 Hz, `dt = 0.02 s`):

```
a_cmd[k] = a_ff[k] + Kp·e[k] + Ki·T_s·Σe[j] + Kd·(v[k]−v[k−1])/T_s
```

Transfer function (plant approximated as first-order lag from `a` to `v`):

```
G_va(s) = V(s)/A(s) = 1 / (s·τ_a + 1)        (τ_a ≈ 0.3 s driveline)
```

Closed loop with PI:

```
T_v(s) = (Kp·s + Ki) / ( τ_a·s² + (1+Kp)·s + Ki )
```

Tuning targets: bandwidth ~2 rad/s, damping ratio ζ ≥ 0.8.

### 4.2 Adaptive Cruise Control (ACC)

When a leading vehicle is detected, the speed setpoint is replaced by a
spacing controller (constant-time-gap):

```
d_ref = d0 + h · v
e_d   = d_ref − d_lead
v_ref_acc = v_lead + K_acc · e_d
v_ref = min(v_ref_speed_limit, v_ref_acc)
```

- `d0 = 4.0 m` (standstill gap)
- `h  = 1.2 s` (time gap)
- `K_acc = 0.8` (spacing gain, 1/s)

The ACC loop is a cascade: outer spacing loop produces `v_ref`, inner PID
produces `a_cmd`. This keeps the inner loop identical in normal and ACC modes.

```
   d_lead ──►(−)──► K_acc ──► v_ref ──► PID ──► a_cmd ──► plant ──► v
              ↑                                              │
            d_ref = d0 + h·v ◄────────────────────────────────┘
```

---

## 5. MPC Design Details

### 5.1 Problem formulation (NLP)

```
minimize   J(z, u)                         (cost, Section 3.1.1)
  z,u

subject to z_0 = z_meas                    (initial condition)
           z_{k+1} = f_bike(z_k, u_k)     (bicycle dynamics, RK1)
           |delta_k| ≤ 0.7
           |Δdelta_k| ≤ max_steer_rate·dt
           a_min ≤ a_k ≤ a_max
           v_min ≤ v_k ≤ v_max
           slack_cte ≥ 0   (soft)
```

### 5.2 IPOPT options

| Option              | Value    | Reason                          |
|---------------------|----------|---------------------------------|
| `print_level`       | 0        | silence in production           |
| `max_iter`          | 50       | latency cap                     |
| `tol`               | 1e-4     | tracking accuracy               |
| `acceptable_tol`    | 1e-3     | early accept on tight ticks     |
| `linear_solver`     | MA57     | fast sparse LU                  |
| `warm_start_init_point` | yes   | reuse prior solution            |
| `mu_strategy`       | adaptive | robustness across speeds        |

### 5.3 Warm-start strategy

At tick `t`, the optimizer is seeded with the shifted solution from `t−1`:

```
z_seed[0:N-1] = z_opt[1:N],   z_seed[N-1] = z_opt[N-1]
u_seed[0:N-1] = u_opt[1:N],   u_seed[N-1] = u_opt[N-1]
```

This typically cuts IPOPT iterations from ~20 (cold) to ~5 (warm).

### 5.4 Soft constraints

Hard constraints on CTE can make the NLP infeasible when the planner emits a
path the bicycle model cannot follow within `max_steer`. CTE is therefore
soft:

```
J += w_slack · Σ slack_k²
0 ≤ slack_k ≤ slack_max
```

Steering and acceleration limits remain **hard** (safety-critical).

---

## 6. Emergency & Safety Control

### 6.1 Autonomous Emergency Braking (AEB)

AEB bypasses the longitudinal PID and commands maximum deceleration when a
collision is imminent. Trigger uses Time-To-Collision (TTC):

```
TTC = d_lead / (v − v_lead)        (if v > v_lead)
```

- `TTC < 1.5 s` → **brake alert** (prefill, audible warning).
- `TTC < 0.8 s` → **AEB fire**: `a_cmd = −a_max_brake` (≈ −6 m/s²).
- Below 30 km/h, pedestrian-in-path extends the trigger envelope by 0.2 s.

AEB is a hard override: it sets a latched flag that the supervisor honors
until `v < 0.5 m/s` and the obstacle clears.

### 6.2 Electronic Stability Control (ESC)

ESC monitors the yaw-rate error against the bicycle-model reference:

```
ψ̇_ref = (v / L) · tan(delta) · cos(β)
e_r   = ψ̇_ref − ψ̇_measured
```

If `|e_r|` exceeds a slip threshold (indicating under/over-steer), ESC
intervenes with **differential braking** (not steering), reducing torque to
the inside wheel:

```
ΔM_brake = K_esc · e_r − D_esc · ψ̇
```

- Understeer (`e_r < 0`): brake inside-rear to induce yaw-in.
- Oversteer (`e_r > 0`): brake outside-front to straighten.

ESC uses the **dynamic** bicycle model (lateral force = linear tire):

```
m·(V̇y + Vx·r) = −F_yf − F_yr
I_z·ṙ = lf·F_yf − lr·F_yr
F_yi = −C_αi · α_i,   α_f = δ − (Vy + lf·r)/Vx
```

### 6.3 Fallback ladder

```
MPC ok            → use MPC delta
MPC fail (1–2×)   → reuse last MPC delta
MPC fail (3×)     → Pure Pursuit + warn
PP fail           → Stanley + decelerate to 5 m/s
Any lateral fail  → AEB-grade stop if obstacle, else minimal-risk maneuver
```

---

## 7. Actuator Interface

### 7.1 CAN bus layout

| ID      | Direction | Signal                 | Rate  |
|---------|-----------|------------------------|-------|
| 0x257   | TX        | Steering angle request | 100 Hz|
| 0x118   | TX        | Accelerator pedal      | 100 Hz|
| 0x119   | TX        | Brake pressure request | 100 Hz|
| 0x1F8   | RX        | Vehicle state (v, yaw) | 100 Hz|
| 0x2A0   | RX        | Steering feedback      | 100 Hz|
| 0x3F0   | RX        | ECU heartbeat          | 10 Hz |

### 7.2 Drive-by-wire handshake

```
1. Supervisor asserts ENABLE_DBW.
2. Interface waits for ECU heartbeat (0x3F0) with status = READY.
3. On READY, send a zero-torque steering + zero accel frame for 100 ms.
4. ECU echoes back ACTIVE; interface then forwards controller commands.
5. Watchdog: if heartbeat missing > 100 ms → DISABLE_DBW, revert to manual.
```

### 7.3 Command packing

```
steer_can = int16( delta_cmd / 0.7 · 1500 )      # ±1500 counts = ±0.7 rad
accel_can = uint8( clip(a_cmd, 0, a_max) / a_max · 255 )
brake_can = uint8( clip(−a_cmd, 0, a_brake_max) / a_brake_max · 255 )
```

Only one of `accel_can` / `brake_can` is non-zero at a time (mutually
exclusive torque requests).

---

## 8. Smoothness — Jerk Limiting

Passenger comfort requires bounded jerk. Two mechanisms:

### 8.1 Steering jerk limiter

A second-order rate limiter on `delta_cmd`:

```
ΔΔδ_max = max_steer_jerk · dt²
delta_cmd = clamp( delta_cmd,
                   prev + ΔΔδ_max,
                   prev − ΔΔδ_max )
```

This is applied **after** the controller, so the MPC can plan aggressive
moves but the actuator only sees physically realizable transitions.

### 8.2 Longitudinal jerk limiter

```
a_cmd = clamp( a_cmd,
               a_prev + j_max·dt,
               a_prev − j_max·dt )
j_max = 2.5 m/s³   (comfort),  5.0 m/s³  (AEB)
```

### 8.3 Transfer-function view

The jerk limiter is equivalent to a series saturation + first-order lag:

```
G_jerk(s) = 1 / ( (s/j_max) + 1 )        (approx, small signal)
```

This adds ~`1/j_max` seconds of phase delay, which the MPC preview horizon
must exceed (it does: 0.8–1.2 s ≫ 0.4 s).

---

## 9. Tuning Methodology (No RL)

All gains are derived from **system identification + numerical optimization**.
No reinforcement learning, no learning-based tuning loops.

### 9.1 System identification

1. **Step-response tests** on the drive-by-wire rig:
   - Steering step: command `delta = 0.3 rad`, record yaw rate.
   - Accel step: command `a = 2 m/s²`, record speed.
2. Fit transfer functions:
   - Steering: `G_δψ̇(s) = K_δ / (τ_δ·s + 1)`  →  identify `K_δ, τ_δ`.
   - Driveline: `G_av(s) = 1 / (τ_a·s + 1)`     →  identify `τ_a`.
3. **Parameter estimation** via least-squares on logged data:

```
θ̂ = argmin_θ  Σ ( y_meas − f(θ, u) )²
```

using scipy `least_squares` or CasADi `nlpsol` with the bicycle model as `f`.

### 9.2 PID tuning

From the identified `τ_a`, use internal-model-control (IMC) rules:

```
Kp = τ_a / (τ_c · K_a)
Ki = Kp / τ_a
Kd = 0   (PI is sufficient for the first-order driveline)
```

`τ_c` (desired closed-loop time constant) is chosen as `τ_a / 4` for a
bandwidth 4× the plant.

### 9.3 MPC weight optimization

MPC weights are tuned via **offline trajectory optimization** over a
validation set of recorded scenarios:

```
minimize_{w}   Σ_{scenario} Σ_{t}  [ Q·CTE² + R·jerk² + S·heading_err² ]
subject to     MPC closed-loop sim with weights w
               actuator limits respected
               w within bounds [w_lo, w_hi]
```

- Solver: CMA-ES or scipy `differential_evolution` (gradient-free, robust).
- No policy gradient, no agent-environment loop — purely a simulation-based
  black-box optimization over the weight vector.
- The gain-scheduled structure (Section 3.1.1) is fixed; only the scalar
  coefficients of `curve_factor`, `speed_norm`, `confidence` are optimized.

### 9.4 Validation gates

Every tuning change must pass:

| Gate                | Metric                          | Threshold |
|---------------------|---------------------------------|-----------|
| Tracking            | max CTE on curve set            | < 0.30 m  |
| Comfort             | 95th-pct lateral jerk           | < 2.0 m/s³|
| Stability           | yaw-rate overshoot (lane change)| < 15%     |
| Robustness          | CTE on ±20% mass perturbation   | < 0.45 m  |
| Latency             | 99th-pct MPC solve time         | < 15 ms   |

Failure on any gate reverts to the previous tuning commit.

---

## 10. Data Flow Summary

```
 Perception ──► Planner ──► Supervisor ──┬─► MPC ──────────┐
                                         ├─► Pure Pursuit ─┤
                                         ├─► Stanley ──────┤
                                         ├─► PID+FF/ACC ───┤
                                         ├─► AEB ──────────┤
                                         └─► ESC ──────────┘
                                                              │
                                          Jerk limiter ◄──────┘
                                               │
                                          CAN interface
                                               │
                                          Drive-by-wire ECU
                                               │
                                          Vehicle
```

All blocks are deterministic, model-based, and tunable through system
identification and numerical optimization — consistent with the project's
**no-RL** constraint.
