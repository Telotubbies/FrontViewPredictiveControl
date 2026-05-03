# Simplified Workflow Proposal

## Current Issues
1. **Too many abstraction layers** - 3 pipeline levels
2. **Duplicate data structures** - PerceptionResult in 2 places
3. **Mixed config systems** - old config.py + new config_clean.py
4. **Complex control flow** - main → system → runner → pipeline

## Proposed Simplified Architecture

```
main.py (entry point)
├── CarlaManager (carla + vehicle + camera)
├── PerceptionManager (UNet/Classical)
├── ControlManager (MPC + safety)  
└── DisplayManager (GUI + logging)
```

### Key Changes

#### 1. Single Data Structures
```python
# Use only utils/type_hints.py definitions
# Remove duplicate classes from mpc_runner.py
```

#### 2. Direct Component Communication
```python
# Instead of: main → system → runner → pipeline → trajectory
# Use: main → direct component calls
```

#### 3. Unified Configuration
```python
# Use only config_clean.py
# Remove old config.py imports
```

#### 4. Simplified Main Loop
```python
def main_loop():
    while running:
        # Get frame
        frame = carla.get_frame()
        
        # Process perception
        perception = perception_manager.process(frame)
        
        # Compute control
        control = control_manager.compute(perception, vehicle_state)
        
        # Apply control
        carla.apply_control(control)
        
        # Update display
        display.update(frame, perception, control)
```

## Benefits
- **50% less code** in main loop
- **Single source of truth** for data structures
- **Easier debugging** with direct calls
- **Better performance** with less overhead

## Migration Steps
1. Remove duplicate PerceptionResult classes
2. Consolidate pipeline layers
3. Switch to config_clean.py everywhere
4. Simplify main.py control loop
