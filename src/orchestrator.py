"""
PipelineOrchestrator — runs pipeline nodes in sequence, passing FrameState through each.

Usage:
    orchestrator = PipelineOrchestrator()
    orchestrator.add_node(SensorNode(carla))
    orchestrator.add_node(PerceptionNode(pipeline))
    ...
    frame_state = orchestrator.run(frame_state, context)

If a node raises an exception, the orchestrator logs it and continues with the
previous state so a single node failure does not crash the whole loop.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from state import FrameState
from nodes import PipelineNode

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Runs nodes in sequence, passing FrameState through each."""

    def __init__(self) -> None:
        self._nodes: List[PipelineNode] = []

    def add_node(self, node: PipelineNode) -> "PipelineOrchestrator":
        """Append a node to the pipeline. Returns self for chaining."""
        self._nodes.append(node)
        return self

    @property
    def nodes(self) -> List[PipelineNode]:
        """Registered nodes (read-only view)."""
        return list(self._nodes)

    def run(self, initial_state: Optional[FrameState], context: dict) -> Optional[FrameState]:
        """Run all nodes in order, passing state + context through each.

        On node failure, logs the error and continues with the previous state.
        If a node returns ``None`` (e.g. SensorNode when no frame is available),
        the run short-circuits and returns ``None`` immediately so the caller
        can skip the rest of the frame.
        """
        state = initial_state
        for node in self._nodes:
            try:
                state = node.process(state, context)
            except Exception as e:
                logger.error(f"Node {node.name} failed: {e}")
                # Continue with previous state
            if state is None:
                # A node signalled "skip this frame" (e.g. no sensor data)
                return None
        return state
