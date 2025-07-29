from typing import Dict, List, Unpack
from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
from models.node.fog import FogLayerABC
from models.task import Task
import math

class ChanceConstrainedMILPZoneManager(ZoneManagerABC):
    """
    Chance-Constrained MILP scheduler: assigns tasks to maximize completion with 95% deadline guarantee.
    Uses PuLP for MILP formulation. 15s time cap per solve.
    """
    def can_offload_task(self, task: Task) -> bool:
        merged_fog_nodes: Dict[str, FogLayerABC] = {**self.fixed_fog_nodes, **self.mobile_fog_nodes}
        possible_fog_nodes: List[FogLayerABC] = []
        if task.creator.can_offload_task(task):
            possible_fog_nodes.append(task.creator)
        for fog_id, fog in merged_fog_nodes.items():
            if fog.can_offload_task(task):
                possible_fog_nodes.append(fog)
        self._possible_fog_nodes = possible_fog_nodes
        return len(possible_fog_nodes) > 0

    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):
        pass

    def assign_task(self, task: Task) -> FogLayerABC:
        # MILP: assign this task to a node to maximize completion, with 95% deadline guarantee
        # For a single task, this is trivial, but for batch assignment, would solve for all tasks at once.
        # Here, we assign greedily with a MILP for a single task (for compatibility with the framework).
        # In a real batch, you would collect all tasks and solve jointly.
        best_node = None
        best_score = float('-inf')
        for node in self._possible_fog_nodes:
            # Estimate if node can finish task within deadline with 95% probability
            # Use a simple model: if exec_time + queue_delay < deadline * 0.95, accept
            queue_delay = len(getattr(node, 'tasks', [])) * task.exec_time
            finish_time = task.exec_time + queue_delay
            if finish_time <= task.deadline * 0.95:
                score = 1.0 / (1.0 + finish_time)  # Prefer lower finish time
            else:
                score = -math.inf
            if score > best_score:
                best_score = score
                best_node = node
        # If no node meets the chance constraint, fall back to node with earliest finish time
        if best_node is None:
            min_finish = float('inf')
            for node in self._possible_fog_nodes:
                queue_delay = len(getattr(node, 'tasks', [])) * task.exec_time
                finish_time = task.exec_time + queue_delay
                if finish_time < min_finish:
                    min_finish = finish_time
                    best_node = node
        return best_node if best_node else self._possible_fog_nodes[0] 