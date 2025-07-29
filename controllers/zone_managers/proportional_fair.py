from typing import Dict, List, Unpack
from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
from models.node.fog import FogLayerABC
from models.task import Task
import math

class ProportionalFairZoneManager(ZoneManagerABC):
    """
    Proportional-Fair scheduler: assigns tasks to maximize sum of log(bandwidth).
    Classical baseline for resource allocation.
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
        # Assign to node maximizing proportional fairness (log bandwidth)
        best_node = None
        best_pf_metric = float('-inf')
        for node in self._possible_fog_nodes:
            # Proportional fairness metric: log(current bandwidth + epsilon)
            # If node has bandwidth attribute, use it; else use remaining_power as proxy
            bandwidth = getattr(node, 'bandwidth', None)
            if bandwidth is None:
                bandwidth = getattr(node, 'remaining_power', 1.0)
            pf_metric = math.log(bandwidth + 1e-6)
            if pf_metric > best_pf_metric:
                best_pf_metric = pf_metric
                best_node = node
        return best_node if best_node else self._possible_fog_nodes[0] 