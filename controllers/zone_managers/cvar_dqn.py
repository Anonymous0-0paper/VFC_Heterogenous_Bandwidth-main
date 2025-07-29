from typing import Dict, List, Tuple, Unpack
from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
from models.node.fog import FogLayerABC
from models.task import Task
import numpy as np
import random

class TabularDoubleQ:
    """Tabular double Q-learning for discrete state-action pairs."""
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99):
        self.Q1 = np.zeros((n_states, n_actions))
        self.Q2 = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma

    def update(self, s, a, r, s_next, a_next):
        if random.random() < 0.5:
            max_a = np.argmax(self.Q1[s_next])
            td_target = r + self.gamma * self.Q2[s_next, max_a]
            self.Q1[s, a] += self.alpha * (td_target - self.Q1[s, a])
        else:
            max_a = np.argmax(self.Q2[s_next])
            td_target = r + self.gamma * self.Q1[s_next, max_a]
            self.Q2[s, a] += self.alpha * (td_target - self.Q2[s, a])

    def get_q(self, s, a):
        return (self.Q1[s, a] + self.Q2[s, a]) / 2

    def get_qs(self, s):
        return (self.Q1[s] + self.Q2[s]) / 2

    def cvar_action(self, s, alpha=0.95):
        qs = self.get_qs(s)
        # For tabular, CVaR is just the lower quantile of Q-values
        sorted_qs = np.sort(qs)
        k = int(np.ceil(alpha * len(sorted_qs)))
        return np.argmin(qs[:k])


class CVaRDQNZoneManager(ZoneManagerABC):
    def __init__(self, zone, n_queue=10, n_band=10, alpha=0.995):
        super().__init__(zone)
        self.n_queue = n_queue
        self.n_band = n_band
        self.alpha = alpha
        self.state_map = {}
        self.action_map = {}
        self.reverse_action_map = {}
        self.q_table = None
        self._possible_fog_nodes = []
        # Delay Q-table init until we know the available nodes
        self.n_actions = 1
        self.n_states = self.n_queue * self.n_band * self.n_actions

    def _init_action_space(self):
        # Rebuild action map based on available fog nodes
        self.action_map.clear()
        self.reverse_action_map.clear()

        idx = 0
        for node in self._possible_fog_nodes:
            if node.id not in self.action_map:
                self.action_map[node.id] = idx
                self.reverse_action_map[idx] = node.id
                idx += 1

        self.n_actions = max(1, len(self.action_map))
        self.n_states = self.n_queue * self.n_band * self.n_actions
        self.q_table = TabularDoubleQ(self.n_states, self.n_actions)

    def _discretize_state(self, node, task):
        queue_bin = min(self.n_queue - 1, len(getattr(node, 'tasks', [])))
        bandwidth = getattr(node, 'bandwidth', 1.0)
        band_bin = min(self.n_band - 1, int(bandwidth // (1000 / self.n_band)))
        node_idx = self.action_map.get(node.id, 0)
        state_idx = queue_bin * self.n_band * self.n_actions + band_bin * self.n_actions + node_idx
        return state_idx

    def can_offload_task(self, task: Task) -> bool:
        merged_fog_nodes: Dict[str, FogLayerABC] = {**self.fixed_fog_nodes, **self.mobile_fog_nodes}
        possible_fog_nodes: List[FogLayerABC] = []

        if task.creator and task.creator.can_offload_task(task):
            possible_fog_nodes.append(task.creator)

        for fog in merged_fog_nodes.values():
            if fog.can_offload_task(task):
                possible_fog_nodes.append(fog)

        self._possible_fog_nodes = possible_fog_nodes
        self._init_action_space()
        return len(possible_fog_nodes) > 0

    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):
        pass  # Future experience updates here

    def assign_task(self, task: Task) -> FogLayerABC:
        best_node = None
        best_cvar = float('-inf')

        for node in self._possible_fog_nodes:
            s = self._discretize_state(node, task)
            a = self.action_map.get(node.id, 0)
            q_val = self.q_table.get_q(s, a)
            if q_val > best_cvar:
                best_cvar = q_val
                best_node = node

        return best_node if best_node else self._possible_fog_nodes[0]
