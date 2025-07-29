# deep_rl_zone_manager_ddpg.py

import numpy as np
from typing import Unpack

from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
# ایمپورت کردن عامل DDPG به جای عامل DQN
from controllers.zone_managers.DDPG.ddpg_agent import DDPGAgent
from controllers.zone_managers.deepRL.deep_rl_env import DeepRLEnvironment
from controllers.zone_managers.heuristic import HeuristicZoneManager
from models.node.fog import FogLayerABC
from models.task import Task
from config import Config
from utils.distance import get_distance


class DeepRLZoneManager_DDPG(ZoneManagerABC):
    """
    A zone manager that uses Deep Deterministic Policy Gradient (DDPG) for task offloading.
    """

    def __init__(self, zone):
        super().__init__(zone)

        # مقداردهی اولیه عامل DDPG
        # state_dim: تعداد ویژگی‌های حالت (6)
        # action_dim: بعد فضای عمل (1، چون خروجی یک عدد پیوسته است)
        # max_action: حداکثر مقدار خروجی عمل (1.0، چون از تابع tanh استفاده می‌کنیم)
        self.agent = DDPGAgent(state_dim=6, action_dim=1, max_action=1.0)

        self.env = None

        # بارگذاری مدل از پیش آموزش‌دیده در صورت وجود
        try:
            self.agent.load_model("ddpg_model")
            print("[DDPG] Loaded pre-trained model.")
        except FileNotFoundError:
            print("[DDPG] No pre-trained model found, starting fresh.")

    def set_simulator(self, simulator):
        """
        Transfers the simulator reference to the environment.
        """
        self.env = DeepRLEnvironment(simulator)

    def can_offload_task(self, task: Task) -> bool:
        """
        Checks if there is an available node to offload the task.
        (This method remains unchanged)
        """
        available_nodes = list(self.fixed_fog_nodes.values()) + list(self.mobile_fog_nodes.values())
        return any(node.can_offload_task(task) for node in available_nodes)

    def assign_task(self, task: Task) -> FogLayerABC:
        """
        یک گره مناسب را برای وظیفه برمی‌گرداند.
        نکته: در معماری MADDPG ما، این متد مستقیماً توسط حلقه اصلی شبیه‌ساز فراخوانی نمی‌شود،
        زیرا تصمیم‌گیری به صورت متمرکز انجام می‌شود. اما برای سازگاری با کلاس پایه،
        یک پیاده‌سازی منطقی (پیدا کردن بهترین گره) در اینجا ارائه می‌شود.
        """
        return self._get_best_fog_node(task)

    def propose_candidate(self, task: Task, current_time: float):
        """
        Uses DDPG to decide where to offload a task.
        It suggests a node and returns the zone manager, the node, and the continuous action.
        """
        # دریافت حالت فعلی از محیط
        state = self.env._get_state(task)

        # انتخاب یک عمل پیوسته توسط عامل DDPG
        # خروجی یک آرایه است، ما به اولین عنصر آن نیاز داریم
        continuous_action = self.agent.select_action(state)[0]

        # نگاشت خروجی پیوسته (بین -1.0 و 1.0) به یک عمل گسسته (0, 1, 2)
        if continuous_action < -0.33:
            discrete_action = 0  # تصمیم: پردازش محلی (Local)
        elif continuous_action < 0.33:
            discrete_action = 2  # تصمیم: ارسال به ابر (cloud)
        else:
            discrete_action = 1  # تصمیم: ارسال به مه (Cloud)

        # انتخاب اجراکننده کاندید بر اساس عمل گسسته
        if discrete_action == 0:
            candidate_executor = task.creator
        elif discrete_action == 1:
            candidate_executor = self._get_best_fog_node(task)
        else:
            candidate_executor = self.env.simulator.cloud_node

        # برگرداندن خود مدیر منطقه، اجراکننده کاندید و عمل پیوسته اصلی
        # عمل پیوسته برای ذخیره در بافر تجربه (Replay Buffer) لازم است
        return self, candidate_executor, continuous_action

    def _get_best_fog_node(self, task):
        creator = task.creator
        all_fog_nodes = list(self.fixed_fog_nodes.values()) + list(self.mobile_fog_nodes.values())
        eligible_nodes = [node for node in all_fog_nodes if node.can_offload_task(task)]

        if not eligible_nodes:
            return None

        if len(eligible_nodes) == 1:
            return eligible_nodes[0]

        distances_by_id = {
            node.id: get_distance(node.x, node.y, creator.x, creator.y)
            for node in eligible_nodes
        }

        min_dist = min(distances_by_id.values())
        max_dist = max(distances_by_id.values())

        mobile_node_ids = {node.id for node in self.mobile_fog_nodes.values()}

        def calculate_score(node):
            if node.id in mobile_node_ids:
                normalized_power = node.power / Config.MobileFogNodeConfig.DEFAULT_COMPUTATION_POWER
            else:
                normalized_power = node.power / Config.FixedFogNodeConfig.DEFAULT_COMPUTATION_POWER

            distance = distances_by_id[node.id]

            if max_dist == min_dist:
                normalized_distance = 0.0
            else:
                normalized_distance = (distance - min_dist) / (max_dist - min_dist)

            distance_score = 1.0 - normalized_distance

            final_score = (0.5 * normalized_power) + (0.5 * distance_score)
            return final_score

        chosen_node = max(
            eligible_nodes,
            key=calculate_score,
            default=None
        )

        return chosen_node

    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):
        """
        Updates the DDPG agent by training it on a batch of experiences.
        """
        # آموزش عامل با یک بچ از تجربیات ذخیره شده
        # به‌روزرسانی شبکه‌های هدف به صورت "soft update" در داخل متد train انجام می‌شود
        self.agent.train()