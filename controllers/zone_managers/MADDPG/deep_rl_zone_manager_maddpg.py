from typing import Unpack

from config import Config
from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
from models.node.fog import FogLayerABC
from models.task import Task
from models.zone import Zone
from controllers.zone_managers.heuristic import HeuristicZoneManager
from utils.distance import get_distance


class DeepRLZoneManagerMADDGP(ZoneManagerABC):
    """
    یک ZoneManager که از یک کنترلر مرکزی MADDPG برای تصمیم‌گیری استفاده می‌کند.
    این کلاس با پیاده‌سازی تمام متدهای انتزاعی ZoneManagerABC، با ساختار پایه سازگار است.
    """

    def __init__(self, zone: Zone):
        super().__init__(zone)
        # این مقادیر توسط شبیه‌ساز پس از مقداردهی اولیه، تنظیم می‌شوند
        self.agent_id: int = -1
        self.controller = None

    def can_offload_task(self, task: Task) -> bool:
        """
        بررسی می‌کند که آیا حداقل یک گره مه (ثابت یا متحرک) برای تخلیه وظیفه در دسترس است.
        """
        all_fog_nodes = list(self.fixed_fog_nodes.values()) + list(self.mobile_fog_nodes.values())
        return any(node.can_offload_task(task) for node in all_fog_nodes)

    def assign_task(self, task: Task) -> FogLayerABC:
        """
        یک گره مناسب را برای وظیفه برمی‌گرداند.
        نکته: در معماری MADDPG ما، این متد مستقیماً توسط حلقه اصلی شبیه‌ساز فراخوانی نمی‌شود،
        زیرا تصمیم‌گیری به صورت متمرکز انجام می‌شود. اما برای سازگاری با کلاس پایه،
        یک پیاده‌سازی منطقی (پیدا کردن بهترین گره) در اینجا ارائه می‌شود.
        """
        print("gand zadi")
        return self._get_best_fog_node(task)

    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):
        """
        متد آپدیت برای عامل.
        در معماری MADDPG، منطق یادگیری و به‌روزرسانی به صورت متمرکز در MADDPGController
        انجام می‌شود. بنابراین، این متد برای هر عامل منفرد، نیازی به پیاده‌سازی خاصی ندارد.
        """
        pass

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