import random
from collections import defaultdict
from typing import Dict, List

from NoiseConfigs.utilsFunctions import UtilsFunc
from config import Config
from controllers.loader import Loader
from controllers.metric import MetricsController
from controllers.zone_managers.base import ZoneManagerABC
from controllers.zone_managers.DDPG.deep_rl_zone_manager_ddpg import DeepRLZoneManager_DDPG
from models.node.base import MobileNodeABC, NodeABC
from models.node.cloud import CloudNode
from models.node.fog import FixedFogNode
from models.node.fog import MobileFogNode
from models.node.user import UserNode
from models.task import Task
from utils.clock import Clock
from utils.enums import Layer

from controllers.finalChoice import FinalChoice

import sys
import os
import pandas as pd

sys.path.append(os.path.abspath("E:/VANET - Copy/NoiseConfigs"))


# توابع رنگی برای لاگ در ترمینال
def yellow_bg(text):
    return f"\033[43m{text}\033[0m"


def red_bg(text):
    return f"\033[41m{text}\033[0m"


def blue_bg(text):
    return f"\033[44m{text}\033[0m"


def green_bg(text):
    return f"\033[42m{text}\033[0m"


class SimulatorDDPG:
    def __init__(self, loader: Loader, clock: Clock, cloud: CloudNode):
        self.metrics: MetricsController = MetricsController()
        self.loader: Loader = loader
        self.cloud_node: CloudNode = cloud
        self.zone_managers: Dict[str, ZoneManagerABC] = {}
        self.fixed_fog_nodes: Dict[str, FixedFogNode] = {}
        self.mobile_fog_nodes: Dict[str, MobileFogNode] = {}
        self.user_nodes: Dict[str, UserNode] = {}
        self.clock: Clock = clock
        self.task_zone_managers: Dict[str, ZoneManagerABC] = {}
        self.retransmission_tasks: Dict[float, List[Task]] = {}
        self.missed_deadline_data: List[Dict] = []
        self.success_deadline_data: List[Dict] = []

    def init_simulation(self):
        self.clock.set_current_time(0)
        self.zone_managers = self.loader.load_zones()
        self.fixed_fog_nodes = self.loader.load_fixed_zones()
        self.assign_fixed_nodes()
        self.update_mobile_fog_nodes_coordinate()
        self.update_user_nodes_coordinate()
        for zm in self.zone_managers.values():
            if hasattr(zm, "set_simulator"):
                zm.set_simulator(self)

    def schedule_retransmission(self, task: Task, scheduled_time: float):
        if scheduled_time not in self.retransmission_tasks:
            self.retransmission_tasks[scheduled_time] = []
        self.retransmission_tasks[scheduled_time].append(task)

    def assign_fixed_nodes(self):
        for z_id, zone_manager in self.zone_managers.items():
            fixed_nodes: List[FixedFogNode] = []
            for n_id, fixed_node in self.fixed_fog_nodes.items():
                if zone_manager.zone.is_in_coverage(fixed_node.x, fixed_node.y):
                    fixed_nodes.append(fixed_node)
            zone_manager.add_fixed_fog_nodes(fixed_nodes)

    def retransmission(self, zone_managers, current_time):
        tasks_to_retransmit = []
        for scheduled_time in list(self.retransmission_tasks.keys()):
            if scheduled_time <= current_time:
                tasks_to_retransmit.extend(self.retransmission_tasks.pop(scheduled_time))

        if tasks_to_retransmit:
            for task in tasks_to_retransmit:
                possible_zone_managers = self.find_zone_manager_offload_task(zone_managers, task, current_time)
                if self.choose_executor_and_assign(possible_zone_managers, task, current_time):
                    continue

    def find_zone_manager_offload_task(self, zone_managers, task, current_time):
        zone_manager_offload_task = []
        for zone_manager in zone_managers:
            if zone_manager.can_offload_task(task):
                if hasattr(zone_manager, "propose_candidate"):
                    # <<< CHANGED: دریافت عمل پیوسته از متد propose_candidate
                    # این متد در نسخه DDPG سه مقدار برمی‌گرداند
                    proposed_zone_manager, proposed_executor, continuous_action = zone_manager.propose_candidate(task,
                                                                                                                 current_time)
                else:
                    proposed_zone_manager = zone_manager
                    proposed_executor = zone_manager.offload_task(task, current_time)
                    continuous_action = None  # برای الگوریتم‌های دیگر مقدار پیش‌فرض

                if proposed_executor and (
                proposed_zone_manager, proposed_executor, continuous_action) not in zone_manager_offload_task:
                    # +++ ADDED: ذخیره کردن عمل پیوسته همراه با کاندیدا
                    zone_manager_offload_task.append((proposed_zone_manager, proposed_executor, continuous_action))
        return zone_manager_offload_task

    def choose_executor_and_assign(self, zone_manager_offload_task, task, current_time):
        if len(zone_manager_offload_task) != 0:
            # print(green_bg("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
            finalCandidates = []

            for candidate in zone_manager_offload_task:
                # <<< CHANGED: استخراج عمل پیوسته از لیست کاندیداها
                zone_manager, candidate_executor, continuous_action = candidate

                # +++ ADDED: اضافه کردن عمل پیوسته به لیست تضعیف سیگنال برای انتخاب نهایی
                finalCandidates.append((zone_manager, candidate_executor, continuous_action))

            # فرض می‌شود که `makeFinalChoice` می‌تواند مقدار چهارم (continuous_action) را مدیریت کند
            finalChoiceToOffload = FinalChoice().makeFinalChoice(finalCandidates, Config.FinalDeciderMethod.DEFAULT_METHOD)

            # <<< CHANGED: استخراج عمل پیوسته از انتخاب نهایی
            # فرض بر این است که انتخاب نهایی شامل چهار مقدار است
            chosen_zone_manager, chosen_executor, chosen_continuous_action = finalChoiceToOffload


            self.task_zone_managers[task.id] = chosen_zone_manager
            self.metrics.inc_node_tasks(chosen_executor.id)
            if isinstance(chosen_zone_manager, DeepRLZoneManager_DDPG):
                state = chosen_zone_manager.env._get_state(task)
                reward, _ = chosen_zone_manager.env._compute_reward2(task, chosen_executor, current_time)

                if not chosen_executor.can_offload_task(task) and (reward > Config.NEGATIVE_REWARD):
                    reward = Config.NEGATIVE_REWARD
                    timeout_time = current_time + 1
                    self.schedule_retransmission(task, timeout_time)
                elif reward < Config.NEGATIVE_REWARD:
                    timeout_time = current_time + 1
                    self.schedule_retransmission(task, timeout_time)
                else:
                    chosen_executor.assign_task(task, current_time)

                next_state = chosen_zone_manager.env._get_state(task)
                # <<< CHANGED: ذخیره کردن عمل پیوسته در بافر تجربه عامل DDPG
                chosen_zone_manager.agent.store_experience(state, chosen_continuous_action, reward, next_state,
                                                           done=False)
                chosen_zone_manager.agent.train()
            else:
                chosen_executor.assign_task(task, current_time)
        else:
            if task.creator.can_offload_task(task):
                task.creator.assign_task(task, current_time)
                self.metrics.inc_local_execution()
            else:
                self.offload_to_cloud(task, current_time)

    def start_simulation(self):
        self.init_simulation()
        while (current_time := self.clock.get_current_time()) < Config.SimulatorConfig.SIMULATION_DURATION:
            print(red_bg(f"current_time:{current_time}"))

            nodes_tasks = self.load_tasks(current_time)
            user_possible_zones = self.assign_mobile_nodes_to_zones(self.user_nodes, layer=Layer.USER)
            mobile_possible_zones = self.assign_mobile_nodes_to_zones(self.mobile_fog_nodes, layer=Layer.FOG)
            merged_possible_zones: Dict[str, List[ZoneManagerABC]] = {**user_possible_zones, **mobile_possible_zones}

            for creator_id, tasks in nodes_tasks.items():
                zone_managers = merged_possible_zones.get(creator_id, [])
                self.retransmission(zone_managers, current_time)

                for task in tasks:
                    self.metrics.inc_total_tasks()
                    zone_manager_offload_task = self.find_zone_manager_offload_task(zone_managers, task, current_time)
                    self.choose_executor_and_assign(zone_manager_offload_task, task, current_time)

            self.update_graph()
            self.execute_tasks_for_one_step()
            self.metrics.flush()
            # self.metrics.log_metrics()
            self.metrics.add_data(current_time)

        self.drop_not_completed_tasks()
        self.metrics.plot_transmission()
        self.save_missed_deadlines_to_excel(f"missed_deadlines_report_{Config.ZoneManagerConfig.DEFAULT_ALGORITHM}_{Config.FinalDeciderMethod.DEFAULT_METHOD}_{Config.Scenario.DEFAULT_SCENARIO}.xlsx")
        self.save_success_deadlines_to_excel(f"success_deadlines_report_{Config.ZoneManagerConfig.DEFAULT_ALGORITHM}_{Config.FinalDeciderMethod.DEFAULT_METHOD}_{Config.Scenario.DEFAULT_SCENARIO}.xlsx")
        self.metrics.save_to_excel(f"final_metrics_summary_{Config.ZoneManagerConfig.DEFAULT_ALGORITHM}_{Config.FinalDeciderMethod.DEFAULT_METHOD}_{Config.Scenario.DEFAULT_SCENARIO}.xlsx")
    def load_tasks(self, current_time: float) -> Dict[str, List[Task]]:
        tasks: Dict[str, List[Task]] = defaultdict(list)
        for creator_id, creator_tasks in self.loader.load_nodes_tasks(current_time).items():
            creator = None
            if creator_id in self.user_nodes:
                creator = self.user_nodes[creator_id]
            elif creator_id in self.mobile_fog_nodes:
                creator = self.mobile_fog_nodes[creator_id]

            if creator is None:
                print(f"Creator with id {creator_id} not found.")
            else:
                for task in creator_tasks:
                    task.creator = creator
                    tasks[creator_id].append(task)
        return tasks

    def execute_tasks_for_one_step(self):
        executed_tasks: List[Task] = []
        merged_nodes: Dict[str, NodeABC] = {
            **self.mobile_fog_nodes,
            **self.user_nodes,
            **self.fixed_fog_nodes,
            self.cloud_node.id: self.cloud_node,
        }
        for node_id, node in merged_nodes.items():
            tasks = node.execute_tasks(self.clock.get_current_time(), self.fixed_fog_nodes, self.metrics)
            executed_tasks.extend(tasks)
            for task in tasks:
                zone_manager = self.task_zone_managers.get(task.id)
                if zone_manager:
                    zone_manager.update(current_task=task)
                    all_fog_nodes = {**zone_manager.fixed_fog_nodes, **zone_manager.mobile_fog_nodes}
                    loads = [len(node.tasks) for node in all_fog_nodes.values() if node.can_offload_task(task)]
                    if loads:
                        min_load = min(loads)
                        max_load = max(loads)
                        self.metrics.inc_task_load_diff(task.id, min_load, max_load)

                if isinstance(task.executor, (FixedFogNode, MobileFogNode)):
                    self.metrics.inc_fog_execution()
                elif task.creator.id == task.executor.id:
                    self.metrics.inc_local_execution()
                elif isinstance(task.executor, CloudNode):
                    self.metrics.inc_cloud_tasks()

                if task.is_deadline_missed:
                    # print(blue_bg(
                    #     f"{task.id}: release_time:{task.release_time}, deadline:{task.deadline}, exec_time:{task.exec_time}, finish_time:{task.finish_time}, {task.executor.id}, {task.dataSize}, diff:{task.finish_time - task.deadline}"))
                    missed_info = {
                        'task_id': task.id,
                        'release_time': task.release_time,
                        'deadline': task.deadline,
                        'exec_time': task.exec_time,
                        'finish_time': task.finish_time,
                        'executor_id': task.executor.id,
                        'data_size': task.dataSize,
                        'deadline_diff': task.finish_time - task.deadline
                    }
                    self.missed_deadline_data.append(missed_info)
                    self.metrics.inc_deadline_miss()
                else:
                    success_task_info = {
                        'task_id': task.id,
                        'release_time': task.release_time,
                        'deadline': task.deadline,
                        'exec_time': task.exec_time,
                        'finish_time': task.finish_time,
                        'executor_id': task.executor.id,
                        'data_size': task.dataSize,
                        'deadline_diff': task.finish_time - task.deadline
                    }
                    self.success_deadline_data.append(success_task_info)
                    self.metrics.inc_completed_task()

    def update_graph(self):
        self.clock.tick()
        self.update_user_nodes_coordinate()
        self.update_mobile_fog_nodes_coordinate()

    def offload_to_cloud(self, task: Task, current_time: float):
        if self.cloud_node.can_offload_task(task):
            self.cloud_node.assign_task(task, current_time)
            self.metrics.inc_cloud_tasks()
        else:
            self.metrics.inc_no_resource_found()

    def assign_mobile_nodes_to_zones(
            self,
            mobile_nodes: dict[str, MobileNodeABC],
            layer: Layer
    ) -> Dict[str, List[ZoneManagerABC]]:
        nodes_possible_zones: Dict[str, List[ZoneManagerABC]] = defaultdict(list)
        for z_id, zone_manager in self.zone_managers.items():
            nodes: List[MobileNodeABC] = []
            for n_id, mobile_node in mobile_nodes.items():
                if zone_manager.zone.is_in_coverage(mobile_node.x, mobile_node.y):
                    nodes.append(mobile_node)
                    nodes_possible_zones[n_id].append(zone_manager)
            if layer == Layer.FOG:
                zone_manager.set_mobile_fog_nodes(nodes)
        return nodes_possible_zones

    def update_mobile_fog_nodes_coordinate(self) -> None:
        new_nodes_data = self.loader.load_mobile_fog_nodes(self.clock.get_current_time())
        self.mobile_fog_nodes = self.update_nodes_coordinate(self.mobile_fog_nodes, new_nodes_data)

    def update_user_nodes_coordinate(self) -> None:
        new_nodes_data = self.loader.load_user_nodes(self.clock.get_current_time())
        self.user_nodes = self.update_nodes_coordinate(self.user_nodes, new_nodes_data)

    @staticmethod
    def update_nodes_coordinate(old_nodes: dict[str, MobileNodeABC], new_nodes: dict[str, MobileNodeABC]):
        data: Dict[str, MobileNodeABC] = {}
        for n_id, new_node in new_nodes.items():
            if n_id not in old_nodes:
                node = new_node
            else:
                node = old_nodes[n_id]
                node.x = new_node.x
                node.y = new_node.y
                node.angle = new_node.angle
                node.speed = new_node.speed
            data[n_id] = node
        return data

    def drop_not_completed_tasks(self) -> List[Task]:
        left_tasks: list[Task] = []
        merged_nodes: Dict[str, NodeABC] = {
            **self.mobile_fog_nodes,
            **self.user_nodes,
            self.cloud_node.id: self.cloud_node,
        }

        for node_id, node in merged_nodes.items():
            left_tasks.extend(node.tasks)
            for i in range(len(node.tasks)):
                self.metrics.inc_deadline_miss()
        return left_tasks

    def save_missed_deadlines_to_excel(self, filename: str = "missed_deadlines.xlsx"):

        output_dir = "Results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory '{output_dir}' created.")
        full_path = os.path.join(output_dir, filename)

        df = pd.DataFrame(self.missed_deadline_data)
        # index=False از نوشتن ایندکس ردیف‌ها در فایل جلوگیری می‌کند
        try:
            df.to_excel(full_path, index=False)
            print(green_bg(f"Successfully saved missed deadline data to {filename}"))
        except Exception as e:
            print(red_bg(f"Error saving to Excel file: {e}"))

    def save_success_deadlines_to_excel(self, filename: str = "success_deadlines.xlsx"):

        output_dir = "Results_Success"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory '{output_dir}' created.")
        full_path = os.path.join(output_dir, filename)

        df = pd.DataFrame(self.success_deadline_data)

        try:
            df.to_excel(full_path, index=False)
            print(green_bg(f"Successfully saved success deadline data to {filename}"))
        except Exception as e:
            print(red_bg(f"Error saving to Excel file: {e}"))