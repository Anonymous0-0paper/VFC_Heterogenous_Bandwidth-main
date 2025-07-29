import os
import random
from collections import defaultdict
from typing import Dict, List

from config import Config
from controllers.finalChoice import FinalChoice
from controllers.loader import Loader
from controllers.metric import MetricsController, green_bg
from controllers.zone_managers.STGT.stgt_zone_manager import STGTZoneManager
from models.node.base import MobileNodeABC, NodeABC
from models.node.cloud import CloudNode
from models.node.fog import FixedFogNode
from models.node.fog import MobileFogNode
from models.node.user import UserNode
from models.task import Task
from utils.clock import Clock
from utils.enums import Layer
import pandas as pd


def yellow_bg(text):
    return f"\033[43m{text}\033[0m"


def red_bg(text):
    return f"\033[41m{text}\033[0m"


def blue_bg(text):
    return f"\033[44m{text}\033[0m"


class SimulatorSTGT:
    def __init__(self, loader: Loader, clock: Clock, cloud: CloudNode):
        self.metrics: MetricsController = MetricsController()
        self.loader: Loader = loader
        self.cloud_node: CloudNode = cloud
        self.zone_managers: Dict[str, STGTZoneManager] = {}
        self.fixed_fog_nodes: Dict[str, FixedFogNode] = {}
        self.mobile_fog_nodes: Dict[str, MobileFogNode] = {}
        self.user_nodes: Dict[str, UserNode] = {}
        self.clock: Clock = clock
        self.task_zone_managers: Dict[str, STGTZoneManager] = {}
        self.retransmission_tasks: Dict[float, List[Task]] = {}
        self.missed_deadline_data: List[Dict] = []
        
        # STGT-specific parameters
        self.feature_dim = 6
        self.hidden_dim = 128
        self.num_layers = 2
        self.k_neighbors = 10
        self.alpha = 0.95  # CVaR confidence level
        self.lambda1 = 1.0  # Throughput multiplier
        self.lambda2 = 0.1  # Fairness multiplier
        self.l_max = 100.0  # Maximum latency constraint

    def init_simulation(self):
        self.clock.set_current_time(0)
        self.zone_managers = self.loader.load_zones()
        self.fixed_fog_nodes = self.loader.load_fixed_zones()
        self.assign_fixed_nodes()
        self.update_mobile_fog_nodes_coordinate()
        self.update_user_nodes_coordinate()
        
        # Initialize STGT zone managers
        for zone_id, zone_manager in self.zone_managers.items():
            if not isinstance(zone_manager, STGTZoneManager):
                # Replace with STGT zone manager
                stgt_manager = STGTZoneManager(
                    zone=zone_manager.zone,
                    feature_dim=self.feature_dim,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    k_neighbors=self.k_neighbors,
                    alpha=self.alpha,
                    lambda1=self.lambda1,
                    lambda2=self.lambda2,
                    l_max=self.l_max
                )
                # Copy existing nodes
                stgt_manager.fixed_fog_nodes = zone_manager.fixed_fog_nodes
                stgt_manager.mobile_fog_nodes = zone_manager.mobile_fog_nodes
                self.zone_managers[zone_id] = stgt_manager

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
                # For STGT, we use the propose_candidate method
                if hasattr(zone_manager, "propose_candidate"):
                    proposed_zone_manager, proposed_executor = zone_manager.propose_candidate(task, current_time)
                else:
                    proposed_zone_manager = zone_manager
                    proposed_executor = zone_manager.offload_task(task, current_time)

                if proposed_executor not in zone_manager_offload_task:
                    zone_manager_offload_task.append(proposed_executor)

        return zone_manager_offload_task

    def choose_executor_and_assign(self, zone_manager_offload_task, task, current_time):
        if not zone_manager_offload_task:
            return False

        # Use STGT-based selection
        if Config.FinalDeciderMethod.DEFAULT_METHOD == Config.FinalDeciderMethod.FIRST_CHOICE:
            executor = zone_manager_offload_task[0]
        elif Config.FinalDeciderMethod.DEFAULT_METHOD == Config.FinalDeciderMethod.RANDOM_CHOICE:
            executor = random.choice(zone_manager_offload_task)
        elif Config.FinalDeciderMethod.DEFAULT_METHOD == Config.FinalDeciderMethod.MIN_DISTANCE:
            executor = min(zone_manager_offload_task, key=lambda x: x.get_creator_and_executor_distance(task))
        else:
            executor = zone_manager_offload_task[0]

        if executor.assign_task(task, current_time):
            self.metrics.inc_completed_task()
            self.metrics.inc_node_tasks(executor.id)
            return True
        else:
            self.metrics.inc_no_resource_found()
            return False

    def start_simulation(self):
        self.init_simulation()
        current_time = 0

        while current_time < Config.SimulatorConfig.SIMULATION_DURATION:
            # Load tasks for current time
            tasks_by_zone = self.load_tasks(current_time)

            # Execute tasks for one step
            self.execute_tasks_for_one_step()

            # Update mobile nodes coordinates
            self.update_mobile_fog_nodes_coordinate()
            self.update_user_nodes_coordinate()

            # Update zone managers with STGT information
            self.update_zone_managers(current_time, tasks_by_zone)

            # Handle retransmissions
            self.retransmission(list(self.zone_managers.values()), current_time)

            # Update metrics
            self.metrics.flush()

            current_time += 1

        # Finalize simulation
        self.finalize_simulation()

    def load_tasks(self, current_time: float) -> Dict[str, List[Task]]:
        tasks_by_zone = defaultdict(list)
        
        # Load tasks from data files
        chunk = current_time // Config.CHUNK_SIZE
        task_file = f"./data/tasks/chunk_{chunk}.xml"

        if os.path.exists(task_file):
            tasks = self.loader.load_tasks_from_file(task_file, current_time)
            # Assign creator objects
            for task in tasks:
                creator = self.user_nodes.get(task.creator_id) or self.mobile_fog_nodes.get(task.creator_id)
                task.creator = creator
            for task in tasks:
                if task.creator is None:
                    continue  # or handle error/log
                for zone_id, zone_manager in self.zone_managers.items():
                    if zone_manager.zone.is_in_coverage(task.creator.x, task.creator.y):
                        tasks_by_zone[zone_id].append(task)
                        break
        return tasks_by_zone

    def execute_tasks_for_one_step(self):
        # Execute tasks on all nodes
        for node_id, node in self.fixed_fog_nodes.items():
            if hasattr(node, 'execute_tasks'):
                node.execute_tasks(self.clock.get_current_time(), self.fixed_fog_nodes, self.metrics)

        for node_id, node in self.mobile_fog_nodes.items():
            if hasattr(node, 'execute_tasks'):
                node.execute_tasks(self.clock.get_current_time(), self.fixed_fog_nodes, self.metrics)

        for node_id, node in self.user_nodes.items():
            if hasattr(node, 'execute_tasks'):
                node.execute_tasks(self.clock.get_current_time(), self.fixed_fog_nodes, self.metrics)

        # Execute cloud tasks
        if hasattr(self.cloud_node, 'execute_tasks'):
            self.cloud_node.execute_tasks(self.clock.get_current_time(), self.fixed_fog_nodes, self.metrics)

    def update_zone_managers(self, current_time: float, tasks_by_zone: Dict[str, List[Task]]):
        """Update zone managers with STGT-specific information"""
        for zone_id, zone_manager in self.zone_managers.items():
            if isinstance(zone_manager, STGTZoneManager):
                # Update with current tasks and time
                zone_manager.update(
                    current_time=current_time,
                    all_zone_managers=list(self.zone_managers.values()),
                    tasks=tasks_by_zone.get(zone_id, []),
                    current_task=None
                )

    def update_graph(self):
        # Update graph structure for STGT
        pass

    def offload_to_cloud(self, task: Task, current_time: float):
        if self.cloud_node.assign_task(task, current_time):
            self.metrics.inc_cloud_tasks()
            return True
        return False

    def assign_mobile_nodes_to_zones(
            self,
            mobile_nodes: dict[str, MobileNodeABC],
            layer: Layer
    ) -> Dict[str, List[STGTZoneManager]]:
        zone_assignments = defaultdict(list)
        
        for node_id, node in mobile_nodes.items():
            for zone_id, zone_manager in self.zone_managers.items():
                if zone_manager.zone.is_in_coverage(node.x, node.y):
                    zone_assignments[zone_id].append(zone_manager)
                    break
        
        return zone_assignments

    def update_mobile_fog_nodes_coordinate(self) -> None:
        # Load updated mobile fog node coordinates
        chunk = self.clock.get_current_time() // Config.CHUNK_SIZE
        vehicle_file = f"./data/vehicles/chunk_{chunk}.xml"
        
        if os.path.exists(vehicle_file):
            new_mobile_fog_nodes = self.loader.load_mobile_fog_nodes_from_file(vehicle_file)
            self.update_nodes_coordinate(self.mobile_fog_nodes, new_mobile_fog_nodes)
            self.mobile_fog_nodes = new_mobile_fog_nodes

    def update_user_nodes_coordinate(self) -> None:
        # Load updated user node coordinates
        chunk = self.clock.get_current_time() // Config.CHUNK_SIZE
        vehicle_file = f"./data/vehicles/chunk_{chunk}.xml"
        
        if os.path.exists(vehicle_file):
            new_user_nodes = self.loader.load_user_nodes_from_file(vehicle_file)
            self.update_nodes_coordinate(self.user_nodes, new_user_nodes)
            self.user_nodes = new_user_nodes

    @staticmethod
    def update_nodes_coordinate(old_nodes: dict[str, MobileNodeABC], new_nodes: dict[str, MobileNodeABC]):
        for node_id, new_node in new_nodes.items():
            if node_id in old_nodes:
                old_node = old_nodes[node_id]
                old_node.x = new_node.x
                old_node.y = new_node.y
                if hasattr(old_node, 'speed'):
                    old_node.speed = new_node.speed
                if hasattr(old_node, 'angle'):
                    old_node.angle = new_node.angle

    def drop_not_completed_tasks(self) -> List[Task]:
        dropped_tasks = []
        
        # Collect tasks from all nodes
        for node_id, node in self.fixed_fog_nodes.items():
            if hasattr(node, 'task_queue'):
                dropped_tasks.extend(node.task_queue)
                node.task_queue.clear()

        for node_id, node in self.mobile_fog_nodes.items():
            if hasattr(node, 'task_queue'):
                dropped_tasks.extend(node.task_queue)
                node.task_queue.clear()

        for node_id, node in self.user_nodes.items():
            if hasattr(node, 'task_queue'):
                dropped_tasks.extend(node.task_queue)
                node.task_queue.clear()

        # Update metrics
        for task in dropped_tasks:
            self.metrics.inc_deadline_miss()
            self.missed_deadline_data.append({
                'task_id': task.creator_id,
                'deadline': task.deadline,
                'release_time': task.release_time,
                'exec_time': task.exec_time,
                'creator_id': task.creator_id,
                'executor_id': task.executor.id if task.executor else None,
                'finish_time': task.finish_time,
                'is_completed': task.is_completed,
                'has_migrated': task.has_migrated if task.executor else False
            })

        return dropped_tasks

    def save_missed_deadlines_to_excel(self, filename: str = "missed_deadlines.xlsx"):
        if self.missed_deadline_data:
            df = pd.DataFrame(self.missed_deadline_data)
            df.to_excel(filename, index=False)
            print(f"Missed deadlines data saved to {filename}")

    def finalize_simulation(self):
        """Finalize simulation and save STGT models"""
        # Drop incomplete tasks
        dropped_tasks = self.drop_not_completed_tasks()
        
        # Save STGT models
        for zone_id, zone_manager in self.zone_managers.items():
            if isinstance(zone_manager, STGTZoneManager):
                model_path = f"./checkpoints/stgt_model_{zone_id}.pth"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                zone_manager.save_model(model_path)
                print(f"STGT model for zone {zone_id} saved to {model_path}")
        
        # Save missed deadlines
        self.save_missed_deadlines_to_excel("missed_deadlines_stgt.xlsx")
        
        # Print final metrics
        print("\n" + "="*50)
        print("STGT SIMULATION RESULTS")
        print("="*50)
        print(f"Total Tasks: {self.metrics.total_tasks}")
        print(f"Completed Tasks: {self.metrics.completed_tasks}")
        print(f"Deadline Misses: {self.metrics.deadline_misses}")
        print(f"Migrations: {self.metrics.migrations_count}")
        print(f"Cloud Tasks: {self.metrics.cloud_tasks}")
        print(f"Local Executions: {self.metrics.local_execution}")
        print(f"Fog Executions: {self.metrics.fog_execution}")
        print(f"Migrate and Miss: {self.metrics.migrate_and_miss}")
        
        if self.metrics.total_tasks > 0:
            completion_rate = (self.metrics.completed_tasks / self.metrics.total_tasks) * 100
            miss_rate = (self.metrics.deadline_misses / self.metrics.total_tasks) * 100
            print(f"Completion Rate: {completion_rate:.2f}%")
            print(f"Miss Rate: {miss_rate:.2f}%")
        
        print("="*50)