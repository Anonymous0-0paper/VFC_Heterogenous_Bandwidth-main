from __future__ import annotations

import abc
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque
import numpy as np

from NoiseConfigs.utilsFunctions import UtilsFunc
from config import Config
from controllers.metric import green_bg
from models.base import ModelBaseABC
from utils.enums import Layer
from utils.distance import get_distance
from task_and_user_generator import Config as Cnf

def blue_bg(text):
    return f"\033[44m{text}\033[0m"

def calcPathLoss(task, node):
    """
    node : nearest_node or executor node
    """
    return UtilsFunc().path_loss_km_ghz(
        d_km=UtilsFunc().distance(task.creator.x, task.creator.y,
                                  node.x, node.y) / 1000,
        f_ghz=UtilsFunc().FREQUENCY_GH,
        n=2
    )

def findExecTimeInEachKindOfNode(task, executor=None):
    from models.node.user import UserNode
    from models.node.cloud import CloudNode
    from models.node.fog import FixedFogNode
    from models.node.fog import MobileFogNode

    taskExecutor = task.executor
    if executor:
        taskExecutor = executor
    if isinstance(taskExecutor, UserNode):
        # print("UserNode()")
        return task.real_exec_time(executor=taskExecutor) * 1.5
    elif isinstance(taskExecutor, CloudNode):
        # print("CloudNode()")
        return task.real_exec_time(executor=taskExecutor) / 5
    elif isinstance(taskExecutor, FixedFogNode):
        # print("FixedFogNode()")
        return task.real_exec_time(executor=taskExecutor) / 3
    elif isinstance(taskExecutor, MobileFogNode):
        # print("MobileFogNode()")
        return task.real_exec_time(executor=taskExecutor) / 2
    else:
        print("errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrorr")
        return -1

    # return real_exec_time_base

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_closest_fn(x, y, fn_nodes, taskPower):
    closest_fn = None
    min_distance = float('inf')
    # print(fn_nodes)

    for fn in fn_nodes.values():
        # print(fn)
        distance = calculate_distance(x, y, fn.x, fn.y)
        if distance < min_distance:
            min_distance = distance
            closest_fn = fn

    return closest_fn

def findDataRate(task, executor, closest_fn) -> float:
    from models.node.cloud import CloudNode
    from models.node.fog import FixedFogNode
    from models.node.fog import MobileFogNode
    eta = 0.0
    pathLoss = 0.0
    if isinstance(executor, MobileFogNode) or isinstance(executor, FixedFogNode):
        # print(blue_bg(f"{executor.id}:{len(executor.tasks)}"))
        # eta = task.power / executor.power
        eta = 1/(len(executor.tasks)+1)
        pathLoss = calcPathLoss(task, executor)
    elif isinstance(executor, CloudNode):
        # eta = task.power / closest_fn.power
        eta = 1/(len(closest_fn.tasks)+1)
        pathLoss = calcPathLoss(task, closest_fn)

    # print(green_bg(
    #     f"task.id: {task.id}, task.executor.id: {executor.id}, task.power: {task.power}, task.executor.power: {executor.power}, eta : {eta}"))

    path_loss_linear = 10 ** (pathLoss / 10)
    received_power = Cnf().VehicleConfig().HIGH_TRANSMISSION_POWER * path_loss_linear
    snr = received_power / 10**-13
    dataRate = eta * task.creator.bandwidth * np.log2(1 + snr)
    # print(green_bg(f"received_power:{received_power}, snr:{snr}, eta:{eta}"))
    # print(blue_bg(f"{task.creator.id} ==> bandwidth: {task.creator.bandwidth}, dataRate: {dataRate}"))
    return dataRate


@dataclass
class NodeABC(ModelBaseABC, abc.ABC):
    """
    Represents any computational resource in the system. Nodes can belong to different layers,
    such as User, Fog, or Cloud.
    """

    x: float = 0
    y: float = 0
    power: float = 0  # The computational power available at this node.

    radius: float = 0  # The radius that this node can cover.
    bandwidth: float = 0

    remaining_power: float = 0  # The amount of computational resourced left after executing current tasks.
    tasks: Deque = field(default_factory=deque)  # The list of tasks that are currently offloaded in this node.
    finished_tasks: Deque = field(default_factory=deque)  # The list of tasks that are finished executing in this node.

    def can_offload_task(self, task) -> bool:
        """Checks whether the task can be offloaded in this node."""
        # print("----------------------------------test----------------------------------")
        if len(self.tasks) >= self.max_tasks_queue_len:
            # print(blue_bg(f"max_tasks_queue_len"))
            return False
        # if 1.0 - self.remaining_power / self.power > self.used_power_limit:
        #     # print(blue_bg(f"test????????????"))
        #     return False
        task_power = task.power
        if self.id == task.creator.id and self.layer == Layer.USER:
            task_power *= Config.UserNodeConfig.LOCAL_OFFLOAD_POWER_OVERHEAD
        if task_power > self.remaining_power:
            # print(blue_bg(f"remaining_power"))
            return False
        if get_distance(self.x, self.y, task.creator.x, task.creator.y) > self.radius:
            # print(blue_bg(f"distance"))
            return False
        return True

    def assign_task(self, task, current_time: float) -> None:
        """Offload a task in the current node."""
        # print(f"task:{task}")
        # note: i think below code has been checked
        # if not self.can_offload_task(task):
        #     raise Exception

        self.tasks.append(task)
        task.executor = self

        task_power = task.power
        if self.id == task.creator.id and self.layer == Layer.USER:
            task_power *= Config.UserNodeConfig.LOCAL_OFFLOAD_POWER_OVERHEAD
        self.remaining_power -= task_power
        task.start_time = current_time
        # task.start_time = task.release_time

    def execute_tasks(self, current_time: float, fixed_fog_nodes, metrics) -> list:
        """Execute current tasks and return all completed tasks."""
        remaining_tasks = deque()
        finished_tasks = deque()

        # print(f"{self.id}: {self.tasks}")
        while self.tasks:
            task = self.tasks.popleft()
            # print(f"task: {task}")
            # print(f"[DEBUG] Processing task {task.id}: start_time={task.start_time}, current_time={current_time}, base_exec_time={task.real_exec_time}")

            task.real_exec_time_base = findExecTimeInEachKindOfNode(task)
            # if task.id == "PKW10_2":
            #     print(f"[DEBUG]{task.id}: real_exec_time_base={task.real_exec_time_base}")

            if current_time - task.release_time >= task.real_exec_time_base:
                # if task.id == "PKW10_2":
                #     print(f"[DEBUG]{task.id} +++: real_exec_time_base={task.real_exec_time_base}")
                finished_tasks.append(task)

                task_power = task.power
                if self.id == task.creator.id and self.layer == Layer.USER:
                    task_power *= Config.UserNodeConfig.LOCAL_OFFLOAD_POWER_OVERHEAD
                self.remaining_power += task_power
            else:
                remaining_tasks.append(task)

        self.tasks = remaining_tasks
        self.finished_tasks = self.finished_tasks + finished_tasks

        final_finished_tasks = []
        remaining_finished_tasks = deque()
        while self.finished_tasks:

            task = self.finished_tasks.popleft()
            # if task.id == "PKW10_2":
            #     print(f"[DEBUG]{task.id} &&&&&&&: real_exec_time_base={task.real_exec_time_base}")
            distance = task.get_creator_and_executor_distance()
            if not task:
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            real_exec_time = task.real_exec_time_base

            transmission_daly = 0.0
            if task.creator.id != task.executor.id:
                # print(green_bg(f"dataRate = {dataRate}"))
                # print(self.id)
                if self.layer == Layer.FOG:
                    dataRate = findDataRate(task, task.executor, 0)
                    transmission_daly = task.task_transmission_multiplier * (task.dataSize / dataRate)
                    real_exec_time += transmission_daly

                    # print(blue_bg(f"executor: {task.executor.id}::: delay: {task.dataSize / dataRate}, dataRate: {dataRate}"))
                elif self.layer == Layer.CLOUD:

                    closest_fn = find_closest_fn(task.creator.x, task.creator.y, fixed_fog_nodes, task.power)
                    dataRate = findDataRate(task, task.executor, closest_fn)

                    if closest_fn.x == 4214.90 and closest_fn.y == 1932.26:
                        transmission_daly = task.task_transmission_multiplier * ((task.dataSize / dataRate) + (
                                task.dataSize / Config.CloudConfig.CLOUD_BANDWIDTH))
                        real_exec_time += transmission_daly
                        # print(blue_bg(f"executor: {task.executor.id}::: delay: {(task.dataSize / dataRate) + (task.dataSize / Config.CloudConfig.CLOUD_BANDWIDTH)}, dataRate: {dataRate}"))
                    else:
                        transmission_daly = task.task_transmission_multiplier * ((task.dataSize / dataRate) + 2 * (
                                task.dataSize / Config.CloudConfig.CLOUD_BANDWIDTH))
                        real_exec_time += transmission_daly
                        # print(blue_bg(f"executor: {task.executor.id}::: firstStepDelay: {(task.dataSize / dataRate)}, delay: {(task.dataSize / dataRate) + 2 * (task.dataSize / Config.CloudConfig.CLOUD_BANDWIDTH)}, dataRate: {dataRate}"))

                        # real_exec_time += Config.TaskConfig.CLOUD_PROCESSING_OVERHEAD
            if transmission_daly != 0.0:
                metrics.add_transmission(transmission_daly)
            # elif task.has_migrated:
            #     real_exec_time += Config.TaskConfig.MIGRATION_OVERHEAD * task.dataSize

            # print(f"[DEBUG] Re-checking task {task.id}: new_exec_time={real_exec_time}, elapsed={current_time - task.start_time}")

            if current_time - task.start_time >= real_exec_time:
                task.finish_time = task.start_time + real_exec_time
                final_finished_tasks.append(task)
            else:
                remaining_finished_tasks.append(task)

        self.finished_tasks = remaining_finished_tasks

        # print(f"finished_tasks: {finished_tasks}")
        return final_finished_tasks

    @property
    @abc.abstractmethod
    def layer(self) -> Layer:
        """The layer to which this node belongs (User, Fog, or Cloud)."""
        raise NotImplemented

    @property
    @abc.abstractmethod
    def max_tasks_queue_len(self) -> int:
        """The max number of tasks that this node can process in parallel."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def used_power_limit(self) -> float:
        """The maximum power that the node can use to execute its tasks."""
        raise NotImplementedError


@dataclass
class MobileNodeABC(NodeABC, abc.ABC):
    """
    Represents any computational resource in the system which can move from one location to another. Mobile nodes may
    belong to different layers, such as User or Fog.
    """

    speed: float = 0
    angle: float = 0
