import csv
import math
import os
import random
import xml.etree.ElementTree as Et
from collections import defaultdict
from dataclasses import dataclass
from xml.dom import minidom

import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET


from config import Config as Cnf

def load_accidents_from_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        accidents_data = []
        for acc in root.findall('accident'):
            accidents_data.append({
                'time': float(acc.get('time')),
                'duration': float(acc.get('duration')),
                'x': float(acc.get('x')),
                'y': float(acc.get('y')),
                'radius': float(acc.get('radius')),
            })
        return accidents_data
    except FileNotFoundError:
        print(f"Error: File not found at path '{file_path}'.")
        return []
    except ET.ParseError:
        print(f"Error: File '{file_path}' is not a valid XML file.")
        return []

class Config:
    CHUNK_SIZE = 1200  # Chunk size in seconds

    class TaskConfig:
        MIN_EXEC_TIME: float = 12.5  # Slightly increased execution times
        MAX_EXEC_TIME: float = 25.0  # Tasks take longer to complete
        MIN_POWER_CONSUMPTION: float = 1.0  # Higher power consumption than before
        MAX_POWER_CONSUMPTION: float = 3.5
        DEADLINE_MIN_FREE_TIME: float = 3.0  # Less deadline flexibility # note : next time make it a little bit more
        DEADLINE_MAX_FREE_TIME: float = 15.0

    class VehicleConfig:
        TASK_GENERATION_RATE: float = 0.35  # More frequent task generation
        FUCKED_UP_TASK_GENERATION_RATE: float = 0.55
        TRAFFIC_MIN_SPEED_THRESHOLD: float = 10  # Lowered speed, causing occasional congestion
        LANE_TRAFFIC_THRESHOLD: int = 15  # More vehicles per lane (moderate traffic)
        MAX_COMPUTATION_POWER: float = 6  # note : change it in feature change !
        MIN_COMPUTATION_POWER: float = 2
        COMPUTATION_POWER_ROUND_DIGIT: int = 2
        LOW_TRANSMISSION_POWER = 20  # todo : it's homogeneous right now and it's good to make it heterogeneous
        # MEDIUM_TRANSMISSION_POWER = 10
        HIGH_TRANSMISSION_POWER = 5
        # TRANSMISSION_LIST = [LOW_TRANSMISSION_POWER, MEDIUM_TRANSMISSION_POWER, HIGH_TRANSMISSION_POWER]

    class MobileFogConfig:
        MAX_COMPUTATION_POWER: float = 12.0  # Slightly reduced power in fog nodes
        MIN_COMPUTATION_POWER: float = 7.0
        COMPUTATION_POWER_ROUND_DIGIT: int = 2


@dataclass
class Vehicle:
    """Represents a mobile fog node in the network with a unique identifier and spatial coordinates."""

    id: str
    x: float
    y: float
    angle: float
    speed: float
    power: float
    type: str
    lane: str
    bandwidth: float


@dataclass
class Task:
    id: str
    deadline: float
    exec_time: float  # The amount of time that this task required to execute.
    power: float  # The amount of power unit that this tasks consumes while executing.
    creator: str  # Thd id of the node who created the task.
    data_size: float


def is_in_accident_zone(ACCIDENTS_DATA, current_time, vehicle):
    for accident in ACCIDENTS_DATA:
        start_time = accident['time']
        end_time = start_time + accident['duration']

        if start_time <= current_time <= end_time:
            distance = math.sqrt(
                (vehicle.x - accident['x']) ** 2 +
                (vehicle.y - accident['y']) ** 2
            )

            if distance <= accident['radius']:
                return True
    return False


class Generator:
    def __init__(self):
        self.scenario = Cnf.Scenario.DEFAULT_SCENARIO
        self.current_chunk = 0
        self.current_vehicles = []
        self.current_tasks = []
        self.tasks_count_per_step = defaultdict(int)
        self.average_speed_per_step = defaultdict(float)
        self.total_task_power_per_step = defaultdict(float)
        self.average_data_size_per_step = defaultdict(float)

        if Cnf.Scenario.DEFAULT_SCENARIO == Cnf.Scenario.RAIN_AND_ACCIDENT or Cnf.Scenario.DEFAULT_SCENARIO == Cnf.Scenario.SNOW_AND_ACCIDENT:
            self.accidents_data = load_accidents_from_xml(f"F:\\BaseVersion\\accidents.xml")
        else:
            self.accidents_data = None

        # Create output directories
        os.makedirs("./data/vehicles", exist_ok=True)
        os.makedirs("./data/tasks", exist_ok=True)

    @staticmethod
    def get_chunk_number(step: int) -> int:
        return step // Config.CHUNK_SIZE

    def save_current_chunk(self, step: int):
        chunk_num = self.get_chunk_number(step)
        if chunk_num > self.current_chunk and (self.current_vehicles or self.current_tasks):
            self._save_vehicles_chunk()
            self._save_tasks_chunk()
            self.current_vehicles = []
            self.current_tasks = []
            self.current_chunk = chunk_num

    def _save_vehicles_chunk(self):
        root = Et.Element('fcd-export')
        root.set("version", "1.0")

        for time_data in self.current_vehicles:
            time_elem = Et.SubElement(root, 'timestep')
            time_elem.set('time', f"{time_data['step']}")
            for vehicle in time_data['vehicles']:
                v_elem = Et.SubElement(time_elem, 'vehicle')
                v_elem.set('id', vehicle.id)
                v_elem.set('x', f"{vehicle.x:.2f}")
                v_elem.set('y', f"{vehicle.y:.2f}")
                v_elem.set('angle', f"{vehicle.angle:.2f}")
                v_elem.set('speed', f"{vehicle.speed:.2f}")
                v_elem.set('lane', vehicle.lane)
                v_elem.set('type', vehicle.type)
                v_elem.set('power', f"{vehicle.power:.2f}")
                v_elem.set('bandwidth', f"{vehicle.bandwidth}")

        xml_str = minidom.parseString(Et.tostring(root)).toprettyxml(indent="    ")
        with open(f"./data/vehicles/chunk_{self.current_chunk}.xml", 'w', encoding='utf-8') as f:
            f.write(xml_str)

    def _save_tasks_chunk(self):
        root = Et.Element('fcd-export')
        root.set("version", "1.0")

        for time_data in self.current_tasks:
            time_elem = Et.SubElement(root, 'timestep')
            time_elem.set('time', f"{time_data['step']}")
            for task in time_data['tasks']:
                t_elem = Et.SubElement(time_elem, 'task')
                t_elem.set('id', task.id)
                t_elem.set('deadline', f"{task.deadline:.2f}")
                t_elem.set('exec_time', f"{task.exec_time:.2f}")
                t_elem.set('power', f"{task.power:.2f}")
                t_elem.set('creator', task.creator)
                t_elem.set('data_size', f"{task.data_size:.2f}")

        xml_str = minidom.parseString(Et.tostring(root)).toprettyxml(indent="    ")
        with open(f"./data/tasks/chunk_{self.current_chunk}.xml", 'w', encoding='utf-8') as f:
            f.write(xml_str)

    def calculate_metrics(self, step: float, vehicles: list[Vehicle], tasks: list[Task]):
        """Calculate metrics for the current timestep."""
        # Count tasks for this step
        self.tasks_count_per_step[step] = len(tasks)

        # Calculate average speed of user nodes (PKW_special type)
        if vehicles:
            avg_speed = sum(v.speed for v in vehicles) / len(vehicles)
            self.average_speed_per_step[step] = round(avg_speed, 2)
        else:
            self.average_speed_per_step[step] = 0.0

        if tasks:
            avg_data_size = sum(t.data_size for t in tasks) / len(tasks)
            self.average_data_size_per_step[step] = round(avg_data_size, 2)
        else:
            self.average_data_size_per_step[step] = 0.0

            # Calculate total power of tasks for this step
        self.total_task_power_per_step[step] = round(sum(task.power for task in tasks), 2)

    @staticmethod
    def generate_one_step_task(step, vehicle, lane_counter, rate_multiplier: float, dataSize_multiplier: float):
        """Generate tasks for each mobile fog node."""
        exec_time = round(
            random.uniform(
                Config.TaskConfig.MIN_EXEC_TIME,
                Config.TaskConfig.MAX_EXEC_TIME,
            ),
            2
        )
        deadline_free = round(
            random.uniform(
                Config.TaskConfig.DEADLINE_MIN_FREE_TIME,
                Config.TaskConfig.DEADLINE_MAX_FREE_TIME,
            ),
            2
        )
        deadline = round(exec_time + deadline_free) + step
        power = round(
            random.uniform(
                Config.TaskConfig.MIN_POWER_CONSUMPTION,
                Config.TaskConfig.MAX_POWER_CONSUMPTION
            ),
            2
        )
        chance = random.random()
        threshold = Config.VehicleConfig.TASK_GENERATION_RATE
        if (
                lane_counter > Config.VehicleConfig.LANE_TRAFFIC_THRESHOLD or
                vehicle.speed < Config.VehicleConfig.TRAFFIC_MIN_SPEED_THRESHOLD
        ):
            threshold = Config.VehicleConfig.FUCKED_UP_TASK_GENERATION_RATE

        final_threshold = threshold * rate_multiplier

        if chance > final_threshold:
            return None

        normalDataSize = 320 + (exec_time / 25) * 320

        final_DataSize = normalDataSize * dataSize_multiplier

        return Task(
            id=f"{vehicle.id}_{step}",
            deadline=deadline,
            exec_time=exec_time,
            power=power,
            creator=vehicle.id,
            data_size=final_DataSize
        )

    def generate_one_step(self, step, time_data, seen_ids_power):
        """Generate vehicles for each mobile fog node."""
        current_vehicles = []
        current_tasks = []
        lane_counter = defaultdict(int)

        bandwidth_candidates = [Cnf.BandwidthCandidates.B1, Cnf.BandwidthCandidates.B2, Cnf.BandwidthCandidates.B3,
                                Cnf.BandwidthCandidates.B4, Cnf.BandwidthCandidates.B5]
        probabilities = [0.10, 0.40, 0.25, 0.15, 0.10]

        for vehicle in time_data.findall('vehicle'):
            v_id = vehicle.get('id')
            selected_bandwidth = np.random.choice(bandwidth_candidates, p=probabilities)
            data = dict(
                id=v_id,
                x=float(vehicle.get('x')),
                y=float(vehicle.get('y')),
                angle=90 - float(vehicle.get('angle')),
                speed=float(vehicle.get('speed')),
                lane=vehicle.get('lane'),
                type=vehicle.get('type'),
                bandwidth=selected_bandwidth
            )

            if v_id in seen_ids_power:
                power = seen_ids_power[v_id]
            elif vehicle.get('type') == "LKW_special":
                power = round(
                    random.uniform(
                        Config.MobileFogConfig.MIN_COMPUTATION_POWER,
                        Config.MobileFogConfig.MAX_COMPUTATION_POWER
                    ),
                    Config.MobileFogConfig.COMPUTATION_POWER_ROUND_DIGIT
                )
            elif vehicle.get('type') == "PKW_special":
                power = round(
                    random.uniform(
                        Config.VehicleConfig.MIN_COMPUTATION_POWER,
                        Config.VehicleConfig.MAX_COMPUTATION_POWER
                    ),
                    Config.MobileFogConfig.COMPUTATION_POWER_ROUND_DIGIT
                )
            else:
                continue

            seen_ids_power[v_id] = power
            data["power"] = power
            vehicle_obj = Vehicle(**data)
            current_vehicles.append(vehicle_obj)
            lane_counter[vehicle_obj.lane] += 1

            rate_multiplier = 1.0
            dataSize_multiplier = 1.0
            if Cnf.Scenario.START_TIME <= step <= Cnf.Scenario.FINISH_TIME:
                if self.scenario == Cnf.Scenario.HEAVY_RAIN:
                    rate_multiplier = random.uniform(1.15, 1.20)
                    dataSize_multiplier = random.uniform(1.01, 1.02)
                elif self.scenario == Cnf.Scenario.HEAVY_SNOW:
                    rate_multiplier = random.uniform(1.25, 1.35)
                    dataSize_multiplier = random.uniform(0.99, 1.03)
                elif self.scenario == Cnf.Scenario.RAIN_AND_ACCIDENT:
                    if is_in_accident_zone(self.accidents_data, step, vehicle_obj):
                        rate_multiplier = random.uniform(1.40, 1.60)
                        dataSize_multiplier = random.uniform(1.02, 1.04)
                    else:
                        rate_multiplier = random.uniform(1.15, 1.20)
                        dataSize_multiplier = random.uniform(1.01, 1.02)
                elif self.scenario == Cnf.Scenario.SNOW_AND_ACCIDENT:
                    if is_in_accident_zone(self.accidents_data, step, vehicle_obj):
                        rate_multiplier = random.uniform(1.45, 1.65)
                        dataSize_multiplier = random.uniform(1.01, 1.03)
                    else:
                        rate_multiplier = random.uniform(1.25, 1.35)
                        dataSize_multiplier = random.uniform(0.99, 1.03)


            if task := self.generate_one_step_task(step, vehicle_obj, lane_counter[vehicle_obj.lane], rate_multiplier, dataSize_multiplier):
                current_tasks.append(task)

        # Calculate metrics before saving the chunk
        self.calculate_metrics(step, current_vehicles, current_tasks)

        # Add current timestep data to the chunk
        self.current_vehicles.append({"step": step, "vehicles": current_vehicles})
        self.current_tasks.append({"step": step, "tasks": current_tasks})

        self.save_current_chunk(step)

        return seen_ids_power

    def generate_data(self, path: str):
        """Parse the time data from the given content."""
        with open(path, 'rb') as f:
            root = Et.parse(f).getroot()
        seen_ids_power = {}
        for time in root.findall('.//timestep'):
            step = round(float(time.get('time')))
            seen_ids_power = self.generate_one_step(step, time, seen_ids_power)

        # Save the last chunk if there's any data left
        if self.current_vehicles or self.current_tasks:
            self._save_vehicles_chunk()
            self._save_tasks_chunk()

    def save_metrics_to_csv(self, metrics_file: str):
        """Save the collected metrics to a CSV file."""
        all_steps = sorted(set(self.tasks_count_per_step.keys()) |
                           set(self.average_speed_per_step.keys()) |
                           set(self.total_task_power_per_step.keys()) |
                           set(self.average_data_size_per_step.keys()))

        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestep', 'task_count', 'average_speed', 'total_task_power'])
            for step in all_steps:
                writer.writerow([
                    f"{step:.2f}",
                    self.tasks_count_per_step[step],
                    self.average_speed_per_step[step],
                    self.total_task_power_per_step[step],
                    self.average_data_size_per_step[step]
                ])

    def plot_metrics(self, output_file: str):
        """Create a visualization of the metrics."""
        steps = sorted(set(self.tasks_count_per_step.keys()) |
                       set(self.average_speed_per_step.keys()) |
                       set(self.total_task_power_per_step.keys()) |
                       set(self.average_data_size_per_step.keys()))
        task_counts = [self.tasks_count_per_step[step] for step in steps]
        avg_speeds = [self.average_speed_per_step[step] for step in steps]
        total_powers = [self.total_task_power_per_step[step] for step in steps]
        avg_data_size = [self.average_data_size_per_step[step] for step in steps]

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))

        ax1.plot(steps, task_counts, 'b-', label='Tasks per Step')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Number of Tasks')
        ax1.set_title('Tasks Generated per Time Step')
        ax1.grid(True)
        ax1.legend()

        ax2.plot(steps, avg_speeds, 'r-', label='Average Speed')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Speed')
        ax2.set_title('Average Speed of User Nodes per Time Step')
        ax2.grid(True)
        ax2.legend()

        ax3.plot(steps, total_powers, 'g-', label='Total Task Power')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Power Units')
        ax3.set_title('Total Power of Tasks per Time Step')
        ax3.grid(True)
        ax3.legend()

        ax4.plot(steps, avg_data_size, 'g-', label='Average Data Size')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Data Size')
        ax4.set_title('Average Data Size of Tasks per Time Step')
        ax4.grid(True)
        ax4.legend()

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()


def main(path: str):
    """Main function to run the task generator."""
    generator = Generator()
    generator.generate_data(path)
    generator.save_metrics_to_csv("./data/metrics.csv")
    generator.plot_metrics("./data/metrics_visualization.png")


if __name__ == '__main__':
    main("./simulation.out.xml")
