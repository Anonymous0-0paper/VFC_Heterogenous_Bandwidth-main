import random
import xml.etree.ElementTree as Et
import numpy as np
from typing import List, Set, Tuple
from dataclasses import dataclass
from xml.dom import minidom
import copy # For deep copying the best generator object

class Config:
    class Simulation:
        # Number of times to run the algorithm to find the best result
        RUN_COUNT: int = 100

    class ZoneConfig:
        """Represents the whole configuration for each zone."""
        # The maximum number of zones to be generated
        MAX_ZONES: int = 16

        # Multiple radius options for different density areas
        ALL_COVERAGE_RADIUS: List[float] = [400, 450, 500]
        MAX_ATTEMPTS_PER_ZONE: int = 50
        MAX_TARGET_ADJUSTMENTS: int = 5
        INITIAL_NODES_PER_ZONE: int = 30
        MIN_NODES_PER_ZONE: int = 20
        COVERAGE_DEVIATION_THRESHOLD: float = 0.5
        # Radius selection parameters
        DENSITY_RADIUS: float = 500  # Radius to check node density
        HIGH_DENSITY_THRESHOLD: int = 100  # Nodes within density radius

    class FixedFogConfig:
        """Represents the configuration for our fixed fog nodes."""
        MAX_COMPUTATION_POWER: float = 20.0
        MIN_COMPUTATION_POWER: float = 15.0
        COMPUTATION_POWER_ROUND_DIGIT: int = 2


@dataclass
class Node:
    id: str
    x: float
    y: float


@dataclass
class Zone:
    id: str
    x: float
    y: float
    radius: float

    def covers_node(self, node: Node) -> bool:
        distance = np.sqrt((self.x - node.x) ** 2 + (self.y - node.y) ** 2)
        return distance <= self.radius


@dataclass
class FixedFogNode:
    id: str
    x: float
    y: float
    power: float
    radius: float


class BalancedZoneGenerator:
    def __init__(self):
        self.zones: List[Zone] = []
        self.fixed_fog_nodes: List[FixedFogNode] = []
        self.target_nodes_per_zone = Config.ZoneConfig.INITIAL_NODES_PER_ZONE

    @classmethod
    def generate_zone_id(cls, index: int) -> str:
        return f"Z{index:04d}"

    @classmethod
    def generate_fixed_fog_node_id(cls, index: int) -> str:
        return f"FN{index:04d}"

    @staticmethod
    def calculate_node_density(center: Node, nodes: List[Node]) -> int:
        """Calculate the number of nodes within the density radius of a point."""
        count = 0
        for node in nodes:
            distance = np.sqrt((center.x - node.x) ** 2 + (center.y - node.y) ** 2)
            if distance <= Config.ZoneConfig.DENSITY_RADIUS:
                count += 1
        return count

    def select_radius(self, center: Node, nodes: List[Node]) -> float:
        """Select appropriate radius based on node density."""
        density = self.calculate_node_density(center, nodes)

        if density > Config.ZoneConfig.HIGH_DENSITY_THRESHOLD:
            # Use smaller radius for high density areas
            return min(Config.ZoneConfig.ALL_COVERAGE_RADIUS)
        elif density < Config.ZoneConfig.HIGH_DENSITY_THRESHOLD // 2:
            # Use larger radius for low density areas
            return max(Config.ZoneConfig.ALL_COVERAGE_RADIUS)
        else:
            # Use medium radius for medium density areas
            radii = sorted(Config.ZoneConfig.ALL_COVERAGE_RADIUS)
            return radii[len(radii) // 2]

    @staticmethod
    def get_covered_nodes(zone: Zone, nodes: List[Node]) -> Set[str]:
        """Get the set of node IDs covered by a zone."""
        return {node.id for node in nodes if zone.covers_node(node)}

    def calculate_coverage_score(self, covered_count: int) -> float:
        """Calculate a score for how well the coverage matches the target."""
        if covered_count < Config.ZoneConfig.MIN_NODES_PER_ZONE:
            return 0

        deviation = abs(covered_count - self.target_nodes_per_zone) / self.target_nodes_per_zone
        if deviation > Config.ZoneConfig.COVERAGE_DEVIATION_THRESHOLD:
            return 0
        return 1 - (deviation / Config.ZoneConfig.COVERAGE_DEVIATION_THRESHOLD)

    def find_best_zone_location(
            self,
            uncovered_nodes: List[Node],
    ) -> Tuple[Node, Set[str], float, float]:
        """Find the best location for a new zone based on balanced coverage."""
        if not uncovered_nodes:
            return None, set(), 0, 0

        best_score = -1
        best_center = None
        best_covered_nodes = set()
        best_radius = 0

        # Sample a subset of nodes as potential centers if there are too many
        potential_centers = uncovered_nodes
        if len(potential_centers) > Config.ZoneConfig.MAX_ATTEMPTS_PER_ZONE:
            potential_centers = random.sample(potential_centers, Config.ZoneConfig.MAX_ATTEMPTS_PER_ZONE)

        for center_node in potential_centers:
            # Select radius based on local node density
            radius = self.select_radius(center_node, uncovered_nodes)

            temp_zone = Zone(
                id="",
                x=center_node.x,
                y=center_node.y,
                radius=radius,
            )

            covered = self.get_covered_nodes(temp_zone, uncovered_nodes)
            if not covered:
                continue

            coverage_score = self.calculate_coverage_score(len(covered))

            if coverage_score > best_score:
                best_score = coverage_score
                best_center = center_node
                best_covered_nodes = covered
                best_radius = radius

        return best_center, best_covered_nodes, best_score, best_radius

    def adjust_target_coverage(self, remaining_nodes: int, remaining_attempts: int) -> bool:
        """Adjust target coverage based on remaining nodes and attempts."""
        if remaining_attempts <= 0:
            return False

        new_target = max(
            remaining_nodes // (remaining_attempts + 1),
            Config.ZoneConfig.MIN_NODES_PER_ZONE
        )

        if new_target != self.target_nodes_per_zone:
            self.target_nodes_per_zone = new_target
            return True
        return False

    def generate_zones(self, nodes: List[Node]) -> None:
        """Generate zones with balanced node coverage."""
        uncovered_nodes = nodes.copy()
        counter = 1
        adjustment_attempts = Config.ZoneConfig.MAX_TARGET_ADJUSTMENTS

        # Calculate target based on the maximum number of zones allowed
        if Config.ZoneConfig.MAX_ZONES > 0:
            # Calculate what the ideal number of nodes per zone should be
            ideal_target = len(nodes) // Config.ZoneConfig.MAX_ZONES
            # Ensure the target is not below the configured minimum
            self.target_nodes_per_zone = max(
                ideal_target,
                Config.ZoneConfig.MIN_NODES_PER_ZONE
            )
        else:
            # Fallback to initial config if MAX_ZONES is 0 or less
            self.target_nodes_per_zone = Config.ZoneConfig.INITIAL_NODES_PER_ZONE

        # print(
        #     f"Targeting {self.target_nodes_per_zone} nodes per zone to create a maximum of {Config.ZoneConfig.MAX_ZONES} zones.")

        # Main loop with the new stopping condition
        while uncovered_nodes and \
                adjustment_attempts >= 0 and \
                len(self.zones) < Config.ZoneConfig.MAX_ZONES:

            best_center, best_covered_nodes, score, radius = self.find_best_zone_location(
                uncovered_nodes,
            )

            if best_center is None or score == 0:
                # Try adjusting target coverage
                if self.adjust_target_coverage(
                        len(uncovered_nodes),
                        adjustment_attempts
                ):
                    adjustment_attempts -= 1
                    continue
                else:
                    # If we can't adjust anymore, just use the best we found
                    if best_center is None:
                        # print("Could not find any more suitable locations. Stopping.")
                        break

            # Create and add the zone
            self.zones.append(
                Zone(
                    id=self.generate_zone_id(counter),
                    x=best_center.x,
                    y=best_center.y,
                    radius=radius,
                )
            )

            # Create corresponding fixed fog node
            self.fixed_fog_nodes.append(
                FixedFogNode(
                    id=self.generate_fixed_fog_node_id(counter),
                    x=best_center.x,
                    y=best_center.y,
                    radius=radius,
                    power=round(
                        random.uniform(
                            Config.FixedFogConfig.MIN_COMPUTATION_POWER,
                            Config.FixedFogConfig.MAX_COMPUTATION_POWER
                        ),
                        Config.FixedFogConfig.COMPUTATION_POWER_ROUND_DIGIT
                    ),
                )
            )
            counter += 1

            # Remove covered nodes from uncovered list
            uncovered_nodes = [
                node for node in uncovered_nodes if node.id not in best_covered_nodes
            ]

    def get_coverage_stats(self, nodes: List[Node]) -> Tuple[int, float]:
        """Calculates and returns total covered nodes and the standard deviation."""
        coverage_counts = [sum(1 for node in nodes if zone.covers_node(node)) for zone in self.zones]
        if not coverage_counts:
            return 0, 0.0

        std_dev = np.std(coverage_counts)

        covered_nodes_ids = set()
        for zone in self.zones:
            for node in nodes:
                if zone.covers_node(node):
                    covered_nodes_ids.add(node.id)

        return len(covered_nodes_ids), std_dev

    def print_info(self, nodes: List[Node]) -> None:
        """Print information about zones and coverage distribution"""
        print(f"\nGenerated {len(self.zones)} zones (Maximum allowed: {Config.ZoneConfig.MAX_ZONES}):\n")
        coverage_counts = []

        for zone in self.zones:
            covered = sum(1 for node in nodes if zone.covers_node(node))
            coverage_counts.append(covered)
            print(f"Zone {zone.id}: Center({zone.x:.2f}, {zone.y:.2f}), "
                  f"Radius={zone.radius:.2f}, Covers {covered} nodes")

        if coverage_counts:
            avg_coverage = np.mean(coverage_counts)
            std_coverage = np.std(coverage_counts)
            print(f"\nCoverage Statistics:")
            print(f"Average nodes per zone: {avg_coverage:.2f}")
            print(f"Standard deviation: {std_coverage:.2f}")
            print(f"Coefficient of variation: {(std_coverage / avg_coverage) * 100:.2f}%")


        covered_nodes_ids = set()
        for zone in self.zones:
            for node in nodes:
                if zone.covers_node(node):
                    covered_nodes_ids.add(node.id)

        print(f"\nTotal coverage: {len(covered_nodes_ids)}/{len(nodes)} nodes\n")

        all_node_ids = {node.id for node in nodes}
        uncovered_node_ids = all_node_ids - covered_nodes_ids

        if uncovered_node_ids:
            print(f"Found {len(uncovered_node_ids)} uncovered nodes:")
            print(", ".join(sorted(list(uncovered_node_ids))))
            print("\n")
        else:
            print("All nodes have been successfully covered!\n")

    def save_zones_to_xml(self, output_file: str) -> None:
        """Save zones to an XML file"""
        root = Et.Element("zones")
        root.set("version", "1.0")

        for zone in self.zones:
            zone_elem = Et.SubElement(root, "zone")
            zone_elem.set("id", zone.id)
            zone_elem.set("x", f"{zone.x:.2f}")
            zone_elem.set("y", f"{zone.y:.2f}")
            zone_elem.set("radius", f"{zone.radius:.2f}")

        xml_str = minidom.parseString(Et.tostring(root)).toprettyxml(indent="    ")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(xml_str)

    def save_fixed_fog_nodes_to_xml(self, output_file: str) -> None:
        """Save fixed fog nodes to an XML file"""
        root = Et.Element("nodes")
        root.set("version", "1.0")

        for node in self.fixed_fog_nodes:
            node_elem = Et.SubElement(root, "node")
            node_elem.set("id", node.id)
            node_elem.set("x", f"{node.x:.2f}")
            node_elem.set("y", f"{node.y:.2f}")
            node_elem.set("power", f"{node.power:.2f}")
            node_elem.set("radius", f"{node.radius:.2f}")

        xml_str = minidom.parseString(Et.tostring(root)).toprettyxml(indent="    ")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(xml_str)


def parse_nodes(content: str) -> List[Node]:
    """Parse XML content and extract nodes"""
    root = Et.fromstring(content)
    nodes = root.findall('.//node')
    return [
        Node(
            id=node.get('id'),
            x=float(node.get('x')),
            y=float(node.get('y'))
        ) for node in nodes
    ]


def main(content: str, zone_output_file: str, fixed_fog_node_output_file: str):
    nodes = parse_nodes(content)

    best_generator = None
    best_coverage = -1
    best_std_dev = float('inf')

    print(
        f"Starting optimization: Running the algorithm {Config.Simulation.RUN_COUNT} times to find the best solution...\n")

    for i in range(Config.Simulation.RUN_COUNT):
        # Create a new generator for each run
        current_generator = BalancedZoneGenerator()
        current_generator.generate_zones(nodes)

        # Get stats for the current run
        covered_nodes_count, std_dev = current_generator.get_coverage_stats(nodes)

        # Print progress
        print(f"Run {i + 1}/{Config.Simulation.RUN_COUNT}: Covered {covered_nodes_count} nodes, StdDev: {std_dev:.2f}")

        # --- Decision Logic: Check if the current run is better than the best one found so far ---
        is_better = False
        if covered_nodes_count > best_coverage:
            is_better = True
        elif covered_nodes_count == best_coverage and std_dev < best_std_dev:
            is_better = True

        if is_better:
            print(f"Found a new best solution in run {i + 1}!")
            best_coverage = covered_nodes_count
            best_std_dev = std_dev
            # Use deepcopy to save a snapshot of the generator state
            best_generator = copy.deepcopy(current_generator)

    print("\n--- Optimization Finished ---")
    if best_generator:
        print("Displaying the best result found:")
        best_generator.print_info(nodes)
        best_generator.save_zones_to_xml(zone_output_file)
        best_generator.save_fixed_fog_nodes_to_xml(fixed_fog_node_output_file)
        print(f"Best solution saved to {zone_output_file} and {fixed_fog_node_output_file}")
    else:
        print("No solution could be found.")


if __name__ == "__main__":
    try:
        # Make sure to have the 'data' directory or change the file paths
        with open('data/hamburg.nod.xml', 'r', encoding='utf-8') as file:
            xml_content = file.read()
        main(xml_content, 'data/hamburg.zon.xml', 'data/hamburg.fn.xml')
    except FileNotFoundError:
        print("Error: Could not find the input file 'data/hamburg.nod.xml'.")
        print("Please make sure the file exists and the path is correct.")