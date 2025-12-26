import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import Config
from controllers.loader import Loader
from controllers.simulator import Simulator, red_bg
from models.node.cloud import CloudNode
from utils.clock import Clock

from controllers.Simulator.simulator_ddpg import SimulatorDDPG
from controllers.Simulator.simulator_ppo import SimulatorPPO
from controllers.Simulator.simulator_sac import SimulatorSAC
from controllers.Simulator.simulator_DeepRL_Bandwidth import Simulator as Simulator_DeepRL_Bandwidth
from controllers.Simulator.simulator_ddpg_Bandwidth import SimulatorDDPG as Simulator_DDPG_Bandwidth
from controllers.Simulator.simulator_ppo_Bandwidth import SimulatorPPO as Simulator_PPO_Bandwidth
from controllers.Simulator.simulator_stgt import SimulatorSTGT


def visualize_metrics(metrics_controller):
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    overview_data = [
        metrics_controller.completed_tasks,
        metrics_controller.migrations_count,
        metrics_controller.deadline_misses,
        metrics_controller.cloud_tasks
    ]
    labels = ['Completed Tasks', 'Migrations', 'Deadline Misses', 'Cloud Tasks']
    colors = ['#00C49F', '#0088FE', '#FF8042', '#FFBB28']
    ax1.pie(overview_data, labels=labels, colors=colors, autopct='%1.1f%%')
    ax1.set_title('System Overview')

    ax2 = fig.add_subplot(gs[0, 1])
    steps = range(1, len(metrics_controller.migration_counts_per_step) + 1)
    ax2.plot(steps, metrics_controller.deadline_misses_per_step,
             label='Deadline Misses', color='#FF8042')
    ax2.plot(steps, metrics_controller.completed_task_per_step,
             label='Completed Tasks', color='#00C49F')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Count')
    ax2.set_title('Metrics Per Step')
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, :])
    categories = ['Total Tasks', 'Completed Tasks', 'Missed Tasks']
    values = [
        metrics_controller.total_tasks,
        metrics_controller.completed_tasks,
        metrics_controller.deadline_misses,
    ]
    colors = ['#8884d8', '#00C49F', '#FFBB28']
    bars = ax3.bar(categories, values, color=colors)
    ax3.set_title('Task Distribution')
    ax3.set_ylabel('Count')

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_load_differences(metrics_controller, num_points=50):
    task_ids = sorted(metrics_controller.task_load_diff.keys())
    min_loads = [metrics_controller.task_load_diff[id][0] for id in task_ids]
    max_loads = [metrics_controller.task_load_diff[id][1] for id in task_ids]

    if not task_ids:
        print("No data to plot")
        return

    if len(task_ids) > num_points:
        indices = np.linspace(0, len(task_ids) - 1, num_points, dtype=int)
        task_ids = [task_ids[i] for i in indices]
        min_loads = [min_loads[i] for i in indices]
        max_loads = [max_loads[i] for i in indices]

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(task_ids)), min_loads, 'b-', label='Min Load', alpha=0.7)
    plt.plot(range(len(task_ids)), max_loads, 'r-', label='Max Load', alpha=0.7)
    plt.fill_between(range(len(task_ids)), min_loads, max_loads, alpha=0.2)

    show_n_ticks = min(10, len(task_ids))
    tick_indices = np.linspace(0, len(task_ids) - 1, show_n_ticks, dtype=int)
    plt.xticks(tick_indices, [task_ids[i] for i in tick_indices], rotation=45)

    plt.xlabel('Task ID')
    plt.ylabel('Load')
    plt.title('Task Load Range')
    plt.legend()
    plt.grid(True)

    avg_diff = np.mean([max_loads[i] - min_loads[i] for i in range(len(task_ids))])
    plt.annotate(f'Average Load Difference: {avg_diff:.3f}',
                 xy=(0.02, 0.98), xycoords='axes fraction',
                 bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


def run_one(params):
    algorithm, method = params
    Config.ZoneManagerConfig.DEFAULT_ALGORITHM = algorithm
    Config.FinalDeciderMethod.DEFAULT_METHOD = method

    loader = Loader(
        zone_file= Config.Directory.zon,
        fixed_fn_file= Config.Directory.fn,
        mobile_file="./data/vehicles",
        task_file="./data/tasks",
        checkpoint_path="./checkpoints",
    )
    cloud = CloudNode(
        id="CLOUD0",
        x=Config.CloudConfig.DEFAULT_X,
        y=Config.CloudConfig.DEFAULT_Y,
        power=Config.CloudConfig.DEFAULT_COMPUTATION_POWER,
        remaining_power=Config.CloudConfig.DEFAULT_COMPUTATION_POWER,
        radius=Config.CloudConfig.DEFAULT_RADIUS,
    )

    if algorithm == Config.ZoneManagerConfig.ALGORITHM_DDPG:
        sim = SimulatorDDPG(loader, Clock(), cloud)
    elif algorithm == Config.ZoneManagerConfig.ALGORITHM_PPO:
        sim = SimulatorPPO(loader, Clock(), cloud)
    elif algorithm == Config.ZoneManagerConfig.ALGORITHM_SAC:
        sim = SimulatorSAC(loader, Clock(), cloud)
    elif algorithm == Config.ZoneManagerConfig.ALGORITHM_DDPG_BANDWIDTH:
        sim = Simulator_DDPG_Bandwidth(loader, Clock(), cloud)
    elif algorithm == Config.ZoneManagerConfig.ALGORITHM_DEEP_RL_BANDWIDTH:
        sim = Simulator_DeepRL_Bandwidth(loader, Clock(), cloud)
    elif algorithm == Config.ZoneManagerConfig.ALGORITHM_PPO_BANDWIDTH:
        sim = Simulator_PPO_Bandwidth(loader, Clock(), cloud)
    elif algorithm == Config.ZoneManagerConfig.ALGORITHM_STGT:
        sim = SimulatorSTGT(loader, Clock(), cloud)
    else:
        sim = Simulator(loader, Clock(), cloud)

    sim.start_simulation()
    return {
        "algorithm": algorithm,
        "method": method,
        "total": sim.metrics.total_tasks,
        "completed": sim.metrics.completed_tasks,
        "missed": sim.metrics.deadline_misses,
        "migrations": sim.metrics.migrations_count,
        "cloud": sim.metrics.cloud_tasks,
        "metrics": sim.metrics,
    }

if __name__ == "__main__":
    algorithms = [
        # Config.ZoneManagerConfig.ALGORITHM_DEEP_RL_BANDWIDTH,
        Config.ZoneManagerConfig.ALGORITHM_PPO_BANDWIDTH,
        # Config.ZoneManagerConfig.ALGORITHM_PROP_FAIR,
        # Config.ZoneManagerConfig.ALGORITHM_MILP,
        # Config.ZoneManagerConfig.ALGORITHM_CVARDQN,
        # Config.ZoneManagerConfig.ALGORITHM_DDPG_BANDWIDTH,
    ]
    methods = [Config.FinalDeciderMethod.MIN_DISTANCE]
    tasks = [(a, m) for a in algorithms for m in methods]

    results = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_one, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            print(red_bg(f"Finished {res['algorithm']} / {res['method']}"))
            print("SCENARIO\tALGORITHM\tMETHOD\tTOTAL\tCOMPLETED\tMISSED\tMIGRATIONS\tCLOUD")
            print(
                f"Rainy\t{res['algorithm']}\t{res['method']}\t"
                f"{res['total']}\t{res['completed']}\t{res['missed']}\t"
                f"{res['migrations']}\t{res['cloud']}"
            )
            results.append(res)

    # To visualize the last run's metrics:
    # visualize_metrics(results[-1]["metrics"])
    # plot_load_differences(results[-1]["metrics"])
