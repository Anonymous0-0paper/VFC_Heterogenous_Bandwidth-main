class Config:
    CHUNK_SIZE = 1200
    NEGATIVE_REWARD = -10

    class SimulatorConfig:
        SIMULATION_DURATION = 1200
        BANDWIDTH = 150

    class BandwidthCandidates:
        B1 = 5
        B2 = 10
        B3 = 20
        B4 = 40
        B5 = 80

    class CloudConfig:
        DEFAULT_X = 6000
        DEFAULT_Y = 1500
        DEFAULT_RADIUS = 10000
        CLOUD_BANDWIDTH = 60
        MAX_TASK_QUEUE_LEN = 2000
        DEFAULT_COMPUTATION_POWER = 4000
        POWER_LIMIT = 0.99

    class FixedFogNodeConfig:
        MAX_TASK_QUEUE_LEN = 400
        DEFAULT_COMPUTATION_POWER = 500
        POWER_LIMIT = 0.9

    class MobileFogNodeConfig:
        DEFAULT_RADIUS = 150
        MAX_TASK_QUEUE_LEN = 150
        DEFAULT_COMPUTATION_POWER = 200
        POWER_LIMIT = 0.6

    class UserNodeConfig:
        MAX_TASK_QUEUE_LEN = 10
        DEFAULT_COMPUTATION_POWER = 20
        LOCAL_OFFLOAD_POWER_OVERHEAD = 1
        LOCAL_EXECUTE_TIME_OVERHEAD = 1
        POWER_LIMIT = 0.4

    class ZoneManagerConfig:
        ALGORITHM_RANDOM = "Random"
        ALGORITHM_HEURISTIC = "Heuristic"
        ALGORITHM_ONLY_CLOUD = "Only Cloud"
        ALGORITHM_ONLY_FOG = "Only Fog"
        ALGORITHM_DEEP_RL = "DeepRL"
        ALGORITHM_MADDPG = "MADDPG"
        ALGORITHM_DDPG = "DDPG"
        ALGORITHM_PPO = "PPO"
        ALGORITHM_SAC = "SAC"
        ALGORITHM_DEEP_RL_BANDWIDTH = "DeepRL_Bandwidth"
        ALGORITHM_DDPG_BANDWIDTH = "DDPG_Bandwidth"
        ALGORITHM_STGT = "STGT"  # Spatio-Temporal Graph Transformer
        ALGORITHM_PROP_FAIR = "ProportionalFair"  # Proportional-Fair baseline
        ALGORITHM_MILP = "ChanceConstrainedMILP"  # Chance-Constrained MILP baseline
        ALGORITHM_CVARDQN = "CVaRDQN"  # CVaR-DQN baseline
        ALGORITHM_PPO_BANDWIDTH = "PPO_Bandwidth"

        DEFAULT_ALGORITHM = ALGORITHM_DDPG

    class FinalDeciderMethod:
        FIRST_CHOICE = "First Choice"
        RANDOM_CHOICE = "Random Choice"
        MIN_DISTANCE = "Min Distance"

        DEFAULT_METHOD = FIRST_CHOICE

    class RandomZoneManagerConfig:
        OFFLOAD_CHANCE: float = 0.5

    class Scenario:
        BASE_SCENARIO = "Base Scenario"
        HEAVY_RAIN = "Heavy Rain"
        HEAVY_SNOW = "Heavy Snow"
        RAIN_AND_ACCIDENT = "Rain and Accident"
        SNOW_AND_ACCIDENT = "Snow and Accident"

        START_TIME = 300
        FINISH_TIME = 700

        DEFAULT_SCENARIO = BASE_SCENARIO

    class TaskConfig:
        PACKET_COST_PER_METER = 0.001

        TASK_COST_PER_METER = 0.005

        MIGRATION_OVERHEAD = 0.01
        CLOUD_PROCESSING_OVERHEAD = 0.5

    class STGTConfig:
        """Configuration for Spatio-Temporal Graph Transformer"""
        FEATURE_DIM = 6  # [queue_length, sinr, bandwidth_budget, x, y, time_embedding]
        HIDDEN_DIM = 128
        NUM_LAYERS = 2
        NUM_HEADS = 2
        D_FF = 512
        DROPOUT = 0.1
        K_NEIGHBORS = 10
        ALPHA = 0.95  # CVaR confidence level
        LAMBDA1 = 1.0  # Throughput multiplier
        LAMBDA2 = 0.1  # Fairness multiplier
        L_MAX = 100.0  # Maximum latency constraint
        MAX_WINDOW_SIZE = 4  # Temporal window size
        BATCH_SIZE = 64
        MAX_BUFFER_SIZE = 10000
        CLIP_EPSILON = 0.2  # PPO clipping parameter
        POLICY_LR = 3e-4
        CRITIC_LR = 1e-3
        DUAL_LR = 1e-4


