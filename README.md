# D-STAR: Digital Twin-Guided Spatio-Temporal Graph Transformer RL

## About The Paper

Internet-of-Vehicles (IoV) systems are evolving into heterogeneous vehicular IoT ecosystems that combine connected vehicles, roadside infrastructure, edge computing nodes, and cloud platforms to support latency-sensitive and computation-intensive services. However, conventional task-offloading policies that optimize mean end-to-end latency can fail to control the tail of the latency distribution under mobility, wireless contention, and environmental disruptions.

To address this gap, this repository implements D-STAR, a digital twin-guided risk-aware reinforcement learning framework. D-STAR treats vehicular edge offloading as a tail-risk-constrained online orchestration problem. It uses real-time telemetry to characterize mobility and bandwidth conditions, learning resource-allocation policies that explicitly target the Conditional Value-at-Risk (CVaR) of end-to-end task latency.

## Manual Scenario Configuration

Before running the simulation, you must manually configure the scenario data. For any specific scenario (for example, Melbourne with 200 vehicles in rainy conditions or Hamburg with 1000 vehicles in snowy conditions), please follow these steps:

1. Navigate to the `scenario` folder and locate your desired scenario directory.
2. Find the `chunk_0.xml` files related to `vehicles` and `tasks` inside that specific directory.
3. Copy these files and replace the existing `chunk_0.xml` files located in the `data/vehicles/` and `data/tasks/` paths respectively.

## Configuration and Path Settings

Before executing the simulation, you must configure the environment path settings inside the `config.py` file. Open `config.py` and update the absolute paths for the accident files to match your local repository location:

```python
RAIN_ACCIDENT = f"E:\\YOUR_LOCAL_PATH\\data\\accidents\\rain_accidents.xml"
SNOW_ACCIDENT = f"E:\\YOUR_LOCAL_PATH\\data\\accidents\\snow_accidents.xml"
```

## How to Run

Once the scenario files are manually placed in the correct data directories, you can start the project by running the main script:

```bash
python main.py