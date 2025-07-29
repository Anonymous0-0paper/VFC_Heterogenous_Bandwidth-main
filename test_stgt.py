# #!/usr/bin/env python3
# """
# Test script for Spatio-Temporal Graph Transformer (STGT) implementation
# """
#
# import torch
# import numpy as np
# import sys
# import os
#
# # Add the project root to the path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#
# def test_stgt_encoder():
#     """Test the STGT encoder functionality"""
#     print("Testing STGT Encoder...")
#
#     try:
#         from controllers.zone_managers.STGT.stgt_zone_manager import STGTEncoder, NodeFeatures
#
#         # Create encoder
#         encoder = STGTEncoder(NodeFeatures
#             feature_dim=6,
#             hidden_dim=64,
#             num_layers=2,
#             num_heads=4,
#             d_ff=128,
#             dropout=0.1,
#             k_neighbors=5
#         )
#
#         # Create synthetic snapshots
#         snapshots = []
#         for t in range(4):
#             snapshot = {
#                 'nodes': {},
#                 'edges': [],
#                 'timestamp': t * 10
#             }
#
#             # Add nodes
#             for i in range(5):
#                 node_id = f"node_{i}_{t}"
#                 snapshot['nodes'][node_id] = NodeFeatures(
#                     queue_length=np.random.uniform(0, 10),
#                     sinr=np.random.uniform(5, 25),
#                     bandwidth_budget=np.random.uniform(50, 200),
#                     position=(np.random.uniform(0, 1000), np.random.uniform(0, 1000)),
#                     time_embedding=np.sin(2 * np.pi * t / 4)
#                 )
#
#             snapshots.append(snapshot)
#
#         # Test forward pass
#         node_embeddings, global_context = encoder(snapshots)
#         print(f"✓ STGT Encoder forward pass successful")
#         print(f"  - Node embeddings shape: {node_embeddings.shape}")
#         print(f"  - Global context shape: {global_context.shape}")
#         return True
#     except Exception as e:
#         print(f"✗ STGT Encoder forward pass failed: {e}")
#         return False
#
#
# def test_risk_critic():
#     """Test the risk critic functionality"""
#     print("\nTesting Risk Critic...")
#
#     try:
#         from controllers.zone_managers.STGT.stgt_zone_manager import RiskCritic
#
#         # Create risk critic
#         critic = RiskCritic(input_dim=64, hidden_dim=128)
#
#         # Test forward pass
#         state = torch.randn(1, 64)
#         cvar_estimate = critic(state)
#         print(f"✓ Risk Critic forward pass successful")
#         print(f"  - CVaR estimate shape: {cvar_estimate.shape}")
#         return True
#     except Exception as e:
#         print(f"✗ Risk Critic forward pass failed: {e}")
#         return False
#
#
# def test_stgt_zone_manager():
#     """Test the STGT zone manager functionality"""
#     print("\nTesting STGT Zone Manager...")
#
#     try:
#         from controllers.zone_managers.STGT.stgt_zone_manager import STGTZoneManager
#         from models.zone import Zone
#         from models.node.fog import FixedFogNode, MobileFogNode
#         from models.node.user import UserNode
#         from models.task import Task
#
#         # Create zone
#         zone = Zone(0, 0, 1000, 1000)
#
#         # Create STGT zone manager
#         stgt_manager = STGTZoneManager(
#             zone=zone,
#             feature_dim=6,
#             hidden_dim=64,
#             num_layers=2,
#             k_neighbors=5,
#             alpha=0.95,
#             lambda1=1.0,
#             lambda2=0.1,
#             l_max=100.0
#         )
#
#         # Create test nodes
#         fixed_node = FixedFogNode(
#             id="fixed_1",
#             x=100,
#             y=100,
#             power=500,
#             remaining_power=500,
#             radius=200
#         )
#
#         mobile_node = MobileFogNode(
#             id="mobile_1",
#             x=200,
#             y=200,
#             power=200,
#             remaining_power=200,
#             radius=150
#         )
#
#         user_node = UserNode(
#             id="user_1",
#             x=150,
#             y=150,
#             power=20,
#             remaining_power=20,
#             radius=50
#         )
#
#         # Add nodes to zone manager
#         stgt_manager.add_fixed_fog_nodes([fixed_node])
#         stgt_manager.set_mobile_fog_nodes([mobile_node])
#
#         # Create test task
#         task = Task(
#             release_time=0.0,
#             deadline=100.0,
#             exec_time=10.0,
#             power=5.0,
#             creator_id="user_1",
#             dataSize=100.0,
#             creator=user_node
#         )
#
#         # Test task assignment
#         can_offload = stgt_manager.can_offload_task(task)
#         print(f"✓ Can offload task: {can_offload}")
#
#         if can_offload:
#             executor = stgt_manager.assign_task(task)
#             print(f"✓ Task assigned to: {executor.id if executor else 'None'}")
#
#         return True
#     except Exception as e:
#         print(f"✗ STGT Zone Manager task assignment failed: {e}")
#         return False
#
#
# def test_cvar_calculation():
#     """Test CVaR calculation"""
#     print("\nTesting CVaR Calculation...")
#
#     try:
#         from controllers.zone_managers.STGT.stgt_zone_manager import STGTZoneManager
#         from models.zone import Zone
#
#         # Create zone manager for testing
#         zone = Zone(0, 0, 1000, 1000)
#         stgt_manager = STGTZoneManager(zone=zone)
#
#         # Test data
#         latencies = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
#         alpha = 0.9
#
#         cvar = stgt_manager.calculate_cvar(latencies, alpha)
#         expected_cvar = np.mean(sorted(latencies)[:int(alpha * len(latencies))])
#
#         print(f"✓ CVaR calculation successful")
#         print(f"  - Calculated CVaR: {cvar:.2f}")
#         print(f"  - Expected CVaR: {expected_cvar:.2f}")
#         print(f"  - Difference: {abs(cvar - expected_cvar):.2f}")
#
#         return abs(cvar - expected_cvar) < 1e-6
#     except Exception as e:
#         print(f"✗ CVaR calculation failed: {e}")
#         return False
#
#
# def test_integration_with_config():
#     """Test integration with configuration system"""
#     print("\nTesting Configuration Integration...")
#
#     try:
#         from config import Config
#
#         # Check if STGT config exists
#         assert hasattr(Config, 'STGTConfig'), "STGTConfig not found in Config"
#
#         # Check required config parameters
#         required_params = [
#             'FEATURE_DIM', 'HIDDEN_DIM', 'NUM_LAYERS', 'NUM_HEADS',
#             'D_FF', 'DROPOUT', 'K_NEIGHBORS', 'ALPHA', 'LAMBDA1',
#             'LAMBDA2', 'L_MAX', 'MAX_WINDOW_SIZE', 'BATCH_SIZE',
#             'MAX_BUFFER_SIZE', 'CLIP_EPSILON', 'POLICY_LR', 'CRITIC_LR', 'DUAL_LR'
#         ]
#
#         for param in required_params:
#             assert hasattr(Config.STGTConfig, param), f"Missing config parameter: {param}"
#
#         # Check if STGT algorithm is registered
#         assert hasattr(Config.ZoneManagerConfig, 'ALGORITHM_STGT'), "STGT algorithm not registered"
#         assert Config.ZoneManagerConfig.ALGORITHM_STGT == "STGT", "STGT algorithm name mismatch"
#
#         print(f"✓ Configuration integration successful")
#         print(f"  - STGT algorithm: {Config.ZoneManagerConfig.ALGORITHM_STGT}")
#         print(f"  - Feature dimension: {Config.STGTConfig.FEATURE_DIM}")
#         print(f"  - Hidden dimension: {Config.STGTConfig.HIDDEN_DIM}")
#         print(f"  - Number of layers: {Config.STGTConfig.NUM_LAYERS}")
#
#         return True
#     except Exception as e:
#         print(f"✗ Configuration integration test failed: {e}")
#         return False
#
#
# def main():
#     """Run all tests"""
#     print("=" * 60)
#     print("STGT Implementation Test Suite")
#     print("=" * 60)
#
#     tests = [
#         test_stgt_encoder,
#         test_risk_critic,
#         test_stgt_zone_manager,
#         test_cvar_calculation,
#         test_integration_with_config
#     ]
#
#     passed = 0
#     total = len(tests)
#
#     for test in tests:
#         try:
#             if test():
#                 passed += 1
#         except Exception as e:
#             print(f"✗ Test {test.__name__} failed with exception: {e}")
#
#     print("\n" + "=" * 60)
#     print(f"Test Results: {passed}/{total} tests passed")
#     print("=" * 60)
#
#     if passed == total:
#         print("🎉 All tests passed! STGT implementation is working correctly.")
#         return 0
#     else:
#         print("❌ Some tests failed. Please check the implementation.")
#         return 1
#
#
# if __name__ == "__main__":
#     exit(main())

import torch
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())
