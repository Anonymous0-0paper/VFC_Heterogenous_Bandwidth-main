import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math
import logging
import threading
from datetime import datetime

from config import Config
from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
from models.node.fog import FogLayerABC
from models.task import Task
from models.zone import Zone
from utils.enums import Layer

# Logging configuration
logging.basicConfig(
    filename="stgt_logs.txt",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
@dataclass
class NodeFeatures:
    """Node feature vector for STGT"""
    queue_length: float
    sinr: float  # Link-level SINR
    bandwidth_budget: float
    position: Tuple[float, float]  # 2D position (x, y)
    time_embedding: float  # Sinusoidal time embedding


@dataclass
class EdgeFeatures:
    """Edge feature vector for STGT"""
    path_loss: float
    relative_speed: float


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for temporal information"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for STGT"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            scores = scores.masked_fill(~mask, -1e9)

        print("scores shape:", scores.shape)
        print("mask shape:", mask.shape)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(context)


class FeedForward(nn.Module):
    """Feed-forward network for transformer layers"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerLayer(nn.Module):
    """Single transformer layer for STGT"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class STGTEncoder(nn.Module):
    """Spatio-Temporal Graph Transformer Encoder"""
    
    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 2, 
                 num_heads: int = 8, d_ff: int = 512, dropout: float = 0.1, 
                 k_neighbors: int = 10):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k_neighbors = k_neighbors
        
        # Feature projection
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        
        # Time embedding
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
    
    def merge_snapshots(self, snapshots: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Merge temporal snapshots into a single spatio-temporal graph"""
        all_nodes = []
        all_edges = []
        
        for t, snapshot in enumerate(snapshots):
            # Add temporal offset to node IDs
            for node_id, node_features in snapshot['nodes'].items():
                temporal_node_id = f"{node_id}_{t}"
                all_nodes.append((temporal_node_id, node_features, t))
            
            # Add temporal edges
            for edge in snapshot['edges']:
                all_edges.append(edge)
        
        return all_nodes, all_edges
    
    def nearest_neighbors(self, node_positions: torch.Tensor, k: int) -> torch.Tensor:
        """Find k nearest neighbors for each node"""
        # Compute pairwise distances
        dist_matrix = torch.cdist(node_positions, node_positions)
        
        # Get k nearest neighbors (excluding self)
        k_neighbors = torch.topk(dist_matrix, k=k+1, dim=-1, largest=False)[1][:, 1:]
        
        return k_neighbors
    
    def forward(self, snapshots: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of STGT encoder
        
        Args:
            snapshots: List of graph snapshots over time window W
            
        Returns:
            node_embeddings: Node embeddings for all nodes
            global_context: Global context vector
        """
        # Merge snapshots
        all_nodes, all_edges = self.merge_snapshots(snapshots)
        
        # Extract features and positions
        node_features = []
        node_positions = []
        
        for node_id, features, t in all_nodes:
            # Create feature vector: [queue_length, sinr, bandwidth_budget, x, y, time_embedding]
            feature_vec = torch.tensor([
                features.queue_length,
                features.sinr,
                features.bandwidth_budget,
                features.position[0],
                features.position[1],
                features.time_embedding
            ], dtype=torch.float32)
            
            node_features.append(feature_vec)
            node_positions.append(torch.tensor(features.position, dtype=torch.float32))
        
        # Stack into tensors
        node_features = torch.stack(node_features)  # [N, feature_dim]
        node_positions = torch.stack(node_positions)  # [N, 2]

        device = self.feature_projection.weight.device
        node_features = node_features.to(device)
        node_positions = node_positions.to(device)

        # Project features to hidden dimension
        x = self.feature_projection(node_features)  # [N, hidden_dim]
        
        # Add time embeddings
        x = self.time_embedding(x.unsqueeze(0)).squeeze(0)  # [N, hidden_dim]
        
        # Find k-nearest neighbors
        neighbor_indices = self.nearest_neighbors(node_positions, self.k_neighbors)
        
        # Apply transformer layers
        # Apply transformer layers
        for layer in self.transformer_layers:
            num_nodes = x.size(0)
            mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=x.device)

            # Allow attention only to k nearest neighbors
            for i in range(num_nodes):
                mask[i, neighbor_indices[i]] = True
                mask[i, i] = True  # Allow self-attention

            # Reshape to [1, 1, N, N] to match attention scores shape [B, H, N, N]
            mask = mask.unsqueeze(0).unsqueeze(0)

            x = layer(x.unsqueeze(0), mask).squeeze(0)

        # Final projection
        node_embeddings = self.output_projection(x)
        
        # Global context via mean pooling
        global_context = torch.mean(node_embeddings, dim=0)
        
        return node_embeddings, global_context


class RiskCritic(nn.Module):
    """Risk critic for CVaR estimation"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class STGTZoneManager(ZoneManagerABC, nn.Module):
    def __init__(self, zone: Zone, feature_dim: int = 6, hidden_dim: int = 128,
                 num_layers: int = 2, k_neighbors: int = 10, alpha: float = 0.95,
                 lambda1: float = 1.0, lambda2: float = 0.1, l_max: float = 100.0):
        super().__init__(zone)
        nn.Module.__init__(self)

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.l_max = l_max

        self.stgt_encoder = STGTEncoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            k_neighbors=k_neighbors
        )

        self.risk_critic = RiskCritic(hidden_dim)

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.snapshot_window = []
        self.max_window_size = 4
        self.psi = torch.tensor(0.0, requires_grad=True)

        self.policy_optimizer = torch.optim.Adam(
            list(self.stgt_encoder.parameters()) + list(self.policy_head.parameters()),
            lr=3e-4
        )
        self.critic_optimizer = torch.optim.Adam(self.risk_critic.parameters(), lr=1e-3)
        self.dual_optimizer = torch.optim.Adam([self.psi], lr=1e-4)

        self.experience_buffer = []
        self.max_buffer_size = 10000
        self.clip_epsilon = 0.2
        self.batch_size = 64

        self.start_logging()  # Start periodic logging

    def start_logging(self):
        """Start periodic logging every 60 seconds."""
        def log_metrics():
            self.log_status()
            threading.Timer(60.0, log_metrics).start()

        log_metrics()

    def log_status(self):
        """Logs current state of the zone manager for monitoring."""
        try:
            if self.snapshot_window:
                latest_snapshot = self.snapshot_window[-1]
                node_count = len(latest_snapshot['nodes'])
                edge_count = len(latest_snapshot['edges'])
                timestamp = latest_snapshot.get('timestamp', 'N/A')

                logging.info(f"Snapshot @ {timestamp}: {node_count} nodes, {edge_count} edges.")

            if self.experience_buffer:
                last_episode = self.experience_buffer[-1]
                last_reward = last_episode.get('reward', 'N/A')
                last_cvar = last_episode.get('cvar', 'N/A')

                logging.info(f"Latest Experience — Reward: {last_reward:.4f}, CVaR: {last_cvar:.4f}")
            else:
                logging.info("No experience in buffer yet.")
        except Exception as e:
            logging.error(f"Error in log_status: {e}")

    def create_node_features(self, node: FogLayerABC, current_time: float) -> NodeFeatures:
        """Create node features for STGT"""
        # Calculate SINR (simplified)
        sinr = self._calculate_sinr(node)
        
        # Get queue length
        queue_length = len(node.task_queue) if hasattr(node, 'task_queue') else 0
        
        # Get bandwidth budget
        bandwidth_budget = node.remaining_power if hasattr(node, 'remaining_power') else 100
        
        # Position
        position = (node.x, node.y)
        
        # Time embedding
        time_embedding = math.sin(2 * math.pi * current_time / 1000)  # Sinusoidal embedding
        
        return NodeFeatures(
            queue_length=queue_length,
            sinr=sinr,
            bandwidth_budget=bandwidth_budget,
            position=position,
            time_embedding=time_embedding
        )
    
    def _calculate_sinr(self, node: FogLayerABC) -> float:
        """Calculate link-level SINR (simplified model)"""
        # Simplified SINR calculation based on distance and interference
        base_sinr = 20.0  # Base SINR in dB
        
        # Distance-based path loss
        if hasattr(node, 'x') and hasattr(node, 'y'):
            distance = math.sqrt(node.x**2 + node.y**2)
            path_loss = 20 * math.log10(distance + 1)  # Log-distance path loss
        else:
            path_loss = 0
        
        # Interference (simplified)
        interference = 5.0  # dB
        
        sinr = base_sinr - path_loss - interference
        return max(sinr, 0.1)  # Ensure positive SINR
    
    def create_edge_features(self, node1: FogLayerABC, node2: FogLayerABC) -> EdgeFeatures:
        """Create edge features for STGT"""
        # Calculate path loss
        distance = math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
        path_loss = 20 * math.log10(distance + 1)
        
        # Calculate relative speed (simplified)
        relative_speed = 0.0  # Placeholder for relative speed calculation
        
        return EdgeFeatures(
            path_loss=path_loss,
            relative_speed=relative_speed
        )
    
    def create_graph_snapshot(self, current_time: float) -> Dict:
        """Create a graph snapshot for the current time step"""
        nodes = {}
        edges = []
        
        # Add fixed fog nodes
        for node_id, node in self.fixed_fog_nodes.items():
            nodes[node_id] = self.create_node_features(node, current_time)
        
        # Add mobile fog nodes
        for node_id, node in self.mobile_fog_nodes.items():
            nodes[node_id] = self.create_node_features(node, current_time)
        
        # Create edges between nodes (simplified - connect all nodes within range)
        node_list = list(nodes.keys())
        for i, node1_id in enumerate(node_list):
            for j, node2_id in enumerate(node_list[i+1:], i+1):
                # Check if nodes are within communication range
                node1 = self.fixed_fog_nodes.get(node1_id) or self.mobile_fog_nodes.get(node1_id)
                node2 = self.fixed_fog_nodes.get(node2_id) or self.mobile_fog_nodes.get(node2_id)
                
                if node1 and node2:
                    distance = math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
                    if distance <= 300:  # Communication range
                        edge_features = self.create_edge_features(node1, node2)
                        edges.append({
                            'source': node1_id,
                            'target': node2_id,
                            'features': edge_features
                        })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'timestamp': current_time
        }
    
    def update_snapshot_window(self, current_time: float):
        """Update the temporal snapshot window"""
        snapshot = self.create_graph_snapshot(current_time)
        self.snapshot_window.append(snapshot)
        
        # Keep only the most recent snapshots
        if len(self.snapshot_window) > self.max_window_size:
            self.snapshot_window.pop(0)
    
    def calculate_cvar(self, latencies: List[float], alpha: float) -> float:
        """Calculate Conditional Value at Risk (CVaR)"""
        if not latencies:
            return 0.0
        
        latencies = sorted(latencies)
        n = len(latencies)
        k = int(alpha * n)
        
        if k == 0:
            return latencies[0]
        
        # Calculate VaR (Value at Risk)
        var = latencies[k-1]
        
        # Calculate CVaR
        cvar = sum(latencies[:k]) / k
        
        return cvar
    
    def calculate_spectral_efficiency(self, sinr_values: List[float]) -> float:
        """Calculate cumulative spectral efficiency"""
        if not sinr_values:
            return 0.0
        
        # Shannon capacity formula
        spectral_efficiency = sum(math.log2(1 + sinr) for sinr in sinr_values)
        return spectral_efficiency
    
    def calculate_bandwidth_fairness(self, bandwidth_allocations: List[float]) -> float:
        """Calculate bandwidth allocation variance (fairness measure)"""
        if not bandwidth_allocations:
            return 0.0
        
        mean_allocation = sum(bandwidth_allocations) / len(bandwidth_allocations)
        variance = sum((alloc - mean_allocation) ** 2 for alloc in bandwidth_allocations) / len(bandwidth_allocations)
        
        return variance
    
    def compute_objective(self, episode_data: Dict) -> float:
        """Compute the risk-constrained objective"""
        latencies = episode_data.get('latencies', [])
        sinr_values = episode_data.get('sinr_values', [])
        bandwidth_allocations = episode_data.get('bandwidth_allocations', [])
        
        # Calculate components
        cvar_latency = self.calculate_cvar(latencies, self.alpha)
        spectral_efficiency = self.calculate_spectral_efficiency(sinr_values)
        bandwidth_variance = self.calculate_bandwidth_fairness(bandwidth_allocations)
        
        # Objective function
        objective = cvar_latency - self.lambda1 * spectral_efficiency + self.lambda2 * bandwidth_variance
        
        # Lagrangian relaxation for safety constraint
        safety_penalty = self.psi * max(0, cvar_latency - self.l_max)
        
        return objective + safety_penalty
    
    def update_policy(self, batch: List[Dict]):
        """Update policy using PPO with risk constraints"""
        if len(batch) < self.batch_size:
            return
        
        # Sample batch
        batch_data = batch[:self.batch_size]
        
        # Extract states and actions
        states = torch.stack([item['state'] for item in batch_data])
        actions = torch.stack([item['action'] for item in batch_data])
        old_log_probs = torch.stack([item['log_prob'] for item in batch_data])
        rewards = torch.tensor([item['reward'] for item in batch_data])
        
        # Compute advantages
        advantages = rewards - rewards.mean()
        advantages = advantages / (advantages.std() + 1e-8)
        
        # Policy update
        for _ in range(10):  # Multiple epochs
            # Forward pass
            node_embeddings, global_context = self.stgt_encoder([item['snapshots'] for item in batch_data])
            action_probs = self.policy_head(global_context.unsqueeze(0)).squeeze(0)
            
            # Calculate log probabilities
            log_probs = torch.log(action_probs + 1e-8)
            
            # PPO clipped ratio
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            
            # Policy loss
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
        
        # Risk critic update
        risk_estimates = self.risk_critic(states)
        risk_targets = torch.tensor([item['cvar'] for item in batch_data])
        critic_loss = F.mse_loss(risk_estimates.squeeze(), risk_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Dual variable update
        cvar_violations = torch.tensor([max(0, item['cvar'] - self.l_max) for item in batch_data])
        dual_loss = -self.psi * cvar_violations.mean()
        
        self.dual_optimizer.zero_grad()
        dual_loss.backward()
        self.dual_optimizer.step()
        
        # Ensure psi is non-negative
        with torch.no_grad():
            self.psi.clamp_(min=0.0)
    
    def can_offload_task(self, task: Task) -> bool:
        """Check if the zone can offload the given task"""
        merged_fog_nodes: Dict[str, FogLayerABC] = {**self.fixed_fog_nodes, **self.mobile_fog_nodes}
        
        # Check if any node can handle the task
        for fog_id, fog in merged_fog_nodes.items():
            if fog.can_offload_task(task):
                return True
        
        # Check if creator can handle the task locally
        if task.creator and task.creator.can_offload_task(task):
            return True
        
        return False
    
    def assign_task(self, task: Task) -> FogLayerABC:
        """Assign task using STGT policy"""
        # Update snapshot window
        current_time = task.release_time
        self.update_snapshot_window(current_time)
        
        # Ensure we have enough snapshots
        if len(self.snapshot_window) < 2:
            # Fallback to heuristic assignment
            return self._heuristic_assignment(task)
        
        # Create state representation
        with torch.no_grad():
            node_embeddings, global_context = self.stgt_encoder(self.snapshot_window)
            
            # Get action probability
            action_prob = self.policy_head(global_context.unsqueeze(0)).item()
        
        # Get available nodes
        available_nodes = []
        merged_fog_nodes: Dict[str, FogLayerABC] = {**self.fixed_fog_nodes, **self.mobile_fog_nodes}
        
        for fog_id, fog in merged_fog_nodes.items():
            if fog.can_offload_task(task):
                available_nodes.append(fog)
        
        if task.creator and task.creator.can_offload_task(task):
            available_nodes.append(task.creator)
        
        if not available_nodes:
            # No available nodes, return creator if possible
            return task.creator if task.creator else list(merged_fog_nodes.values())[0]
        
        # Use action probability to decide between local and remote execution
        if action_prob > 0.5 and task.creator and task.creator.can_offload_task(task):
            # Prefer local execution
            return task.creator
        else:
            # Choose best remote node based on embeddings
            best_node = None
            best_score = float('-inf')
            
            for node in available_nodes:
                if node == task.creator:
                    continue
                
                # Calculate node score based on features
                score = self._calculate_node_score(node, task)
                if score > best_score:
                    best_score = score
                    best_node = node
            
            return best_node if best_node else available_nodes[0]
    
    def _calculate_node_score(self, node: FogLayerABC, task: Task) -> float:
        """Calculate score for node selection"""
        # Distance-based score
        distance = math.sqrt((node.x - task.creator.x)**2 + (node.y - task.creator.y)**2)
        distance_score = 1.0 / (1.0 + distance)
        
        # Capacity-based score
        capacity_score = 0.0
        if hasattr(node, 'remaining_power'):
            capacity_score = node.remaining_power / 1000.0  # Normalize
        
        # Queue-based score
        queue_score = 0.0
        if hasattr(node, 'task_queue'):
            queue_length = len(node.task_queue)
            queue_score = 1.0 / (1.0 + queue_length)
        
        # Combined score
        total_score = distance_score + capacity_score + queue_score
        return total_score
    
    def _heuristic_assignment(self, task: Task) -> FogLayerABC:
        """Fallback heuristic assignment"""
        merged_fog_nodes: Dict[str, FogLayerABC] = {**self.fixed_fog_nodes, **self.mobile_fog_nodes}
        
        # Find node with highest remaining power
        best_node = None
        best_power = float('-inf')
        
        for fog_id, fog in merged_fog_nodes.items():
            if fog.can_offload_task(task):
                if hasattr(fog, 'remaining_power') and fog.remaining_power > best_power:
                    best_power = fog.remaining_power
                    best_node = fog
        
        if best_node:
            return best_node
        
        # Fallback to creator if possible
        if task.creator and task.creator.can_offload_task(task):
            return task.creator
        
        # Last resort - return first available node
        for fog_id, fog in merged_fog_nodes.items():
            if fog.can_offload_task(task):
                return fog
        
        return list(merged_fog_nodes.values())[0] if merged_fog_nodes else task.creator
    
    def update(self, **kwargs: ZoneManagerUpdate):
        """Update the zone manager with new information"""
        current_time = kwargs.get('current_time', 0.0)
        tasks = kwargs.get('tasks', [])
        
        # Update snapshot window
        self.update_snapshot_window(current_time)
        
        # Collect experience for training
        if tasks:
            episode_data = {
                'latencies': [],
                'sinr_values': [],
                'bandwidth_allocations': [],
                'snapshots': self.snapshot_window.copy()
            }
            
            # Process completed tasks
            for task in tasks:
                if task.is_completed:
                    # Calculate latency
                    latency = task.finish_time - task.release_time
                    episode_data['latencies'].append(latency)
                    
                    # Calculate SINR (simplified)
                    if task.executor:
                        sinr = self._calculate_sinr(task.executor)
                        episode_data['sinr_values'].append(sinr)
                    
                    # Bandwidth allocation (simplified)
                    if hasattr(task.executor, 'remaining_power'):
                        episode_data['bandwidth_allocations'].append(task.executor.remaining_power)
            
            # Add to experience buffer
            if episode_data['latencies']:
                reward = self.compute_objective(episode_data)
                episode_data['reward'] = reward
                episode_data['cvar'] = self.calculate_cvar(episode_data['latencies'], self.alpha)
                
                self.experience_buffer.append(episode_data)
                
                # Limit buffer size
                if len(self.experience_buffer) > self.max_buffer_size:
                    self.experience_buffer.pop(0)
                
                # Update policy if enough experience
                if len(self.experience_buffer) >= self.batch_size:
                    self.update_policy(self.experience_buffer)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'stgt_encoder_state_dict': self.stgt_encoder.state_dict(),
            'risk_critic_state_dict': self.risk_critic.state_dict(),
            'policy_head_state_dict': self.policy_head.state_dict(),
            'psi': self.psi,
            'config': {
                'feature_dim': self.feature_dim,
                'hidden_dim': self.hidden_dim,
                'k_neighbors': self.k_neighbors,
                'alpha': self.alpha,
                'lambda1': self.lambda1,
                'lambda2': self.lambda2,
                'l_max': self.l_max
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        
        self.stgt_encoder.load_state_dict(checkpoint['stgt_encoder_state_dict'])
        self.risk_critic.load_state_dict(checkpoint['risk_critic_state_dict'])
        self.policy_head.load_state_dict(checkpoint['policy_head_state_dict'])
        self.psi = checkpoint['psi'] 