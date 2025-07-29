import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

from controllers.zone_managers.STGT.stgt_zone_manager import STGTZoneManager, STGTEncoder, RiskCritic
from config import Config


class STGTTrainer:
    """Trainer for Spatio-Temporal Graph Transformer models"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'feature_dim': Config.STGTConfig.FEATURE_DIM,
            'hidden_dim': Config.STGTConfig.HIDDEN_DIM,
            'num_layers': Config.STGTConfig.NUM_LAYERS,
            'num_heads': Config.STGTConfig.NUM_HEADS,
            'd_ff': Config.STGTConfig.D_FF,
            'dropout': Config.STGTConfig.DROPOUT,
            'k_neighbors': Config.STGTConfig.K_NEIGHBORS,
            'alpha': Config.STGTConfig.ALPHA,
            'lambda1': Config.STGTConfig.LAMBDA1,
            'lambda2': Config.STGTConfig.LAMBDA2,
            'l_max': Config.STGTConfig.L_MAX,
            'batch_size': Config.STGTConfig.BATCH_SIZE,
            'max_buffer_size': Config.STGTConfig.MAX_BUFFER_SIZE,
            'clip_epsilon': Config.STGTConfig.CLIP_EPSILON,
            'policy_lr': Config.STGTConfig.POLICY_LR,
            'critic_lr': Config.STGTConfig.CRITIC_LR,
            'dual_lr': Config.STGTConfig.DUAL_LR,
            'num_epochs': 1000,
            'eval_interval': 50,
            'save_interval': 100
        }

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA GPU")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Silicon MPS GPU")
        else:
            raise RuntimeError("No supported GPU backend (CUDA or MPS) is available.")

        print(f"Using device: {self.device}")
        
        # Training history
        self.training_history = {
            'policy_loss': [],
            'critic_loss': [],
            'dual_loss': [],
            'total_loss': [],
            'cvar_values': [],
            'spectral_efficiency': [],
            'bandwidth_fairness': [],
            'completion_rate': [],
            'miss_rate': []
        }
        
        # Create output directory
        self.output_dir = f"./results/stgt_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def create_model(self, zone) -> STGTZoneManager:
        """Create a new STGT model"""
        model = STGTZoneManager(
            zone=zone,
            feature_dim=self.config['feature_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            k_neighbors=self.config['k_neighbors'],
            alpha=self.config['alpha'],
            lambda1=self.config['lambda1'],
            lambda2=self.config['lambda2'],
            l_max=self.config['l_max']
        )
        
        # Move to device
        model.stgt_encoder = model.stgt_encoder.to(self.device)
        model.risk_critic = model.risk_critic.to(self.device)
        model.policy_head = model.policy_head.to(self.device)
        
        return model
    
    def generate_synthetic_data(self, num_episodes: int = 100) -> List[Dict]:
        """Generate synthetic training data"""
        episodes = []
        
        for episode in range(num_episodes):
            # Generate random episode data
            num_tasks = np.random.randint(10, 50)
            latencies = np.random.exponential(scale=20, size=num_tasks)
            sinr_values = np.random.uniform(5, 25, size=num_tasks)
            bandwidth_allocations = np.random.uniform(50, 200, size=num_tasks)
            
            # Create synthetic snapshots
            snapshots = []
            for t in range(4):  # 4 time steps
                snapshot = {
                    'nodes': {},
                    'edges': [],
                    'timestamp': t * 10
                }
                
                # Add random nodes
                for i in range(np.random.randint(5, 15)):
                    node_id = f"node_{i}_{t}"
                    snapshot['nodes'][node_id] = type('NodeFeatures', (), {
                        'queue_length': np.random.uniform(0, 10),
                        'sinr': np.random.uniform(5, 25),
                        'bandwidth_budget': np.random.uniform(50, 200),
                        'position': (np.random.uniform(0, 1000), np.random.uniform(0, 1000)),
                        'time_embedding': np.sin(2 * np.pi * t / 4)
                    })()
                
                snapshots.append(snapshot)
            
            episode_data = {
                'latencies': latencies.tolist(),
                'sinr_values': sinr_values.tolist(),
                'bandwidth_allocations': bandwidth_allocations.tolist(),
                'snapshots': snapshots,
                'state': torch.randn(self.config['hidden_dim']),
                'action': torch.randn(1),
                'log_prob': torch.randn(1),
                'reward': np.random.uniform(-10, 10),
                'cvar': np.percentile(latencies, 95)
            }
            
            episodes.append(episode_data)
        
        return episodes
    
    def train_model(self, model: STGTZoneManager, training_data: List[Dict]):
        """Train the STGT model"""
        print("Starting STGT model training...")
        
        for epoch in range(self.config['num_epochs']):
            # Sample batch
            if len(training_data) < self.config['batch_size']:
                continue
                
            batch_indices = np.random.choice(len(training_data), self.config['batch_size'], replace=False)
            batch = [training_data[i] for i in batch_indices]
            
            # Extract batch data
            states = torch.stack([item['state'] for item in batch]).to(self.device)
            actions = torch.stack([item['action'] for item in batch]).to(self.device)
            old_log_probs = torch.stack([item['log_prob'] for item in batch]).to(self.device)
            rewards = torch.tensor([item['reward'] for item in batch], dtype=torch.float32).to(self.device)
            cvar_targets = torch.tensor([item['cvar'] for item in batch], dtype=torch.float32).to(self.device)
            
            # Compute advantages
            advantages = rewards - rewards.mean()
            advantages = advantages / (advantages.std() + 1e-8)
            
            # Policy update
            policy_losses = []
            for _ in range(10):  # Multiple epochs
                # # Forward pass through STGT encoder
                # snapshots_batch = [item['snapshots'] for item in batch]
                # node_embeddings, global_context = model.stgt_encoder(snapshots_batch)
                snapshots = batch[0]['snapshots']
                node_embeddings, global_context = model.stgt_encoder(snapshots)
                
                # Policy head forward pass
                action_probs = model.policy_head(global_context.unsqueeze(0)).squeeze(0)
                
                # Calculate log probabilities
                log_probs = torch.log(action_probs + 1e-8)
                
                # PPO clipped ratio
                ratio = torch.exp(log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.config['clip_epsilon'], 1 + self.config['clip_epsilon'])
                
                # Policy loss
                policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
                policy_losses.append(policy_loss.item())
                
                # Update policy
                model.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy_head.parameters(), max_norm=1.0)
                model.policy_optimizer.step()
            
            # Risk critic update
            risk_estimates = model.risk_critic(states)
            critic_loss = nn.MSELoss()(risk_estimates.squeeze(), cvar_targets)
            
            model.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.risk_critic.parameters(), max_norm=1.0)
            model.critic_optimizer.step()
            
            # Dual variable update
            cvar_violations = torch.clamp(cvar_targets - self.config['l_max'], min=0)
            dual_loss = -model.psi * cvar_violations.mean()
            
            model.dual_optimizer.zero_grad()
            dual_loss.backward()
            model.dual_optimizer.step()
            
            # Ensure psi is non-negative
            with torch.no_grad():
                model.psi.clamp_(min=0.0)
            
            # Record training history
            self.training_history['policy_loss'].append(np.mean(policy_losses))
            self.training_history['critic_loss'].append(critic_loss.item())
            self.training_history['dual_loss'].append(dual_loss.item())
            self.training_history['total_loss'].append(
                np.mean(policy_losses) + critic_loss.item() + dual_loss.item()
            )
            
            # Evaluation
            if epoch % self.config['eval_interval'] == 0:
                self.evaluate_model(model, training_data)
            
            # Save model
            if epoch % self.config['save_interval'] == 0:
                self.save_model(model, epoch)
            
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.config['num_epochs']}: "
                      f"Policy Loss: {np.mean(policy_losses):.4f}, "
                      f"Critic Loss: {critic_loss.item():.4f}, "
                      f"Dual Loss: {dual_loss.item():.4f}")
        
        print("Training completed!")
        self.save_training_history()
        self.plot_training_curves()
    
    def evaluate_model(self, model: STGTZoneManager, test_data: List[Dict]):
        """Evaluate the trained model"""
        model.eval()
        
        with torch.no_grad():
            # Calculate metrics on test data
            cvar_values = []
            spectral_efficiencies = []
            bandwidth_fairnesses = []
            
            for episode in test_data[:50]:  # Use first 50 episodes for evaluation
                # Calculate CVaR
                latencies = episode['latencies']
                cvar = model.calculate_cvar(latencies, self.config['alpha'])
                cvar_values.append(cvar)
                
                # Calculate spectral efficiency
                sinr_values = episode['sinr_values']
                spec_eff = model.calculate_spectral_efficiency(sinr_values)
                spectral_efficiencies.append(spec_eff)
                
                # Calculate bandwidth fairness
                bandwidth_allocations = episode['bandwidth_allocations']
                fairness = model.calculate_bandwidth_fairness(bandwidth_allocations)
                bandwidth_fairnesses.append(fairness)
            
            # Record evaluation metrics
            self.training_history['cvar_values'].append(np.mean(cvar_values))
            self.training_history['spectral_efficiency'].append(np.mean(spectral_efficiencies))
            self.training_history['bandwidth_fairness'].append(np.mean(bandwidth_fairnesses))
        
        model.train()
    
    def save_model(self, model: STGTZoneManager, epoch: int):
        """Save the trained model"""
        model_path = os.path.join(self.output_dir, f"stgt_model_epoch_{epoch}.pth")
        model.save_model(model_path)
        print(f"Model saved to {model_path}")
    
    def save_training_history(self):
        """Save training history to JSON"""
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_dict = {}
            for key, value in self.training_history.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    history_dict[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
                else:
                    history_dict[key] = value
            json.dump(history_dict, f, indent=2)
        print(f"Training history saved to {history_path}")
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Policy loss
        axes[0, 0].plot(self.training_history['policy_loss'])
        axes[0, 0].set_title('Policy Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Critic loss
        axes[0, 1].plot(self.training_history['critic_loss'])
        axes[0, 1].set_title('Critic Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # CVaR values
        if self.training_history['cvar_values']:
            axes[1, 0].plot(self.training_history['cvar_values'])
            axes[1, 0].set_title('CVaR Values')
            axes[1, 0].set_xlabel('Evaluation Step')
            axes[1, 0].set_ylabel('CVaR')
            axes[1, 0].grid(True)
        
        # Spectral efficiency
        if self.training_history['spectral_efficiency']:
            axes[1, 1].plot(self.training_history['spectral_efficiency'])
            axes[1, 1].set_title('Spectral Efficiency')
            axes[1, 1].set_xlabel('Evaluation Step')
            axes[1, 1].set_ylabel('Efficiency')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training curves saved to {plot_path}")
    
    def hyperparameter_tuning(self, param_ranges: Dict):
        """Perform hyperparameter tuning"""
        print("Starting hyperparameter tuning...")
        
        best_config = None
        best_performance = float('inf')
        
        # Grid search over hyperparameters
        for hidden_dim in param_ranges.get('hidden_dim', [64, 128, 256]):
            for num_layers in param_ranges.get('num_layers', [1, 2, 3]):
                for k_neighbors in param_ranges.get('k_neighbors', [5, 10, 15]):
                    for alpha in param_ranges.get('alpha', [0.9, 0.95, 0.99]):
                        # Update config
                        self.config.update({
                            'hidden_dim': hidden_dim,
                            'num_layers': num_layers,
                            'k_neighbors': k_neighbors,
                            'alpha': alpha
                        })
                        
                        # Create and train model
                        print(f"Testing config: hidden_dim={hidden_dim}, num_layers={num_layers}, "
                              f"k_neighbors={k_neighbors}, alpha={alpha}")
                        
                        # Generate synthetic data
                        training_data = self.generate_synthetic_data(num_episodes=50)
                        
                        # Create dummy zone for testing
                        from models.zone import Zone
                        dummy_zone = Zone(0, 0, 1000, 1000)
                        
                        # Create and train model
                        model = self.create_model(dummy_zone)
                        self.train_model(model, training_data)
                        
                        # Evaluate performance
                        performance = np.mean(self.training_history['total_loss'][-10:])  # Last 10 epochs
                        
                        if performance < best_performance:
                            best_performance = performance
                            best_config = self.config.copy()
                        
                        print(f"Performance: {performance:.4f}")
        
        print(f"Best config: {best_config}")
        print(f"Best performance: {best_performance:.4f}")
        
        return best_config


def main():
    """Main function for training STGT models"""
    # Initialize trainer
    trainer = STGTTrainer()
    
    # Generate synthetic training data
    print("Generating synthetic training data...")
    training_data = trainer.generate_synthetic_data(num_episodes=200)
    
    # Create dummy zone for training
    from models.zone import Zone
    dummy_zone = Zone(0, 0, 1000, 1000)
    
    # Create and train model
    model = trainer.create_model(dummy_zone)
    trainer.train_model(model, training_data)
    
    # Optional: Perform hyperparameter tuning
    # param_ranges = {
    #     'hidden_dim': [64, 128, 256],
    #     'num_layers': [1, 2, 3],
    #     'k_neighbors': [5, 10, 15],
    #     'alpha': [0.9, 0.95, 0.99]
    # }
    # best_config = trainer.hyperparameter_tuning(param_ranges)


if __name__ == "__main__":
    main() 