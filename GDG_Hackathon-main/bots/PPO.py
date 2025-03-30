import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
import torch.distributions as dist

class BackboneNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5):
        super(BackboneNetwork, self).__init__()
        self.input_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, state_dict):
        # Combine all inputs into a single tensor.
        location = state_dict['location']
        status = state_dict['status']
        rays = state_dict['rays']
        relative_pos = state_dict.get('relative_pos', torch.zeros_like(location))
        time_features = state_dict.get('time_features', torch.zeros((location.shape[0], 2), device=location.device))
        combined = torch.cat([location, status, rays, relative_pos, time_features], dim=1)
        return self.input_net(combined)

class ActorCritic(nn.Module):
    def __init__(self, backbone, action_dim):
        super(ActorCritic, self).__init__()
        self.backbone = backbone
        # The actor head produces logits over actions.
        self.actor = nn.Linear(64, action_dim)
        # The critic head outputs a scalar state-value.
        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        features = self.backbone(state)
        # Compute action probabilities via softmax.
        action_probs = torch.softmax(self.actor(features), dim=-1)
        state_value = self.critic(features)
        return action_probs, state_value

class MyBot():
    def __init__(self, action_size=16):
        self.action_size = action_size
        # Use a trajectory buffer for on-policy updates.
        self.trajectory = []
        self.epsilon = 0.0  # Placeholder; PPO does not use epsilon-greedy.
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.batch_size = 64  # minibatch size for PPO update.
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else
                                   "cpu")
        print(f"Using device: {self.device}")

        backbone_network = BackboneNetwork(input_dim=38, output_dim=64).to(self.device)
        self.model = ActorCritic(backbone=backbone_network, action_dim=action_size).to(self.device)
        # Target network (optional in PPO).
        self.target_model = ActorCritic(backbone=backbone_network, action_dim=action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Learning rate scheduler for hyperparameter optimization.
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)
        
        self.entropy_coef = 0.01

        # For storing data from the last step.
        self.last_state = None
        self.last_action = None
        self.last_log_prob = None
        self.last_value = None

        # Time tracking.
        self.time_since_last_shot = 0
        self.time_alive = 0

    def normalize_state(self, info):
        """Normalize state values to improve learning stability."""
        try:
            state = {
                'location': torch.tensor([
                    info['location'][0] / 1280.0,
                    info['location'][1] / 1280.0
                ], dtype=torch.float32),
                'status': torch.tensor([
                    info['rotation'] / 360.0,
                    info['current_ammo'] / 30.0
                ], dtype=torch.float32),
                'rays': []
            }
            ray_data = []
            for ray in info.get('rays', []):
                if isinstance(ray, list) and len(ray) == 3:
                    start_pos, end_pos = ray[0]
                    distance = ray[1] if ray[1] is not None else 1500
                    hit_type = ray[2]
                    ray_data.extend([
                        start_pos[0] / 1280.0,
                        start_pos[1] / 1280.0,
                        end_pos[0] / 1280.0,
                        end_pos[1] / 1280.0,
                        distance / 1500.0,
                        1.0 if hit_type == "player" else 0.5 if hit_type == "object" else 0.0
                    ])
            while len(ray_data) < 30:
                ray_data.extend([0.0] * 6)
            state['rays'] = torch.tensor(ray_data[:30], dtype=torch.float32)
            if 'closest_opponent' in info:
                opponent_pos = info['closest_opponent']
                rel_x = (opponent_pos[0] - info['location'][0]) / 1280.0
                rel_y = (opponent_pos[1] - info['location'][1]) / 1280.0
                state['relative_pos'] = torch.tensor([rel_x, rel_y], dtype=torch.float32)
            else:
                state['relative_pos'] = torch.tensor([0.0, 0.0], dtype=torch.float32)
            state['time_features'] = torch.tensor([
                self.time_since_last_shot / 100.0,
                self.time_alive / 2400.0
            ], dtype=torch.float32)
            self.time_alive += 1
            if info.get('shot_fired', False):
                self.time_since_last_shot = 0
            else:
                self.time_since_last_shot += 1
            return state
        except Exception as e:
            print(f"Error in normalize_state: {e}")
            print(f"Info received: {info}")
            raise

    def action_to_dict(self, action):
        """Enhanced action space with more granular rotation."""
        movement_directions = ["forward", "right", "down", "left"]
        rotation_angles = [-30, -5, -1, 0, 1, 5, 30]
        commands = {
            "forward": False,
            "right": False,
            "down": False,
            "left": False,
            "rotate": 0,
            "shoot": False
        }
        if action < 28:
            shoot = False
            local_action = action
        else:
            shoot = True
            local_action = action - 28
        movement_idx = local_action // 7
        angle_idx = local_action % 7
        direction = movement_directions[movement_idx]
        commands[direction] = True
        commands["rotate"] = rotation_angles[angle_idx]
        commands["shoot"] = shoot
        return commands

    def act(self, info):
        """
        Always sample from the policy distribution.
        Record the log probability and state-value for the PPO update.
        """
        try:
            state = self.normalize_state(info)
            state_tensors = {k: v.unsqueeze(0).to(self.device) for k, v in state.items()}
            action_probs, value = self.model(state_tensors)
            distribution = torch.distributions.Categorical(action_probs)
            action = distribution.sample().item()
            log_prob = distribution.log_prob(torch.tensor(action).to(self.device))
            
            # Store information for the trajectory.
            self.last_state = state
            self.last_action = action
            self.last_log_prob = log_prob
            self.last_value = value.item()
            return self.action_to_dict(action)
        except Exception as e:
            print(f"Error in act: {e}")
            return {"forward": False, "right": False, "down": False, "left": False, "rotate": 0, "shoot": False}

    def remember(self, reward, next_info, done):
        """
        Accumulate the on-policy trajectory.
        When an episode finishes (done is True), perform the PPO update.
        """
        try:
            next_state = self.normalize_state(next_info)
            self.trajectory.append((self.last_state, self.last_action, reward, self.last_log_prob, self.last_value, done, next_state))
            if done:
                self.finish_episode()
        except Exception as e:
            print(f"Error in remember: {e}")

    def finish_episode(self):
        """
        Compute returns and advantages from the trajectory,
        then perform several PPO update epochs over the collected data.
        """
        rewards = [t[2] for t in self.trajectory]
        values = [t[4] for t in self.trajectory]
        dones = [t[5] for t in self.trajectory]
        
        last_value = 0 if dones[-1] else self.last_value
        
        returns = []
        advantages = []
        gae = 0
        gamma = self.gamma
        lam = 0.95  # GAE lambda hyperparameter.
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[i]
            else:
                next_value = values[i+1]
                next_non_terminal = 1.0 - dones[i]
            delta = rewards[i] + gamma * next_value * next_non_terminal - values[i]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        state_keys = self.trajectory[0][0].keys()
        states = {key: torch.stack([t[0][key] for t in self.trajectory]).to(self.device)
                  for key in state_keys}
        actions = torch.tensor([t[1] for t in self.trajectory], dtype=torch.long).to(self.device)
        old_log_probs = torch.stack([t[3] for t in self.trajectory]).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        epochs = 4
        dataset_size = len(self.trajectory)
        batch_size = min(self.batch_size, dataset_size)
        for epoch in range(epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                batch_states = {key: states[key][batch_idx] for key in states}
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                
                action_probs, state_values = self.model(batch_states)
                dist_batch = torch.distributions.Categorical(action_probs)
                new_log_probs = dist_batch.log_prob(batch_actions)
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clip_eps = 0.2
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(state_values.squeeze(), batch_returns)
                entropy = dist_batch.entropy().mean()
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
        self.trajectory = []
        self.target_model.load_state_dict(self.model.state_dict())

    def reset_for_new_episode(self):
        self.time_alive = 0
        self.time_since_last_shot = 0

    def get_hyperparameters(self):
        return {
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "model_input_dim": 38,
            "action_size": self.action_size,
            "epsilon_decay": getattr(self, "epsilon_decay", None)
        }

    def save_to_dict(self):
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': self.get_hyperparameters(),
        }

    def load_from_dict(self, checkpoint_dict, map_location=None):
        if map_location is None:
            map_location = self.device
        self.model.load_state_dict(checkpoint_dict['model_state_dict'])
        try:
            self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            if map_location != 'cpu':
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(map_location)
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
        self.device = torch.device(map_location) if isinstance(map_location, str) else map_location
        self.model = self.model.to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model = self.target_model.to(self.device)
