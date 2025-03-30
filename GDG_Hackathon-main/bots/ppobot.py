import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
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
        # Combine all inputs into a single tensor
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
        # Compute action probabilities via softmax
        action_probs = torch.softmax(self.actor(features), dim=-1)
        state_value = self.critic(features)
        return action_probs, state_value

class MyBot():
    def __init__(self, action_size=16):
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.priority_memory = deque(maxlen=50000)
        self.priority_probabilities = deque(maxlen=50000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 1  # Decay can be adjusted as needed
        self.learning_rate = 0.0001
        self.batch_size = 64
        self.min_memory_size = 1000
        self.update_target_freq = 500
        self.train_freq = 4
        self.steps = 0
        self.use_double_dqn = True  # Still available but not used for actor-critic loss here
        self.reset_epsilon = True

        # Prioritized experience replay parameters
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.epsilon_pri = 0.01
        self.max_priority = 1.0

        # Curiosity parameters
        self.exploration_bonus = 0.1
        self.visited_positions = {}
        self.position_resolution = 50

        # Time tracking features
        self.time_since_last_shot = 0
        self.time_alive = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else
                                   "cpu")
        print(f"Using device: {self.device}")

        backbone_network = BackboneNetwork(input_dim=38, output_dim=64).to(self.device)
        self.model = ActorCritic(backbone=backbone_network, action_dim=action_size).to(self.device)
        # Create a separate target network instance with the same architecture.
        # (In actor-critic methods, sometimes a target network is omitted.)
        self.target_model = ActorCritic(backbone=backbone_network, action_dim=action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max',
                                                              factor=0.5, patience=10)
        self.last_state = None
        self.last_action = None
        self.training_started = False

        # Entropy coefficient to encourage exploration
        self.entropy_coef = 0.01

    def normalize_state(self, info):
        """Normalize state values to improve learning stability"""
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
        """Enhanced action space with more granular rotation"""
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
        try:
            state = self.normalize_state(info)
            state_tensors = {k: v.unsqueeze(0).to(self.device) for k, v in state.items()}
            # Use epsilon-greedy exploration: with probability epsilon take a random action.
            if random.random() <= self.epsilon:
                action = random.randrange(self.action_size)
            else:
                with torch.no_grad():
                    action_probs, _ = self.model(state_tensors)
                    # Sample an action from the probability distribution
                    distribution = torch.distributions.Categorical(action_probs)
                    action = distribution.sample().item()
            self.last_state = state
            self.last_action = action
            return self.action_to_dict(action)
        except Exception as e:
            print(f"Error in act: {e}")
            return {"forward": False, "right": False, "down": False, "left": False, "rotate": 0, "shoot": False}

    def remember(self, reward, next_info, done):
        try:
            next_state = self.normalize_state(next_info)
            pos_x = int(next_state['location'][0].item() * self.position_resolution)
            pos_y = int(next_state['location'][1].item() * self.position_resolution)
            grid_pos = (pos_x, pos_y)
            if grid_pos in self.visited_positions:
                self.visited_positions[grid_pos] += 1
                visit_count = self.visited_positions[grid_pos]
                exploration_bonus = self.exploration_bonus / math.sqrt(visit_count)
            else:
                self.visited_positions[grid_pos] = 1
                exploration_bonus = self.exploration_bonus
            reward += exploration_bonus
            self.memory.append((self.last_state, self.last_action, reward, next_state, done))
            self.priority_memory.append((self.last_state, self.last_action, reward, next_state, done))
            self.priority_probabilities.append(self.max_priority)
            if len(self.memory) >= self.min_memory_size and not self.training_started:
                print(f"Starting training with {len(self.memory)} samples in memory")
                self.training_started = True
            self.steps += 1
            if self.training_started and self.steps % self.train_freq == 0:
                self.prioritized_replay()
                if self.steps % 1000 == 0:
                    print(f"Step {self.steps}, epsilon: {self.epsilon:.4f}")
            if self.steps > 0 and self.steps % self.update_target_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                print(f"Updated target network at step {self.steps}")
            if done:
                self.time_alive = 0
        except Exception as e:
            print(f"Error in remember: {e}")

    def prioritized_replay(self):
        """Actor-Critic update using prioritized replay samples"""
        if len(self.priority_memory) < self.batch_size:
            return

        try:
            priorities = np.array(self.priority_probabilities)
            probs = priorities ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(len(self.priority_memory), self.batch_size, p=probs)
            batch = [self.priority_memory[idx] for idx in indices]
            self.beta = min(1.0, self.beta + self.beta_increment)
            weights = (len(self.priority_memory) * probs[indices]) ** (-self.beta)
            weights /= weights.max()
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

            # Prepare batch data
            states = {
                'location': torch.stack([t[0]['location'] for t in batch]).to(self.device),
                'status': torch.stack([t[0]['status'] for t in batch]).to(self.device),
                'rays': torch.stack([t[0]['rays'] for t in batch]).to(self.device),
                'relative_pos': torch.stack([t[0].get('relative_pos', torch.zeros(2)) for t in batch]).to(self.device),
                'time_features': torch.stack([t[0].get('time_features', torch.zeros(2)) for t in batch]).to(self.device)
            }
            next_states = {
                'location': torch.stack([t[3]['location'] for t in batch]).to(self.device),
                'status': torch.stack([t[3]['status'] for t in batch]).to(self.device),
                'rays': torch.stack([t[3]['rays'] for t in batch]).to(self.device),
                'relative_pos': torch.stack([t[3].get('relative_pos', torch.zeros(2)) for t in batch]).to(self.device),
                'time_features': torch.stack([t[3].get('time_features', torch.zeros(2)) for t in batch]).to(self.device)
            }
            actions = torch.tensor([t[1] for t in batch], dtype=torch.long).to(self.device)
            rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32).to(self.device)
            dones = torch.tensor([t[4] for t in batch], dtype=torch.float32).to(self.device)

            # Compute target values using the target network
            with torch.no_grad():
                _, next_values = self.target_model(next_states)  # shape [batch, 1]
                target_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_values

            # Get current estimates from the actor-critic model
            action_probs, state_values = self.model(states)
            # Get probabilities of the taken actions
            chosen_action_probs = action_probs.gather(1, actions.unsqueeze(1)) + 1e-8
            log_probs = torch.log(chosen_action_probs)
            # Compute advantage
            advantage = target_values - state_values

            # Actor loss (policy gradient loss)
            actor_loss = - (log_probs * advantage.detach() * weights.unsqueeze(1)).mean()
            # Critic loss (value function loss)
            critic_loss = F.mse_loss(state_values, target_values, reduction='none')
            critic_loss = (critic_loss * weights.unsqueeze(1)).mean()
            # Entropy bonus to encourage exploration
            entropy = - (action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
            loss = actor_loss + critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Update priorities based on the absolute TD error (advantage)
            td_errors = torch.abs(advantage).detach().cpu().numpy().squeeze()
            for idx, error in zip(indices, td_errors):
                self.priority_probabilities[idx] = error + self.epsilon_pri
                self.max_priority = max(self.max_priority, error + self.epsilon_pri)

            # Decay epsilon for exploration if desired
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        except Exception as e:
            print(f"Error in prioritized_replay: {e}")
            self.replay()

    def replay(self):
        """Fallback replay without prioritized weights (uses ones)"""
        if len(self.memory) < self.batch_size:
            return

        try:
            minibatch = random.sample(self.memory, self.batch_size)
            states = {
                'location': torch.stack([t[0]['location'] for t in minibatch]).to(self.device),
                'status': torch.stack([t[0]['status'] for t in minibatch]).to(self.device),
                'rays': torch.stack([t[0]['rays'] for t in minibatch]).to(self.device),
                'relative_pos': torch.stack([t[0].get('relative_pos', torch.zeros(2)) for t in minibatch]).to(self.device),
                'time_features': torch.stack([t[0].get('time_features', torch.zeros(2)) for t in minibatch]).to(self.device)
            }
            next_states = {
                'location': torch.stack([t[3]['location'] for t in minibatch]).to(self.device),
                'status': torch.stack([t[3]['status'] for t in minibatch]).to(self.device),
                'rays': torch.stack([t[3]['rays'] for t in minibatch]).to(self.device),
                'relative_pos': torch.stack([t[3].get('relative_pos', torch.zeros(2)) for t in minibatch]).to(self.device),
                'time_features': torch.stack([t[3].get('time_features', torch.zeros(2)) for t in minibatch]).to(self.device)
            }
            actions = torch.tensor([t[1] for t in minibatch], dtype=torch.long).to(self.device)
            rewards = torch.tensor([t[2] for t in minibatch], dtype=torch.float32).to(self.device)
            dones = torch.tensor([t[4] for t in minibatch], dtype=torch.float32).to(self.device)

            with torch.no_grad():
                _, next_values = self.target_model(next_states)
                target_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_values

            action_probs, state_values = self.model(states)
            chosen_action_probs = action_probs.gather(1, actions.unsqueeze(1)) + 1e-8
            log_probs = torch.log(chosen_action_probs)
            advantage = target_values - state_values

            actor_loss = - (log_probs * advantage.detach()).mean()
            critic_loss = F.mse_loss(state_values, target_values)
            entropy = - (action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
            loss = actor_loss + critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        except Exception as e:
            print(f"Error in replay: {e}")

    def reset_for_new_episode(self):
        self.time_alive = 0
        self.time_since_last_shot = 0
        if self.steps > 1000000:
            self.exploration_bonus = 0.05
        elif self.steps > 500000:
            self.exploration_bonus = 0.08

    def get_hyperparameters(self):
        return {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "alpha": self.alpha,
            "beta": self.beta,
            "use_double_dqn": self.use_double_dqn,
            "model_input_dim": 38,
            "action_size": self.action_size,
            "steps": self.steps,
        }

    def save_to_dict(self):
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
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
            print("Continuing with fresh optimizer but keeping model weights")
        if not self.reset_epsilon:
            self.epsilon = checkpoint_dict.get('epsilon', self.epsilon)
        self.steps = checkpoint_dict.get('steps', 0)
        self.device = torch.device(map_location) if isinstance(map_location, str) else map_location
        self.model = self.model.to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model = self.target_model.to(self.device)
