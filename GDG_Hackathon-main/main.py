import os
import json
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import pygame
from Environment import Env
from bots.PPO import MyBot
from components.character import Character

def run_game(env, players, bots):
    """Runs the game in display mode for human viewing"""
    env.reset(randomize_objects=True)
    env.steps = 0

    for bot in bots:
        bot.reset_for_new_episode()

    env.set_players_bots_objects(players, bots)
    env.last_damage_tracker = {player.username: 0 for player in players}

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return

        pygame.event.pump()
        finished, info = env.step(debugging=False)
        for player, bot in zip(players, bots):
            reward = env.calculate_reward(info, player.username)
            next_info = player.get_info()
            if 'closest_opponent' not in next_info:
                next_info['closest_opponent'] = env.find_closest_opponent(player)
            bot.remember(reward, next_info, finished)

        if finished:
            print("Game finished!")
            pygame.time.delay(3000)
            break

def train_single_episode(env, players, bots, config, current_stage):
    """Trains a single episode in one environment"""
    env.reset(randomize_objects=True)
    env.steps = 0

    for bot in bots:
        bot.reset_for_new_episode()

    episode_metrics = {
        "rewards": {player.username: 0 for player in players},
        "kills": {player.username: 0 for player in players},
        "damage_dealt": {player.username: 0 for player in players},
        "survival_time": {player.username: 0 for player in players}
    }

    env.last_damage_tracker = {player.username: 0 for player in players}

    while env.steps < config["tick_limit"]:
        finished, info = env.step(debugging=False)
        for player, bot in zip(players, bots):
            reward = env.calculate_reward(info, player.username)
            reward *= 1.0 - (current_stage * 0.1)
            episode_metrics["rewards"][player.username] += reward

            player_info = info["players_info"][player.username]
            episode_metrics["kills"][player.username] = player_info.get("kills", 0)

            current_damage = player_info.get("damage_dealt", 0)
            damage_delta = current_damage - env.last_damage_tracker.get(player.username, 0)
            episode_metrics["damage_dealt"][player.username] += max(0, damage_delta)
            env.last_damage_tracker[player.username] = current_damage

            if player_info.get("alive", False):
                episode_metrics["survival_time"][player.username] += 1

            next_info = player.get_info()
            if 'closest_opponent' not in next_info:
                next_info['closest_opponent'] = env.find_closest_opponent(player)
            bot.remember(reward, next_info, finished)
        if finished:
            break
    return episode_metrics

def main():
    "--- KEEP THESE VALUES UNCHANGED ---"
    world_width = 1280
    world_height = 1280
    display_width = 800
    display_height = 800
    "---"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"training_runs/{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f"{run_dir}/models", exist_ok=True)
    os.makedirs(f"{run_dir}/plots", exist_ok=True)

    config = {
        "frame_skip": 4,
        "tick_limit": 2400,
        "num_epochs": 30,
        "action_size": 56,
        "hyperparameters": {
            "double_dqn": True,
            "learning_rate": 0.0001,
            "batch_size": 64,
            "gamma": 0.99,
            "epsilon_decay": 0.9999
        }
    }

    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    curriculum_stages = [
        {"n_obstacles": 10, "duration": 100},
        {"n_obstacles": 15, "duration": 200},
        {"n_obstacles": 20, "duration": 300}
    ]

    training_mode = True

    env = Env(
        training=training_mode,
        use_game_ui=False,
        world_width=world_width,
        world_height=world_height,
        display_width=display_width,
        display_height=display_height,
        n_of_obstacles=curriculum_stages[0]["n_obstacles"],
        frame_skip=config["frame_skip"]
    )

    world_bounds = env.get_world_bounds()

    players = [
        Character(starting_pos=(world_bounds[2] - 100, world_bounds[3] - 100),
                  screen=env.world_surface, boundaries=world_bounds, username="Kaaris"),
        Character(starting_pos=(world_bounds[0] + 10, world_bounds[1] + 10),
                  screen=env.world_surface, boundaries=world_bounds, username="Booba")
    ]

    bots = []
    for _ in players:
        bot = MyBot(action_size=config["action_size"])
        bot.use_double_dqn = config["hyperparameters"]["double_dqn"]
        bot.learning_rate = config["hyperparameters"]["learning_rate"]
        bot.batch_size = config["hyperparameters"]["batch_size"]
        bot.gamma = config["hyperparameters"]["gamma"]
        bot.epsilon_decay = config["hyperparameters"]["epsilon_decay"]
        bot.optimizer = torch.optim.Adam(bot.model.parameters(), lr=bot.learning_rate)
        # Recreate the scheduler using the updated optimizer.
        bot.scheduler = torch.optim.lr_scheduler.ExponentialLR(bot.optimizer, gamma=0.995)
        bots.append(bot)

    env.set_players_bots_objects(players, bots)

    if training_mode:
        all_rewards = {player.username: [] for player in players}

        for epoch in range(config["num_epochs"]):
            print(f"Epoch {epoch + 1}/{config['num_epochs']}")
            total_epochs = 0
            for i, stage in enumerate(curriculum_stages):
                total_epochs += stage["duration"]
                if epoch < total_epochs:
                    current_stage = i
                    break

            env.n_of_obstacles = curriculum_stages[current_stage]["n_obstacles"]
            metrics = train_single_episode(env, players, bots, config, current_stage)

            for idx, bot in enumerate(bots):
                torch.save(bot.model.state_dict(), f"{run_dir}/models/bot_modelPPO_{idx}_epoch_{epoch + 1}.pth")
            for player in players:
                username = player.username
                all_rewards[username].append(metrics["rewards"][username])
                print(f"{username} - Reward: {metrics['rewards'][username]:.2f}, "
                      f"Kills: {metrics['kills'][username]}, "
                      f"Damage: {metrics['damage_dealt'][username]}, "
                      f"Survival: {metrics['survival_time'][username]}")
            
            # Step each bot's learning rate scheduler and log new learning rates.
            for idx, bot in enumerate(bots):
                bot.scheduler.step()
                current_lr = bot.optimizer.param_groups[0]['lr']
                print(f"Bot {idx} new learning rate: {current_lr:.6f}")

        for username, rewards in all_rewards.items():
            plt.plot(rewards, label=username)
        plt.title("Rewards per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{run_dir}/plots/rewards_plot.png")
        plt.close()

    else:
        run_game(env, players, bots)

if __name__ == "__main__":
    main()
