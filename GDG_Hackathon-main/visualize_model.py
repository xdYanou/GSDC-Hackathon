import os
import torch
import pygame
from Environment import Env
from bots.dqn import MyBot
from bots.PPO import MyBot as MyBotPPO
from components.character import Character

def run_visualization_game(env, players, bots):
    """Runs a single visualization game"""
    env.set_players_bots_objects(players, bots)  # ✅ Set players before reset
    env.reset(randomize_objects=True)
    env.steps = 0

    for bot in bots:
        bot.reset_for_new_episode()

    env.last_damage_tracker = {player.username: 0 for player in players}

    game_stats = {
        player.username: {
            "kills": 0,
            "damage_dealt": 0,
            "survival_time": 0
        } for player in players
    }

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return None

        pygame.event.pump()
        finished, info = env.step(debugging=False)

        for player in players:
            if player.username in info["players_info"]:
                player_info = info["players_info"][player.username]
                game_stats[player.username]["kills"] = player_info.get("kills", 0)
                game_stats[player.username]["damage_dealt"] = player_info.get("damage_dealt", 0)
                if player_info.get("alive", False):
                    game_stats[player.username]["survival_time"] += 1

        if finished:
            return game_stats

def main():
    # Configuration
    world_width = 1280
    world_height = 1280
    display_width = 800
    display_height = 800

    # Updated model paths (both are DQN)
    model_paths = [
        r"C:\Users\yanit\OneDrive\Bureau\DS Portfolio\Bot Fighting Hackathon\training_runs\20250331_235035\models\bot_model_0_epoch_66.pth",
        r"C:\Users\yanit\OneDrive\Bureau\DS Portfolio\Bot Fighting Hackathon\training_runs\20250331_235035\models\bot_model_0_epoch_21.pth"
    ]

    # Create environment
    env = Env(
        training=False,
        use_game_ui=True,
        world_width=world_width,
        world_height=world_height,
        display_width=display_width,
        display_height=display_height,
        n_of_obstacles=0,
        frame_skip=1
    )

    world_bounds = env.get_world_bounds()

    # Setup players
    players = [
        Character(
            starting_pos=(world_bounds[2] - 100, world_bounds[3] - 100),
            screen=env.world_surface,
            boundaries=world_bounds,
            username="Ninja"
        ),
        Character(
            starting_pos=(world_bounds[0] + 10, world_bounds[1] + 10),
            screen=env.world_surface,
            boundaries=world_bounds,
            username="Faze Jarvis"
        ),
    ]

    # Setup bots with loaded DQN models
    bots = []
    for idx, model_path in enumerate(model_paths):
        bot = MyBot(action_size=56)  # Adjust if action size is different
        try:
            checkpoint = torch.load(model_path)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                bot.model.load_state_dict(checkpoint['model_state_dict'])
                if 'epsilon' in checkpoint:
                    bot.epsilon = checkpoint['epsilon']
            else:
                bot.model.load_state_dict(checkpoint)

            bot.model.eval()
            bot.epsilon = 0.0  # Disable exploration for visualization
            print(f"Successfully loaded DQN model for bot {idx}")
        except Exception as e:
            print(f"Error loading model for bot {idx}: {e}")
            return

        bots.append(bot)

    # Run games
    num_games = 5
    all_stats = []

    print("\nStarting DQN vs DQN visualization...\n")

    try:
        for game in range(num_games):
            print(f"\nGame {game + 1}/{num_games}")
            game_stats = run_visualization_game(env, players, bots)

            if game_stats is None:
                break

            all_stats.append(game_stats)

            print("\nGame Results:")
            for player in players:
                stats = game_stats[player.username]
                print(f"{player.username}:")
                print(f"  Kills: {stats['kills']}")
                print(f"  Damage Dealt: {stats['damage_dealt']}")
                print(f"  Survival Time: {stats['survival_time']}")

            pygame.time.wait(2000)

        if all_stats:
            print("\nOverall Statistics:")
            for player in players:
                total_kills = sum(game[player.username]["kills"] for game in all_stats)
                total_damage = sum(game[player.username]["damage_dealt"] for game in all_stats)
                avg_survival = sum(game[player.username]["survival_time"] for game in all_stats) / len(all_stats)

                print(f"\n{player.username}:")
                print(f"  Average Kills: {total_kills / len(all_stats):.2f}")
                print(f"  Average Damage: {total_damage / len(all_stats):.2f}")
                print(f"  Average Survival Time: {avg_survival:.2f}")

    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main() 