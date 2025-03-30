import os
import torch
import pygame
from Environment import Env
from components.character import Character

# Define a function to run the game and print summary metrics
def run_game(env, players, bots):
    """Runs the game in display mode for human viewing and prints summary metrics."""
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
            pygame.time.delay(3000)  # Wait 3 seconds to display the final state
            break

    # Print summary metrics after the match
    print("\n=== Match Summary ===")
    for player in players:
        info_summary = env.get_player_info(player.username)
        print(f"{player.username}: Kills = {info_summary.get('kills', 0)}, "
              f"Damage = {info_summary.get('damage_dealt', 0)}, "
              f"Alive = {info_summary.get('alive', False)}")

    kills = [env.get_player_info(p.username).get("kills", 0) for p in players]
    if kills[0] > kills[1]:
        print(f"🏆 Winner: {players[0].username}")
    elif kills[1] > kills[0]:
        print(f"🏆 Winner: {players[1].username}")
    else:
        print("🤝 It's a draw!")

if __name__ == "__main__":
    # --- Manual Bot Selection ---
    print("Select Bot Type for Bot1:")
    print("1. NewBot")
    print("2. OldBot")
    choice1 = input("Enter choice for Bot1 (1 or 2): ").strip()

    if choice1 == "1":
        from bots.ppobot import MyBot as Bot1Class
        # For NewBot, assume action_size remains 16 (or change accordingly)
        bot1 = Bot1Class()
    elif choice1 == "2":
        from bots.example_bot import MyBot as Bot1Class
        # Use the correct action size from training (e.g. 56)
        bot1 = Bot1Class(action_size=56)
    else:
        print("Invalid choice for Bot1.")
        exit(1)

    pth_path1 = input("Enter the full path to the .pth file for Bot1: ").strip()
    if not os.path.isfile(pth_path1):
        print("File for Bot1 not found.")
        exit(1)
    checkpoint1 = torch.load(pth_path1, map_location=bot1.device)
    bot1.load_from_dict(checkpoint1)

    print("\nSelect Bot Type for Bot2:")
    print("1. NewBot")
    print("2. OldBot")
    choice2 = input("Enter choice for Bot2 (1 or 2): ").strip()

    if choice2 == "1":
        from bots.ppobot import MyBot as Bot2Class
        bot2 = Bot2Class()
    elif choice2 == "2":
        from bots.example_bot import MyBot as Bot2Class
        # Adjust action_size if needed; if your old bot was trained with 56 actions, use that
        bot2 = Bot2Class(action_size=56)
    else:
        print("Invalid choice for Bot2.")
        exit(1)

    pth_path2 = input("Enter the full path to the .pth file for Bot2: ").strip()
    if not os.path.isfile(pth_path2):
        print("File for Bot2 not found.")
        exit(1)
    checkpoint2 = torch.load(pth_path2, map_location=bot2.device)
    bot2.load_from_dict(checkpoint2)

    from components.character import Character
    player1 = Character(username="Bot1")
    player2 = Character(username="Bot2")

    from Environment import Env
    env = Env(1280, 1280, 800, 800)

    run_game(env, [player1, player2], [bot1, bot2])

