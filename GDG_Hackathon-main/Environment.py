import math
import os
import pygame
from components.advanced_UI import game_UI
from components.world_gen import spawn_objects

class Env:
    def __init__(self, training=False, use_game_ui=True, world_width=1280, world_height=1280, display_width=640,
                 display_height=640, n_of_obstacles=10, frame_skip=4):
        pygame.init()

        self.training_mode = training

        # ONLY FOR DISPLAY
        # create display window with desired display dimensions
        self.display_width = display_width
        self.display_height = display_height
        # only create a window if not in training mode
        if not self.training_mode:
            self.screen = pygame.display.set_mode((display_width, display_height))
        else:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Disable actual video output
            pygame.display.set_mode((1, 1))  # Minimal display

            self.screen = pygame.Surface((display_width, display_height))

        # REAL WORLD DIMENSIONS
        # create an off-screen surface for the game world
        self.world_width = world_width
        self.world_height = world_height
        self.world_surface = pygame.Surface((world_width, world_height))

        self.clock = pygame.time.Clock()
        self.running = True

        self.use_advanced_UI = use_game_ui
        if self.use_advanced_UI:
            self.advanced_UI = game_UI(self.world_surface, self.world_width, self.world_height)

        if not self.training_mode and self.use_advanced_UI:
            self.advanced_UI.display_opening_screen()

        self.n_of_obstacles = n_of_obstacles
        self.min_obstacle_size = (50, 50)
        self.max_obstacle_size = (100, 100)

        # frame skip for training acceleration
        self.frame_skip = frame_skip if training else 1

        # INIT SOME VARIABLES
        self.OG_bots = None
        self.OG_players = None
        self.OG_obstacles = None

        self.bots = None
        self.players = None
        self.obstacles = None

        """REWARD VARIABLES"""
        self.last_positions = {}
        self.last_damage = {}
        self.last_kills = {}
        self.last_health = {}
        self.visited_areas = {}

        self.visited_areas.clear()
        self.last_positions.clear()
        self.last_health.clear()
        self.last_kills.clear()
        self.last_damage.clear()

        self.steps = 0

    def set_players_bots_objects(self, players, bots, obstacles=None):
        self.OG_players = players
        self.OG_bots = bots
        self.OG_obstacles = obstacles

        self.reset()

    def get_world_bounds(self):
        return (0, 0, self.world_width, self.world_height)

    def find_closest_opponent(self, player):
        """Find the position of the closest opponent for a given player"""
        closest_dist = float('inf')
        closest_pos = None

        for other in self.players:
            if other != player and other.alive:
                dist = math.dist(player.rect.center, other.rect.center)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_pos = other.rect.center

        # return default position if no opponents found
        if closest_pos is None:
            return player.rect.center

        return closest_pos

    def reset(self, randomize_objects=False, randomize_players=False):
        self.running = True
        if not self.training_mode:
            if not self.use_advanced_UI:
                self.screen.fill("green")
                pygame.display.flip()
                self.clock.tick(1)  # 1 frame per second for 1 second = 1 frame
            else:
                self.advanced_UI.display_reset_screen()

        else:
            self.screen.fill("green")

        self.last_positions = {}
        self.last_damage = {}
        self.last_kills = {}
        self.last_health = {}
        self.visited_areas = {}

        self.steps = 0

        # TODO: add variables for parameters
        if self.use_advanced_UI:
            self.obstacles = self.advanced_UI.obstacles
        else:
            if randomize_objects or self.OG_obstacles is None:
                self.OG_obstacles = spawn_objects(
                    (0, 0, self.world_width, self.world_height),
                    self.max_obstacle_size,
                    self.min_obstacle_size,
                    self.n_of_obstacles
                )
            self.obstacles = self.OG_obstacles

        self.players = self.OG_players.copy()
        self.bots = self.OG_bots
        if randomize_players:
            self.bots = self.bots.shuffle()
            for index in range(len(self.players)):
                self.players[index].related_bot = self.bots[index]  # ensuring bots change location

        else:
            for index in range(len(self.players)):
                self.players[index].related_bot = self.bots[index]

        for player in self.players:
            player.reset()
            temp = self.players.copy()
            temp.remove(player)
            player.players = temp  # Other players
            player.objects = self.obstacles

    def step(self, debugging=False):
        # only render if not in training mode
        if not self.training_mode:
            if self.use_advanced_UI:
                # use the background from game_UI
                self.world_surface.blit(self.advanced_UI.background, (0, 0))
            else:
                self.world_surface.fill("purple")

        # frame skipping for training acceleration
        skip_count = self.frame_skip if self.training_mode else 1

        # track if any frame resulted in game over
        game_over = False
        final_info = None

        # get actions once and reuse them for all skipped frames
        player_actions = {}
        if self.training_mode:
            for player in self.players:
                if player.alive:
                    # update player info with closest opponent data before action
                    player_info = player.get_info()
                    player_info['closest_opponent'] = self.find_closest_opponent(player)
                    player_actions[player.username] = player.related_bot.act(player_info)

        # process multiple frames if frame skipping is enabled
        for _ in range(skip_count):
            if game_over:
                break

            self.steps += 1

            players_info = {}
            alive_players = []

            for player in self.players:
                player.update_tick()

                # use stored actions if in training mode with frame skipping
                if self.training_mode and skip_count > 1:
                    actions = player_actions.get(player.username, {})
                else:
                    # update info with closest opponent before getting action
                    player_info = player.get_info()
                    player_info['closest_opponent'] = self.find_closest_opponent(player)
                    actions = player.related_bot.act(player_info)

                if player.alive:
                    alive_players.append(player)
                    player.reload()

                    # skip drawing in training mode for better performance
                    if not self.training_mode:
                        player.draw(self.world_surface)

                    if debugging:
                        print("Bot would like to do:", actions)
                    if actions.get("forward", False):
                        player.move_in_direction("forward")
                    if actions.get("right", False):
                        player.move_in_direction("right")
                    if actions.get("down", False):
                        player.move_in_direction("down")
                    if actions.get("left", False):
                        player.move_in_direction("left")
                    if actions.get("rotate", 0):
                        player.add_rotate(actions["rotate"])
                    if actions.get("shoot", False):
                        player.shoot()

                    if not self.training_mode:
                        # store position for trail
                        if not hasattr(player, 'previous_positions'):
                            player.previous_positions = []
                        player.previous_positions.append(player.rect.center)
                        if len(player.previous_positions) > 10:
                            player.previous_positions.pop(0)

                player_info = player.get_info()
                player_info["shot_fired"] = actions.get("shoot", False)
                player_info["closest_opponent"] = self.find_closest_opponent(player)
                players_info[player.username] = player_info

            new_dic = {
                "general_info": {
                    "total_players": len(self.players),
                    "alive_players": len(alive_players)
                },
                "players_info": players_info
            }

            # store the final state
            final_info = new_dic

            # check if game is over
            if len(alive_players) == 1:
                print("Game Over, winner is:", alive_players[0].username)
                if not self.training_mode:
                    if self.use_advanced_UI:
                        self.advanced_UI.display_winner_screen(alive_players)
                    else:
                        self.screen.fill("green")

                game_over = True
                break

        # skip all rendering operations in training mode for better performance
        if not self.training_mode:
            if self.use_advanced_UI:
                self.advanced_UI.draw_everything(final_info, self.players, self.obstacles)
            else:
                # draw obstacles manually if not using advanced UI
                for obstacle in self.obstacles:
                    obstacle.draw(self.world_surface)

            # scale and display the world surface
            scaled_surface = pygame.transform.scale(self.world_surface, (self.display_width, self.display_height))
            self.screen.blit(scaled_surface, (0, 0))
            pygame.display.flip()

        # in training mode, use a high tick rate but not unreasonably high
        if not self.training_mode:
            self.clock.tick(120)  # normal gameplay speed
        else:
            # skip the clock tick entirely in training mode for maximum speed
            pass  # no tick limiting in training mode for maximum speed

        # return the final state
        if game_over:
            print("Total steps:", self.steps)
            return True, final_info  # Game is over
        else:
            # return the final state from the last frame
            return False, final_info

    def calculate_reward(self, info_dictionary, bot_username):
        players_info = info_dictionary.get("players_info", {})
        bot_info = players_info.get(bot_username, {})

        # Initialize episode tracking if not exists
        if not hasattr(self, 'episode_rewards'):
            self.episode_rewards = {}
        if bot_username not in self.episode_rewards:
            self.episode_rewards[bot_username] = 0.0

        # Initialize position tracking if not exists
        if bot_username not in self.last_positions:
            self.last_positions[bot_username] = {
                'position': None,
                'time_in_position': 0
            }

        step_reward = 0.0

        # 1) Movement reward (normalized by max possible movement)
        distance_moved = bot_info.get("distance", 0.0)
        max_possible_movement = 10.0
        movement_reward = (distance_moved / max_possible_movement) * 0.2
        step_reward += movement_reward

        # Position tracking and stagnation penalty
        current_pos = bot_info.get("location", [0, 0])
        last_pos_data = self.last_positions[bot_username]
        
        # Check if position has changed significantly (more than 5 units)
        if last_pos_data['position'] is None:
            last_pos_data['position'] = current_pos
            last_pos_data['time_in_position'] = 0
        else:
            distance = math.dist(current_pos, last_pos_data['position'])
            if distance < 5:  # If moved less than 5 units
                last_pos_data['time_in_position'] += 1
                if last_pos_data['time_in_position'] > 30:  # After 30 steps in same position
                    # Progressive penalty that increases with time
                    stagnation_penalty = 0.1 * (last_pos_data['time_in_position'] - 30) / 30
                    step_reward -= stagnation_penalty
            else:
                last_pos_data['position'] = current_pos
                last_pos_data['time_in_position'] = 0

        # 2) Obstacle avoidance (with smoother penalty)
        rays = bot_info.get("rays", [])
        if rays:
            def extract_numeric(value):
                while isinstance(value, (list, tuple)):
                    if len(value) > 0:
                        value = value[0]
                    else:
                        break
                return value

            flat_rays = [extract_numeric(r) for r in rays if extract_numeric(r) is not None]
            if flat_rays:
                min_ray = min(flat_rays)
                safe_distance = 50.0
                danger_zone = 20.0
                if min_ray < safe_distance:
                    # Smooth penalty that increases as distance decreases
                    penalty = ((safe_distance - min_ray) / safe_distance) * 0.5
                    if min_ray < danger_zone:
                        penalty *= 2  # Double penalty in danger zone
                    step_reward -= penalty

        # 3) Combat rewards (with better scaling)
        shot_fired = bot_info.get("shot_fired", False)
        damage_dealt = bot_info.get("damage_dealt", 0.0)
        last_damage = self.last_damage.get(bot_username, 0.0)
        damage_this_step = damage_dealt - last_damage

        # Add aiming reward based on ray data
        rays = bot_info.get("rays", [])
        if rays:
            # Track which arrays detect the enemy
            enemy_detected = {
                'array1': False,  # leftmost
                'array2': False,  # left
                'shooting': False,  # center
                'array3': False,  # right
                'array4': False   # rightmost
            }
            
            # Initialize last frame's detections if not exists
            if not hasattr(self, 'last_enemy_detected'):
                self.last_enemy_detected = {}
            if bot_username not in self.last_enemy_detected:
                self.last_enemy_detected[bot_username] = enemy_detected.copy()

            # Check each ray for enemy detection
            for i, ray in enumerate(rays):
                if isinstance(ray, list) and len(ray) == 3:
                    hit_type = ray[2]
                    if hit_type == "player":
                        # Determine which array detected the enemy based on ray index
                        if i < len(rays)//5:  # array1
                            enemy_detected['array1'] = True
                        elif i < 2*len(rays)//5:  # array2
                            enemy_detected['array2'] = True
                        elif i < 3*len(rays)//5:  # shooting array
                            enemy_detected['shooting'] = True
                        elif i < 4*len(rays)//5:  # array3
                            enemy_detected['array3'] = True
                        else:  # array4
                            enemy_detected['array4'] = True

            # Check for successful transitions
            last_detected = self.last_enemy_detected[bot_username]
            
            # Left side transitions
            if last_detected['array1'] and enemy_detected['array2']:
                step_reward += 0.2  # Reward for array1 -> array2 transition
            if last_detected['array2'] and enemy_detected['shooting']:
                step_reward += 1.0  # Reward for array2 -> shooting transition
                
            # Right side transitions
            if last_detected['array4'] and enemy_detected['array3']:
                step_reward += 0.2  # Reward for array4 -> array3 transition
            if last_detected['array3'] and enemy_detected['shooting']:
                step_reward += 1.0  # Reward for array3 -> shooting transition

            # Store current detections for next frame
            self.last_enemy_detected[bot_username] = enemy_detected.copy()

            # Additional reward for maintaining enemy in shooting array
            if enemy_detected['shooting']:
                aiming_reward = 0.3
                step_reward += aiming_reward
                
                # Additional reward for shooting when enemy is in view
                if shot_fired:
                    if damage_this_step > 0:
                        # Increased reward for damage when enemy is in view
                        damage_reward = (damage_this_step / 100.0) * 5.0
                        step_reward += damage_reward
                    else:
                        # Smaller penalty for missing when enemy is in view
                        step_reward -= 0.1
            else:
                # Penalty for shooting when enemy is not in view
                if shot_fired:
                    step_reward -= 0.2

        # 4) Kill rewards (with progressive bonus)
        kills = bot_info.get("kills", 0)
        last_kills = self.last_kills.get(bot_username, 0)
        if kills > last_kills:
            kill_reward = 3.0
            # Add bonus for efficiency (based on health and ammo)
            health_ratio = bot_info.get("health", 0) / 100.0
            ammo_ratio = bot_info.get("current_ammo", 0) / 30.0
            efficiency_bonus = (health_ratio + ammo_ratio) * 2.0
            step_reward += kill_reward + efficiency_bonus

        # 5) Health management (with progressive penalty)
        health = bot_info.get("health", 100)
        last_health = self.last_health.get(bot_username, 100)
        if health < last_health:
            damage_taken = last_health - health
            # Progressive penalty based on remaining health
            health_ratio = health / 100.0
            health_penalty = (damage_taken / 100.0) * (2.0 - health_ratio)
            step_reward -= health_penalty

        # 6) Tactical positioning reward
        closest_opponent = bot_info.get("closest_opponent")
        if closest_opponent:
            current_pos = bot_info.get("location", [0, 0])
            distance_to_opponent = math.sqrt(
                (current_pos[0] - closest_opponent[0])**2 + 
                (current_pos[1] - closest_opponent[1])**2
            )
            optimal_range = 200  # Optimal fighting distance
            # Reward being at optimal range
            position_reward = 0.2 * (1.0 - abs(distance_to_opponent - optimal_range) / optimal_range)
            step_reward += position_reward

        # 7) Ammo management (with progressive penalty)
        current_ammo = bot_info.get("current_ammo", 30)
        max_ammo = 30
        if current_ammo == 0:
            step_reward -= 0.5
        elif current_ammo < max_ammo * 0.2:
            step_reward -= 0.1 * (1 - current_ammo / (max_ammo * 0.2))

        # 8) Survival reward (scaled by health)
        alive = bot_info.get("alive", True)
        if alive:
            health_ratio = health / 100.0
            step_reward += 0.05 * health_ratio

        # Update tracking variables
        self.last_damage[bot_username] = damage_dealt
        self.last_kills[bot_username] = kills
        self.last_health[bot_username] = health
        
        # Clip the step reward
        step_reward = max(-1.0, min(1.0, step_reward))
        
        # Update episode reward
        self.episode_rewards[bot_username] += step_reward

        # If episode is over (bot died or game ended)
        if not alive:
            # Add final episode reward
            final_reward = self.episode_rewards[bot_username]
            # Reset episode tracking
            self.episode_rewards[bot_username] = 0.0
            return final_reward
            
        # Apply curriculum weights
        if hasattr(self, 'reward_weights'):
            step_reward = (
                self.reward_weights["movement"] * movement_reward +
                self.reward_weights["combat"] * damage_reward +
                self.reward_weights["survival"] * step_reward +
                self.reward_weights["tactical"] * position_reward
            )
        
        return step_reward
