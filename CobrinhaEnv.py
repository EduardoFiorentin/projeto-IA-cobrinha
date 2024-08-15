# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import pygame
# from pygame.locals import *
# from random import randint

# class CobrinhaEnv(gym.Env):
    
#     metadata = {'render_modes': ['human'], 'render_fps': 50}
    
#     def __init__(self, render_mode=None, render_tick=None, limit_steps=-1):
#         super(CobrinhaEnv, self).__init__()

#         # Configuração do ambiente
#         self.render_mode = render_mode
#         self.screen_width = 500
#         self.screen_height = 500
#         self.block_size = 20
#         self.render_tick = render_tick
#         self.tryes = 0
#         self.steps = 0
#         self.eat_times = 0
#         self.limit_steps = limit_steps

#         # Definindo os espaços de observação e ação
#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
#         self.action_space = spaces.Discrete(4)  # 4 direções: cima, baixo, esquerda, direita

#         # Inicializa o Pygame
#         pygame.init()
#         self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
#         pygame.display.set_caption("Snake")
#         self.clock = pygame.time.Clock()

#         # Inicializa outros parâmetros
#         self.rng = np.random.default_rng()
#         self.reset()

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.rng, seed = gym.utils.seeding.np_random(seed)
        
#         self.tryes = 0
#         self.steps = 0

#         self.direction_x = self.block_size
#         self.direction_y = 0
#         self.snake_body = [(0, 0)]
#         self.ate = False
#         self.food = (randint(0, self.screen_width // self.block_size - 1) * self.block_size,
#                      randint(0, self.screen_height // self.block_size - 1) * self.block_size)

#         return self._get_state(), {}

#     def step(self, action):

#         self.tryes += 1
#         self.steps += 1
        
#         if action == 0:
#             self.direction_x = 0
#             self.direction_y = -self.block_size
#         elif action == 1:
#             self.direction_x = -self.block_size
#             self.direction_y = 0
#         elif action == 2:
#             self.direction_x = 0
#             self.direction_y = self.block_size
#         elif action == 3:
#             self.direction_x = self.block_size
#             self.direction_y = 0

#         new_position = (self.snake_body[-1][0] + self.direction_x, self.snake_body[-1][1] + self.direction_y)
#         self.snake_body.append(new_position)
        
        
#         # RECOMPENSA 
#         reward = 0

#         if self.food == self.snake_body[-1]:
#             self.ate = True
#             reward = 2500
#             self.tryes = 0
#         else:
#             self.snake_body.pop(0)
#             distance = pow(pow((self.food[0] - self.snake_body[-1][0]), 2) + pow(self.food[1] - self.snake_body[-1][1], 2), (1/2))
#             reward = -(distance // 20) if (distance // 20) > 1 else -1
            

#         if self.ate:
#             self.ate = False
#             self.food = (randint(0, self.screen_width // self.block_size - 1) * self.block_size,
#                          randint(0, self.screen_height // self.block_size - 1) * self.block_size)

#         done = False
#         if (new_position[0] < 0 or new_position[0] >= self.screen_width or
#                 new_position[1] < 0 or new_position[1] >= self.screen_height):
#             done = True
#             reward += -500

#         # limtar tentativas
#         if self.tryes >= 100:
#             self.tryes = 0
#             reward += -1000
#             done = True

#         # Opção para limitar quantidade de steps por episodio
#         if self.limit_steps != -1: 
#             if self.steps > self.limit_steps: 
#                 done = True
        
#         if self.render_mode: self.render()

#         return self._get_state(), reward, done, False, {}

#     def render(self):
#         if self.render_mode:
#             self.screen.fill((0, 0, 0))

#             for position in self.snake_body:
#                 if position == self.snake_body[-1]: pygame.draw.rect(self.screen, (0, 130, 150), (position[0], position[1], self.block_size, self.block_size))
#                 else: pygame.draw.rect(self.screen, (0, 255, 0), (position[0], position[1], self.block_size, self.block_size))

#             pygame.draw.rect(self.screen, (255, 0, 0), (self.food[0], self.food[1], self.block_size, self.block_size))
                
#             pygame.display.flip()
#             if self.render_tick != None: self.clock.tick(self.render_tick)

#     def _get_state(self):
#         dx = (self.food[0] - self.snake_body[-1][0]) / self.screen_width
#         dy = (self.food[1] - self.snake_body[-1][1]) / self.screen_height
#         dist_comida = np.sqrt(dx**2 + dy**2)
#         dir_x = self.direction_x / self.block_size
#         dir_y = self.direction_y / self.block_size

#         # Vetor de observação
#         return np.array([dx, dy, dist_comida, dir_x, dir_y], dtype=np.float32)

#     def close(self):
#         pygame.quit()
        
#     def get_data():
#         pass


# if __name__ == "__main__":
#     from stable_baselines3.common.env_checker import check_env
    
#     env = CobrinhaEnv()
#     check_env(env)





# Cobrinha completa
import pygame
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from random import randint


class CobrinhaEnv(gym.Env):

    metadata = {'render_modes': ['human'], 'render_fps': 50}

    def __init__(self, render_mode=None, render_tick=None, limit_steps=-1):
        super(CobrinhaEnv, self).__init__()

        self.render_mode = render_mode
        self.screen_width = 500
        self.screen_height = 500
        self.block_size = 20
        self.render_tick = render_tick
        self.tryes = 0
        self.steps = 0
        self.eat_times = 0
        self.limit_steps = limit_steps

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()

        self.rng = np.random.default_rng()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng, seed = gym.utils.seeding.np_random(seed)

        self.tryes = 0
        self.steps = 0

        self.direction_x = self.block_size
        self.direction_y = 0
        self.snake_body = [(0, 0)]
        self.ate = False
        self.food = self._generate_food()

        return self._get_state(), {}

    def _generate_food(self):
        while True:
            food_position = (
                randint(0, self.screen_width // self.block_size - 1) * self.block_size,
                randint(0, self.screen_height // self.block_size - 1) * self.block_size
            )
            if food_position not in self.snake_body:
                return food_position

    def step(self, action):

        self.tryes += 1
        self.steps += 1

        if action == 0 and self.direction_y == 0:  # Cima
            self.direction_x = 0
            self.direction_y = -self.block_size
        elif action == 1 and self.direction_x == 0:  # Esquerda
            self.direction_x = -self.block_size
            self.direction_y = 0
        elif action == 2 and self.direction_y == 0:  # Baixo
            self.direction_x = 0
            self.direction_y = self.block_size
        elif action == 3 and self.direction_x == 0:  # Direita
            self.direction_x = self.block_size
            self.direction_y = 0

        new_position = (self.snake_body[-1][0] + self.direction_x, self.snake_body[-1][1] + self.direction_y)
        self.snake_body.append(new_position)

        reward = 0
        done = False

        if self.food == new_position:
            self.ate = True
            reward = 2500
            self.tryes = 0
        else:
            self.snake_body.pop(0)
            distance = np.sqrt((self.food[0] - new_position[0])**2 + (self.food[1] - new_position[1])**2)
            reward = -(distance // 20) if (distance // 20) > 1 else -1

        if new_position in self.snake_body[:-1]:
            done = True
            reward += -1000

        if self.ate:
            self.ate = False
            self.food = self._generate_food()

        if (new_position[0] < 0 or new_position[0] >= self.screen_width or
                new_position[1] < 0 or new_position[1] >= self.screen_height):
            done = True
            reward += -500

        if self.tryes >= 100:
            self.tryes = 0
            reward += -1000
            done = True

        if self.limit_steps != -1 and self.steps > self.limit_steps:
            done = True

        if self.render_mode:
            self.render()

        return self._get_state(), reward, done, False, {}

    def render(self):
        if self.render_mode:
            self.screen.fill((0, 0, 0))

            for position in self.snake_body:
                color = (0, 130, 150) if position == self.snake_body[-1] else (0, 255, 0)
                pygame.draw.rect(self.screen, color, (position[0], position[1], self.block_size, self.block_size))

            pygame.draw.rect(self.screen, (255, 0, 0), (self.food[0], self.food[1], self.block_size, self.block_size))

            pygame.display.flip()
            if self.render_tick is not None:
                self.clock.tick(self.render_tick)

    def _get_state(self):
        dx = (self.food[0] - self.snake_body[-1][0]) / self.screen_width
        dy = (self.food[1] - self.snake_body[-1][1]) / self.screen_height
        dist_comida = np.sqrt(dx**2 + dy**2)
        dir_x = self.direction_x / self.block_size
        dir_y = self.direction_y / self.block_size

        # Verificar se a próxima ação resultará em colisão com o corpo
        # next_position = (self.snake_body[-1][0] + self.direction_x, self.snake_body[-1][1] + self.direction_y)
        # collision_with_self = 1.0 if next_position in self.snake_body[:-1] else 0.0
        
        np_x_1_y_0 = 1.0 if (self.snake_body[-1][0] + self.block_size, self.snake_body[-1][1]) in self.snake_body[:-1] else 0.0
        np_x_0_y_1 =  1.0 if (self.snake_body[-1][0], self.snake_body[-1][1] + self.block_size) in self.snake_body[:-1] else 0.0
        np_x_m1_y_0 =  1.0 if (self.snake_body[-1][0] - self.block_size, self.snake_body[-1][1]) in self.snake_body[:-1] else 0.0
        np_x_0_y_m1 =  1.0 if (self.snake_body[-1][0], self.snake_body[-1][1] - self.block_size) in self.snake_body[:-1] else 0.0
        
        # print(np_x_1_y_0, np_x_0_y_1, np_x_m1_y_0, np_x_0_y_m1)

        # Assegure que o vetor de observação tenha sempre comprimento 7
        state = np.array([dx, dy, dist_comida, dir_x, dir_y, np_x_1_y_0, np_x_0_y_1, np_x_m1_y_0, np_x_0_y_m1], dtype=np.float32)
        return state

    def close(self):
        pygame.quit()

    def get_data(self):
        return {
            "length": len(self.snake_body),
            "steps": self.steps,
            "eaten": self.eat_times,
        }


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    env = CobrinhaEnv()
    check_env(env)
