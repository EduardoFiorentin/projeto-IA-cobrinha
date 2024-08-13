import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from pygame.locals import *
from random import randint

class CobrinhaEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(CobrinhaEnv, self).__init__()
        
        self.render_mode = render_mode

        # Configuração do ambiente
        self.screen_width = 500
        self.screen_height = 500
        self.block_size = 20

        self.tryes = 0

        # Definindo os espaços de observação e ação
        self.observation_space = spaces.Box(low=0, high=255, shape=(100, 100, 3), dtype=np.uint8)  # Redimensionado para 25x25
        self.action_space = spaces.Discrete(4)  # 4 direções: cima, baixo, esquerda, direita

        # # Inicializa o Pygame
        # pygame.init()
        # self.screen = pygame.Surface((self.screen_width, self.screen_height))
        # self.clock = pygame.time.Clock()
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()

        # Inicializa outros parâmetros
        self.rng = np.random.default_rng()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng, seed = gym.utils.seeding.np_random(seed)
        
        self.tryes = 0

        self.direction_x = self.block_size
        self.direction_y = 0
        self.snake_body = [(0, 0)]
        self.ate = False
        self.food = (randint(0, self.screen_width // self.block_size - 1) * self.block_size,
                     randint(0, self.screen_height // self.block_size - 1) * self.block_size)

        return self._get_state(), {}

    def step(self, action):

        self.tryes += 1
        
        if action == 0:
            self.direction_x = 0
            self.direction_y = -self.block_size
        elif action == 1:
            self.direction_x = -self.block_size
            self.direction_y = 0
        elif action == 2:
            self.direction_x = 0
            self.direction_y = self.block_size
        elif action == 3:
            self.direction_x = self.block_size
            self.direction_y = 0

        new_position = (self.snake_body[-1][0] + self.direction_x, self.snake_body[-1][1] + self.direction_y)
        self.snake_body.append(new_position)
        
        
        # RECOMPENSA 
        reward = 0

        if self.food == self.snake_body[-1]:
            self.ate = True
            reward = 2500
            self.tryes = 0
        else:
            self.snake_body.pop(0)
            
            # distance = abs(self.food[0] - self.snake_body[-1][0]) + abs(self.food[1] - self.snake_body[-1][1])
            distance = pow(pow((self.food[0] - self.snake_body[-1][0]), 2) + pow(self.food[1] - self.snake_body[-1][1], 2), (1/2))
            # print(-distance)
            
            reward = -(distance // 20) if (distance // 20) > 1 else -1
            # reward = -5
            

        if self.ate:
            self.ate = False
            self.food = (randint(0, self.screen_width // self.block_size - 1) * self.block_size,
                         randint(0, self.screen_height // self.block_size - 1) * self.block_size)

        done = False
        if (new_position[0] < 0 or new_position[0] >= self.screen_width or
                new_position[1] < 0 or new_position[1] >= self.screen_height or
                len(self.snake_body) != len(set(self.snake_body))):
            done = True
            reward += -500

        # limtar tentativas
        # if self.tryes >= 100:
        #     self.tryes = 0
        #     reward += -1000
        #     done = True

        # reward = 1 if self.ate else -1 if done else 0
        
        if self.render_mode: self.render()

        return self._get_state(), reward, done, False, {}

    def render(self):
        # self.screen.fill((0, 0, 0))

        # for position in self.snake_body:
        #     pygame.draw.rect(self.screen, (0, 255, 0), (position[0], position[1], self.block_size, self.block_size))

        # pygame.draw.rect(self.screen, (255, 0, 0), (self.food[0], self.food[1], self.block_size, self.block_size))

        # pygame.display.flip()
        
        if self.render_mode:
            self.screen.fill((0, 0, 0))

            for position in self.snake_body:
                if position == self.snake_body[-1]: pygame.draw.rect(self.screen, (0, 130, 150), (position[0], position[1], self.block_size, self.block_size))
                else: pygame.draw.rect(self.screen, (0, 255, 0), (position[0], position[1], self.block_size, self.block_size))

            pygame.draw.rect(self.screen, (255, 0, 0), (self.food[0], self.food[1], self.block_size, self.block_size))
                
            pygame.display.flip()
            # self.clock.tick(10)

    def _get_state(self):
        raw_image = pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)
        resized_image = pygame.transform.smoothscale(pygame.surfarray.make_surface(raw_image), (100, 100))
        normalized_image = pygame.surfarray.array3d(resized_image).transpose(1, 0, 2) / 255.0
        # return pygame.surfarray.array3d(resized_image).transpose(1, 0, 2)
        return normalized_image

    def close(self):
        pygame.quit()

# Testando o ambiente
if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    
    env = CobrinhaEnv()
    check_env(env)
