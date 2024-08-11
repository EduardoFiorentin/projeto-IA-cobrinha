import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from pygame.locals import *
from random import randint

# class CobrinhaEnv(Env):
#     def __init__(self):
#         super(CobrinhaEnv, self).__init__()
        
#         # Configurações do jogo
#         self.SCREEN_WIDTH = 500
#         self.SCREEN_HEIGHT = 500
#         self.BLOCK_SIZE = 20
#         self.snake_body = [(0, 0)]
#         self.direction_x = 0
#         self.direction_y = 0
#         self.food = (140, 140)
#         self.ate = False
#         self.done = False
        
#         # Definindo o espaço de ações: 0 - cima, 1 - baixo, 2 - esquerda, 3 - direita
#         self.action_space = spaces.Discrete(4)
        
#         # Espaço de observação: matriz do tamanho da tela, com 3 canais (snake, food, empty)
#         self.observation_space = spaces.Box(low=0, high=1, shape=(self.SCREEN_WIDTH // self.BLOCK_SIZE, 
#                                                                   self.SCREEN_HEIGHT // self.BLOCK_SIZE, 3), dtype=np.float32)
        
#         # Inicializando o pygame
#         self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
#         pygame.display.set_caption("Snake")
#         pygame.init()
#         self.clock = pygame.time.Clock()
        
#     def step(self, action):
#         # Atualizar direção com base na ação
#         if action == 0 and self.direction_y == 0:  # Cima
#             self.direction_x = 0
#             self.direction_y = -self.BLOCK_SIZE
#         elif action == 1 and self.direction_y == 0:  # Baixo
#             self.direction_x = 0
#             self.direction_y = self.BLOCK_SIZE
#         elif action == 2 and self.direction_x == 0:  # Esquerda
#             self.direction_x = -self.BLOCK_SIZE
#             self.direction_y = 0
#         elif action == 3 and self.direction_x == 0:  # Direita
#             self.direction_x = self.BLOCK_SIZE
#             self.direction_y = 0
        
#         # Mover a cabeça da cobra
#         new_position = (self.snake_body[-1][0] + self.direction_x, self.snake_body[-1][1] + self.direction_y)
#         self.snake_body.append(new_position)
        
#         # Verificar se pegou a comida
#         if self.food == self.snake_body[-1]:
#             self.ate = True
        
#         if not self.ate:
#             self.snake_body.pop(0)
#         else:
#             self.ate = False
#             self.food = (randint(0, self.SCREEN_WIDTH // self.BLOCK_SIZE - 1) * self.BLOCK_SIZE, 
#                          randint(0, self.SCREEN_HEIGHT // self.BLOCK_SIZE - 1) * self.BLOCK_SIZE)
        
#         # Verificar colisões
#         if (new_position[0] < 0 or new_position[0] >= self.SCREEN_WIDTH or 
#             new_position[1] < 0 or new_position[1] >= self.SCREEN_HEIGHT or 
#             len(self.snake_body) != len(set(self.snake_body))):
#             self.done = True
            
#         # Calcular a recompensa
#         reward = 1 if self.ate else -0.1 if not self.done else -10
        
#         # Observar o estado atual
#         observation = self._get_observation()
        
#         return observation, reward, self.done, {}
    
#     def render(self, mode='human'):
#         self.screen.fill((0, 0, 0))
#         for position in self.snake_body:
#             pygame.draw.rect(self.screen, (0, 255, 0), (position[0], position[1], self.BLOCK_SIZE, self.BLOCK_SIZE))
#         pygame.draw.rect(self.screen, (255, 0, 0), (self.food[0], self.food[1], self.BLOCK_SIZE, self.BLOCK_SIZE))
#         pygame.display.flip()
    
#     def reset(self):
#         self.snake_body = [(0, 0)]
#         self.direction_x = 0
#         self.direction_y = 0
#         self.food = (140, 140)
#         self.ate = False
#         self.done = False
#         return self._get_observation()
    
#     def _get_observation(self):
#         # Criar uma matriz de zeros para representar o estado do jogo
#         observation = np.zeros((self.SCREEN_WIDTH // self.BLOCK_SIZE, self.SCREEN_HEIGHT // self.BLOCK_SIZE, 3))
        
#         # Preencher com a posição da cobra
#         for x, y in self.snake_body:
#             observation[y // self.BLOCK_SIZE, x // self.BLOCK_SIZE, 0] = 1
        
#         # Preencher com a posição da comida
#         observation[self.food[1] // self.BLOCK_SIZE, self.food[0] // self.BLOCK_SIZE, 1] = 1
        
#         return observation


# teste 2 - 

# class CobrinhaEnv(gym.Env):
#     def __init__(self):
#         super(CobrinhaEnv, self).__init__()

#         # Configuração do ambiente
#         self.screen_width = 500
#         self.screen_height = 500
#         self.block_size = 20

#         # Definindo os espaços de observação e ação
#         self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3), dtype=np.uint8)
#         self.action_space = spaces.Discrete(4)  # 4 direções: cima, baixo, esquerda, direita

#         # Inicializa o Pygame
#         pygame.init()
#         self.screen = pygame.Surface((self.screen_width, self.screen_height))
#         self.clock = pygame.time.Clock()

#         # Inicializa outros parâmetros
#         self.rng = np.random.default_rng()  # Inicializa a seed
#         self.reset()

#     def reset(self, seed=None, options=None):
#         # Configura a semente de aleatoriedade
#         super().reset(seed=seed)
#         self.rng, seed = gym.utils.seeding.np_random(seed)
        
#         # Inicialização do estado
#         self.direction_x = 0
#         self.direction_y = 0
#         self.snake_body = [(0, 0)]
#         self.ate = False
#         self.food = (randint(0, self.screen_width // self.block_size - 1) * self.block_size,
#                      randint(0, self.screen_height // self.block_size - 1) * self.block_size)

#         # Retorna o estado inicial
#         return self._get_state(), {}

#     def step(self, action):
#         # Atualiza a direção da cobra com base na ação
#         if action == 0:  # Cima
#             self.direction_x = 0
#             self.direction_y = -self.block_size
#         elif action == 1:  # Esquerda
#             self.direction_x = -self.block_size
#             self.direction_y = 0
#         elif action == 2:  # Baixo
#             self.direction_x = 0
#             self.direction_y = self.block_size
#         elif action == 3:  # Direita
#             self.direction_x = self.block_size
#             self.direction_y = 0

#         # Move a cobra
#         new_position = (self.snake_body[-1][0] + self.direction_x, self.snake_body[-1][1] + self.direction_y)
#         self.snake_body.append(new_position)

#         # Verifica se a cobra comeu a comida
#         if self.food == self.snake_body[-1]:
#             self.ate = True
#         else:
#             self.snake_body.pop(0)  # Remove a cauda se não comeu

#         # Gera uma nova comida se a cobra comeu
#         if self.ate:
#             self.ate = False
#             self.food = (randint(0, self.screen_width // self.block_size - 1) * self.block_size,
#                          randint(0, self.screen_height // self.block_size - 1) * self.block_size)

#         # Verifica se a cobra colidiu com as paredes ou com ela mesma
#         done = False
#         if (new_position[0] < 0 or new_position[0] >= self.screen_width or
#                 new_position[1] < 0 or new_position[1] >= self.screen_height or
#                 len(self.snake_body) != len(set(self.snake_body))):  # Colisão com ela mesma
#             done = True

#         # Calcula a recompensa
#         reward = 1 if self.ate else -1 if done else 0

#         # Retorna o novo estado, recompensa, se o episódio terminou e informações adicionais
#         return self._get_state(), reward, done, False, {}

#     def render(self):
#         # Preenche a tela com preto
#         self.screen.fill((0, 0, 0))
        
#         # Desenha o corpo da cobra
#         for position in self.snake_body:
#             pygame.draw.rect(self.screen, (0, 255, 0), (position[0], position[1], self.block_size, self.block_size))
        
#         # Desenha a comida
#         pygame.draw.rect(self.screen, (255, 0, 0), (self.food[0], self.food[1], self.block_size, self.block_size))
        
#         # Atualiza a tela
#         pygame.display.flip()

#     def _get_state(self):
#         # Captura o estado atual da tela
#         return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)

#     def close(self):
#         # Encerra o Pygame
#         pygame.quit()

# # Testando o ambiente
# if __name__ == "__main__":
#     from stable_baselines3.common.env_checker import check_env
    
#     env = CobrinhaEnv()
#     check_env(env)


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
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)  # Redimensionado para 84x84
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
        
        reward = 0

        if self.food == self.snake_body[-1]:
            self.ate = True
            reward = 100
        else:
            self.snake_body.pop(0)
            reward = -50
            

        if self.ate:
            self.ate = False
            self.food = (randint(0, self.screen_width // self.block_size - 1) * self.block_size,
                         randint(0, self.screen_height // self.block_size - 1) * self.block_size)

        done = False
        if (new_position[0] < 0 or new_position[0] >= self.screen_width or
                new_position[1] < 0 or new_position[1] >= self.screen_height or
                len(self.snake_body) != len(set(self.snake_body))):
            done = True
            # reward = -150

        if self.tryes >= 100:
            self.tryes = 0
            done = True

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
                pygame.draw.rect(self.screen, (0, 255, 0), (position[0], position[1], self.block_size, self.block_size))

            pygame.draw.rect(self.screen, (255, 0, 0), (self.food[0], self.food[1], self.block_size, self.block_size))
                
            pygame.display.flip()
            self.clock.tick(60)

    def _get_state(self):
        raw_image = pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)
        resized_image = pygame.transform.smoothscale(pygame.surfarray.make_surface(raw_image), (84, 84))
        return pygame.surfarray.array3d(resized_image).transpose(1, 0, 2)

    def close(self):
        pygame.quit()

# Testando o ambiente
if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    
    env = CobrinhaEnv()
    check_env(env)
