from gymnasium import Env
import pygame
from pygame.locals import *
from sys import exit
from random import randint

class CobrinhaEnv(Env):
    def __init__(self):
        pass
        
    def step(self):
        pass
    
    def render(self):
        pass
    
    def reset(self):
        pass


        

running = True

# configuration
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
BLOCK_SIZE = 20

#flags
direction_x = 0
direction_y = 0

snake_body = [(0, 0)]
ate = False
food = (140, 140)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake")
pygame.init()
clock = pygame.time.Clock()

while running: 
    clock.tick(5)
    screen.fill((0, 0, 0))
    
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit() 
            exit()
            
        if event.type == KEYDOWN:
            if event.key == K_w:
                direction_x = 0 
                direction_y = -BLOCK_SIZE
                
            if event.key == K_a:
                direction_x = -BLOCK_SIZE
                direction_y = 0
            
            if event.key == K_s:
                direction_x = 0
                direction_y = BLOCK_SIZE
                
            if event.key == K_d:
                direction_x = BLOCK_SIZE
                direction_y = 0
            
            if event.key == K_m:
                ate = True
                
            
    for position in snake_body:
        pygame.draw.rect(screen, (0, 255, 0), (position[0], position[1], BLOCK_SIZE, BLOCK_SIZE))
        
    pygame.draw.rect(screen, (255, 0, 0), (food[0], food[1], BLOCK_SIZE, BLOCK_SIZE))
        
        
    new_position = (snake_body[-1][0] + direction_x, snake_body[-1][1] + direction_y)
    snake_body.append(new_position)
    
    # Verificar se pegou a comida 
    if food == snake_body[-1]:
        ate = True
        
    if not ate: 
        snake_body.pop(0)
    
    if ate:
        ate = False
        food = (randint(0, SCREEN_WIDTH // BLOCK_SIZE - 1) * BLOCK_SIZE, randint(0, SCREEN_HEIGHT // BLOCK_SIZE - 1) * BLOCK_SIZE)
    
    pygame.display.flip()
    

    
