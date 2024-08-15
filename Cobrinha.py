import gymnasium as gym
from CobrinhaEnv import CobrinhaEnv
import pygame

# Mapeamento das teclas para as ações
key_action_map = {
    pygame.K_w: 0,  # Cima
    pygame.K_a: 1,  # Esquerda
    pygame.K_s: 2,  # Baixo
    pygame.K_d: 3   # Direita
}

def play_manual(env):
    # Reseta o ambiente
    obs, _ = env.reset()
    done = False

    while not done:
        # Aguarda o jogador pressionar uma tecla
        action = None
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key in key_action_map:
                action = key_action_map[event.key]
            if event.type == pygame.QUIT:
                done = True
                break

        if action is not None:
            obs, reward, done, _, _ = env.step(action)
            env.render()

        # Limitar o FPS
        env.clock.tick(env.metadata['render_fps'])

    env.close()

if __name__ == "__main__":
    gym.register(
        id='Cobrinha-v0',
        entry_point='CobrinhaEnv:CobrinhaEnv'
    )

    env = gym.make('Cobrinha-v0', render_mode='human', render_tick=10)

    # Inicia o jogo manual
    play_manual(env)