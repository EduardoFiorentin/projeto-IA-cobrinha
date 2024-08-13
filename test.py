from stable_baselines3 import DQN
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from CobrinhaEnv import CobrinhaEnv

# env = DummyVecEnv([lambda: CobrinhaEnv(render_mode="human", render_tick=50)])

gym.register(
    id='Cobrinha',
    entry_point='CobrinhaEnv:CobrinhaEnv',
    # parâmetros __init__
    # kwargs={'game': None}
)
env = gym.make('Cobrinha', render_mode = "human", render_tick=50)

# Carregando o modelo treinado
model = DQN.load("dqn_snake2")

model.exploration_rate = 0

# Teste do modelo treinado
for i in range(0, 10):
    obs, _ = env.reset()
    print("Rodada: ", i)
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        env.render()
        
        if done:
            break
        