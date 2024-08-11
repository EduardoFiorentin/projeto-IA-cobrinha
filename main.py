import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from CobrinhaEnv import CobrinhaEnv 

from stable_baselines3 import DQN
from CobrinhaEnv import CobrinhaEnv

# Criação do ambiente
# env = CobrinhaEnv()

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from CobrinhaEnv import CobrinhaEnv
import time

# Criação do ambiente e envolvimento com DummyVecEnv
env = DummyVecEnv([lambda: CobrinhaEnv(render_mode="human")])


# Inicialização do modelo DQN com parâmetros ajustados
# model = DQN(
#     "MlpPolicy",
#     env,
#     verbose=1,
#     tensorboard_log="./dqn_snake_tensorboard/",
#     buffer_size=50000,  
#     batch_size=32,     
#     learning_starts=1000
# )

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./dqn_snake_tensorboard/",
    buffer_size=10000,
    batch_size=64,
    learning_starts=1000,
    policy_kwargs=dict(net_arch=[64, 64]),
    exploration_initial_eps=0.8,
    exploration_final_eps=0.3
)

# Treinamento do modelo
model.learn(total_timesteps=5000)

# Salvamento do modelo treinado
model.save("dqn_snake")


# TESTES =============================================================================


# Carregando o modelo treinado
model = DQN.load("dqn_snake")

model.exploration_rate = 0

# Teste do modelo treinado
for i in range(0, 10):
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            break
        
        
# Verificar se a imagem vista pela IA está sendo atualizada 

# Trocar a visão da ia para distância e direção até a comida