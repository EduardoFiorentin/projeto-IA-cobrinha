from stable_baselines3 import DQN
from CobrinhaEnv import CobrinhaEnv 
import gymnasium as gym
import os

from stable_baselines3.common.vec_env import DummyVecEnv

DIR_NAME = "treinoTeste"
TRAIN_STEPS = 50000
DIVISIONS = 10

NOT_ALLOW_REUSE_DIRS = False

if (not os.path.isdir(DIR_NAME)): os.makedirs(DIR_NAME)
else: 
    print(f"Diretório '{DIR_NAME}' já existe. Usá-lo pode afetar o conteúdo pré existente.")
    if NOT_ALLOW_REUSE_DIRS: exit(0)

gym.register(
    id='Cobrinha',
    entry_point='CobrinhaEnv:CobrinhaEnv'
)
env = gym.make(
    'Cobrinha', 
    render_mode = "human"
)
# env = DummyVecEnv([lambda: CobrinhaEnv(render_mode="human")])



model = DQN(
    'MlpPolicy',  # Política MLP, adequada para entradas vetoriais
    env,          # Seu ambiente com o novo espaço de observação
    verbose=0,
    learning_rate=1e-3,  # Taxa de aprendizado ajustada
    batch_size=64,       # Batch size pode ser menor, pois as entradas são mais simples
    buffer_size=50000,   # Buffer de replay maior para melhor generalização
    learning_starts=1000,
    gamma=0.99,          # Fator de desconto mantido para olhar para recompensas futuras
    target_update_interval=500,
    train_freq=4,
    gradient_steps=4,
    exploration_fraction=0.2,
    exploration_initial_eps=0.3,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[64, 64])  # Arquitetura mais simples da rede
)

"""

Treina por um total de 'TRAIN_STEPS' steps, cada um 
com 'TRAIN_STEPS // DIVISIONS' steps em cada treinamento.

Gera um total de 'DIVISIONS' modelos treinados,
numerados de '0' a 'DIVISIONS - 1'

"""
for train in range(0, DIVISIONS):
    print(f"Treino: {train+1} / {DIVISIONS}", end="\r")
    if train != 0: 
        model = DQN.load(DIR_NAME + "/" + str(train - 1))    
        model.set_env(env)
    
    model.learn(total_timesteps=TRAIN_STEPS//DIVISIONS)
    model.save(DIR_NAME + "/" + str(train))

env.close()
