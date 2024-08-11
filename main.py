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

# model = DQN(
#     "MlpPolicy",
#     env,
#     verbose=1,
#     tensorboard_log="./dqn_snake_tensorboard/",
#     buffer_size=50000,              # Aumenta o tamanho do buffer para mais diversidade na memória de replay
#     batch_size=64,                  # Reduzido para atualizações mais frequentes
#     learning_starts=5000,           # Aumentado para dar mais tempo de exploração antes de aprender
#     policy_kwargs=dict(net_arch=[32, 32]),  # Aumenta a complexidade da rede para capturar melhor o ambiente
#     exploration_initial_eps=0.8,    # Aumenta a exploração inicial para garantir ampla cobertura do espaço de estados
#     exploration_final_eps=0.1,      # Reduz o epsilon final para incentivar mais exploração ao longo do tempo
#     gamma=0.4,                     # Leve redução para focar mais em recompensas imediatas
#     target_update_interval=800,    # Atualizações mais frequentes para melhorar a estabilidade
#     train_freq=(50, "step"),         # Atualizações mais frequentes para melhorar o aprendizado
#     learning_rate=0.001,             # Taxa de aprendizado ajustada para ser mais conservadora
#     # gradient_steps=4                # Aumenta o número de atualizações por passo de tempo
# )

model = DQN(
    'MlpPolicy',
    env,
    verbose=1,
    # n_timesteps = 1.2e5,
    learning_rate = 4e-3,
    batch_size = 128,
    buffer_size = 10000,
    learning_starts = 1000,
    gamma = 0.75,
    target_update_interval = 600,
    train_freq = 16,
    gradient_steps = 8,
    exploration_fraction = 0.2,
    exploration_initial_eps=0.8,
    exploration_final_eps = 0.07,
    policy_kwargs = dict(net_arch=[256, 256])
)

# Treinamento do modelo
model.learn(total_timesteps=500000)

# Salvamento do modelo treinado
model.save("dqn_snake")

env.close()



# TESTES =============================================================================

env = DummyVecEnv([lambda: CobrinhaEnv(render_mode="human")])

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