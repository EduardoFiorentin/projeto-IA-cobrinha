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

# model = DQN(
#     'MlpPolicy',
#     env,
#     verbose=1,
#     # n_timesteps = 1.2e5,
#     learning_rate = 1e-3,
#     batch_size = 128,
#     buffer_size = 10000,
#     learning_starts = 2000,
#     gamma = 0.5,
#     target_update_interval = 200,
#     train_freq = 4,
#     gradient_steps = 8,
#     exploration_fraction = 0.2,
#     exploration_initial_eps=0.3,
#     exploration_final_eps = 0.07,
#     policy_kwargs = dict(net_arch=[256, 256])
# )

model = DQN(
    'MlpPolicy',  # Política MLP, adequada para entradas vetoriais
    env,          # Seu ambiente com o novo espaço de observação
    verbose=1,
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

# Treinamento do modelo
model.learn(total_timesteps=10000)

# Salvamento do modelo treinado
model.save("dqn_snake")

env.close()



# TESTES =============================================================================


        
# Verificar se a imagem vista pela IA está sendo atualizada 

# Trocar a visão da ia para distância e direção até a comida