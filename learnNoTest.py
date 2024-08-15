from stable_baselines3 import DQN
import gymnasium as gym
import os

DIR_NAME = "novaCobrinha"
MODEL_NAME = "nova_cobrinha"
TRAIN_STEPS = 50000

NOT_ALLOW_REUSE_DIRS = True

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

model = DQN(
    'MlpPolicy',  # Política MLP, adequada para entradas vetoriais
    env,          # Seu ambiente com o novo espaço de observação
    verbose=3,
    learning_rate=1e-3,  # Taxa de aprendizado ajustada
    batch_size=64,       # Batch size pode ser menor, pois as entradas são mais simples
    buffer_size=50000,   # Buffer de replay maior para melhor generalização
    learning_starts=1000,
    gamma=0.99,          # Fator de desconto mantido para olhar para recompensas futuras
    target_update_interval=500,
    train_freq=4,
    gradient_steps=4,
    exploration_fraction=0.2,
    exploration_initial_eps=0.5,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[64, 64])  # Arquitetura mais simples da rede
)


model.learn(total_timesteps=TRAIN_STEPS)
model.save(DIR_NAME + "/" + MODEL_NAME)

env.close()
