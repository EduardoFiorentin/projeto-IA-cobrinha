from stable_baselines3 import DQN
from CobrinhaEnv import CobrinhaEnv 

from stable_baselines3.common.vec_env import DummyVecEnv


env = DummyVecEnv([lambda: CobrinhaEnv(render_mode="human")])


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
model.learn(total_timesteps=300000)

# Salvamento do modelo treinado
model.save("dqn_snake2")

env.close()
