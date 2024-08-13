


from stable_baselines3 import DQN
from CobrinhaEnv import CobrinhaEnv 
import gymnasium as gym
import os

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

DIR_NAME = "treinoTesteCallback"
TRAIN_STEPS = 50000
DIVISIONS = 10

NOT_ALLOW_REUSE_DIRS = False




class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, save_freq: int):
        super().__init__(0)             # 0 -> verbose 
        self.save_freq = save_freq
        self.log_dir = DIR_NAME
        self.num_saves = 0
        self.save_path = os.path.join(self.log_dir, "0")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
            #   mean_reward = np.mean(y[-100:])
            #   if self.verbose >= 1:
            #     print(f"Num timesteps: {self.num_timesteps}")
            #     print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

            #   # New best model, you could save the agent here
            #   if mean_reward > self.best_mean_reward:
            #       self.best_mean_reward = mean_reward
            #       # Example for saving best model
            #       if self.verbose >= 1:
            #         print(f"Saving new best model to {self.save_path}")
            #       self.model.save(self.save_path)
            self.model.save(self.save_path)
            self.num_saves += 1
            self.save_path = os.path.join(self.log_dir, str(self.num_saves))

        return True
    
if __name__ == "__main__":
    
    # Verifica se o diretório usado para salvar os modelos está em condições 
    if (not os.path.isdir(DIR_NAME)): os.makedirs(DIR_NAME)
    else: 
        print(f"Diretório '{DIR_NAME}' já existe. Usá-lo pode afetar o conteúdo pré existente.")
        if NOT_ALLOW_REUSE_DIRS: exit(0)

    # Declaração do ambiente 
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
    
    env = Monitor(env, DIR_NAME)
    callback = SaveOnBestTrainingRewardCallback(save_freq=5000)

    """

    Treina por um total de 'TRAIN_STEPS' steps, cada um 
    com 'TRAIN_STEPS // DIVISIONS' steps em cada treinamento.

    Gera um total de 'DIVISIONS' modelos treinados,
    numerados de '0' a 'DIVISIONS - 1'

    """
    # for train in range(0, DIVISIONS):
    #     print(f"Treino: {train+1} / {DIVISIONS}", end="\r")
    #     if train != 0: 
    #         model = DQN.load(DIR_NAME + "/" + str(train - 1))    
    #         model.set_env(env)
        
    #     model.learn(total_timesteps=TRAIN_STEPS//DIVISIONS)
    #     model.save(DIR_NAME + "/" + str(train))

    # env.close()
    
    model.learn(TRAIN_STEPS, callback=callback)
