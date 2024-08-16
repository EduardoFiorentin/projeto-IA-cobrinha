from stable_baselines3 import DQN
from CobrinhaEnv import CobrinhaEnv 
import gymnasium as gym
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env


ENV_ID = "Cobrinha"
ENV_ENTRY_POINT = 'CobrinhaEnv:CobrinhaEnv'

gym.register(
    id=ENV_ID,
    entry_point=ENV_ENTRY_POINT
)


# Configurações gerais 
DIR_NAME =              "testRefatoring"      # Diretório onde são salvos modelos intermediários e graficos 
TRAIN_STEPS =           30000                 # Steps de treinamento
MODEL_SAVE_FREQ =       5000                   # Frequência de salvamento de modelos 
NOT_ALLOW_REUSE_DIRS =  False                  # impedir que arquivos com modelos salvos sejam sobrescritos
ENV_RENDER_MODE =       "human"                 #modo de renderização

# Configurações da avaliação do treinamento 
EVAL_LOG_FILE =         os.path.join(DIR_NAME, "evaluations.txt")   # arquivo onde são registradas as avaliações dos modelos 
EVAL_FREQUENCY =        MODEL_SAVE_FREQ
NUM_EVAL_EPISODES =     100                     # numero de episodios de cada avaliação
PLOT_FREQUENCY =        10000                   # Frequência de plot do gráfico de evolução (valor deve ser múltiplo de MODEL_SAVE_FREQ)

LEARN_ENV = gym.make(ENV_ID, render_mode=ENV_RENDER_MODE)
TEST_ENV = gym.make(ENV_ID, render_mode = ENV_RENDER_MODE)


class SaveOnTrainStepsNumCallback(BaseCallback):
    def __init__(self, save_freq: int):
        super().__init__(0)  # 0 -> verbose
        self.save_freq = save_freq
        self.log_dir = DIR_NAME
        self.num_saves = 1
        self.save_path =  os.path.join(self.log_dir, str(self.num_saves))
        self.reward_log = []  # Lista para armazenar recompensas médias
        self.steps_log = []  # Lista para armazenar o número de passos

    def plot_performance(self) -> None:
        # if self.save_path is not None:
        #     os.makedirs(self.save_path, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps_log, self.reward_log, label="Recompensa Média")
        plt.xlabel("Número de Passos")
        plt.ylabel("Recompensa Média")
        plt.title("Desempenho do Modelo Durante o Treinamento")
        plt.legend()
        plt.grid(True)
        # Salvar a figura no diretório especificado
        save_path = os.path.join(DIR_NAME, f"performance_plot{self.num_saves}.png")
        plt.savefig(save_path)
        print(f"Gráfico salvo em {save_path}")
        

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            print(f"Salvando modelo: {self.save_path}")
            self.model.save(self.save_path)
            
            # Teste do modelo atual
            local_model = DQN.load(self.save_path)
            local_model.exploration_rate = 0
            
            file = open(EVAL_LOG_FILE, "a+")
            
            print(f"rodada de testes... {self.save_path}")
            file.write("-"*30+"\n")
            file.write(self.save_path+"\n")
            
            
            # Configurar de acordo com o ambiente 
            local_env = TEST_ENV
            local_env.reset()
            
            episodes_reward_list = []
            episode_steps_num = []
            
            for i in range(0, NUM_EVAL_EPISODES):
                obs, _ = local_env.reset()
                print("Rodada: ", i)
                episode_rew = 0
                episode_steps = 0
                while True:
                    action, _states = local_model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, _ = local_env.step(action)
                    local_env.render()

                    episode_steps += 1
                    episode_rew += reward
                    if done:
                        break
                
                episode_steps_num.append(episode_steps)
                episodes_reward_list.append(episode_rew)
            
            # Média da recompensa por passo
            mean_reward = sum(episodes_reward_list) / sum(episode_steps_num)
            self.reward_log.append(mean_reward)
            self.steps_log.append(self.n_calls)
                
                
            file.write(f"Recompensa média para {NUM_EVAL_EPISODES} rodadas: {sum(episodes_reward_list) / sum(episode_steps_num)}"+"\n")
            file.close()
            
            if self.n_calls % PLOT_FREQUENCY == 0: 
                self.plot_performance()
                print(f"Grafico de {self.n_calls} steps plotado!")
            
            print("Testes Finalizados")
            
            
            self.num_saves += 1
            self.save_path = os.path.join(self.log_dir, str(self.num_saves))

        return True
    


if __name__ == "__main__":
    if not os.path.isdir(DIR_NAME): 
        os.makedirs(DIR_NAME)
    else: 
        print(f"Diretório '{DIR_NAME}' já existe. Usá-lo pode afetar o conteúdo pré-existente.")
        if NOT_ALLOW_REUSE_DIRS: exit(0)

    gym.register(
        id=ENV_ID,
        entry_point=ENV_ENTRY_POINT
    )
    
    env = Monitor(LEARN_ENV, DIR_NAME)
    
    # Callback de salvamento
    save_callback = SaveOnTrainStepsNumCallback(save_freq=MODEL_SAVE_FREQ)

    # Modelo DQN
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
    
    # Treinamento com callbacks
    model.learn(total_timesteps=TRAIN_STEPS, callback=[save_callback])
    
    # plot_performance(save_callback)
