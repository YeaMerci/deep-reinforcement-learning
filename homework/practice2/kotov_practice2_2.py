
"""
Пользуясь алгоритмом Кросс-Энтропии
обучить агента решать задачу Taxi-v3 из Gym.
Исследовать гиперпараметры алгоритма и выбрать лучшие.
"""


from typing import Optional, Any
from warnings import filterwarnings
import random
import time
import os


try:
    import torch.nn as nn
    import torch.nn.functional as F
    import torch
    import gym
    import numpy as np
    import wandb
except ModuleNotFoundError:
    os.system("pip install -r requirements.txt")


class CrossEntropyAgent(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_n: int,
                 depth: list[int, ...],
                 dropout: float = None,
                 batch_norm: bool = False
                 ):
        super().__init__()
        self.state_dim = state_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.action_n = action_n  # Кол-во действий
        self.depth = depth
        self.layers = [state_dim, *depth, action_n]
        self.head_block_index = len(self.layers) - 2

        self.__policy = nn.Sequential(*[
            self.__get_block(index, in_features, out_features)
            for index, (in_features, out_features) in enumerate(zip(self.layers, self.layers[1:]))
        ])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

    def __get_block(self,
                    index: int,
                    in_features: int,
                    out_features: int
                    ) -> nn.Sequential:
        _block = nn.Sequential(nn.Linear(in_features, out_features))
        if index != self.head_block_index:
            _block.append(nn.ReLU())
            if self.batch_norm:
                _block.append(nn.BatchNorm1d(out_features))
            if self.dropout:
                _block.append(nn.Dropout(self.dropout))
        return _block

    def forward(self, state: torch.FloatTensor | np.ndarray | list) -> torch.FloatTensor:
        if not isinstance(state, torch.FloatTensor):
            state = torch.FloatTensor(state)
        return self.__policy(state)

    def get_action(self, state: torch.Tensor | np.ndarray | list) -> int:
        """
        Возвращает действие агента A при условии состояния S
        :param state: текущее состояние
        :return: P(A | S)
        """
        logites = self.forward(state)
        probs = F.softmax(logites).detach().cpu().numpy()
        action = np.random.choice(self.action_n, p=probs)
        return action

    def update(self, elite_trajectories: list) -> None:
        elite_states = []
        elite_actions = []

        for trajectory in elite_trajectories:
            elite_states.extend(trajectory["states"])
            elite_actions.extend(trajectory["actions"])

        elite_states = torch.FloatTensor(elite_states)
        elite_actions = torch.LongTensor(elite_actions)

        preds = self.forward(elite_states)
        loss = self.criterion(preds, elite_actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


class SandBoxEnvironment:
    def __init__(self,
                 agent: CrossEntropyAgent,
                 env_name: Optional[str] = "maze-sample-5x5-v0",
                 render_mode: Optional[str] = "human",
                 render_last_iterations: Optional[int] = None,
                 **env_kwargs
                 ):
        self.env = gym.make(env_name, **env_kwargs)
        self.state_n = self.env.observation_space
        self.action_n = self.env.action_space.n
        self.agent = agent
        self.render_mode = render_mode
        self.render_last_iterations = render_last_iterations

        self.current_iteration = 0
        self.iterations = None
        self.start_time = None

        self.rewards = {
            "elite": [],
            "all": [],
            "q_params": []
        }

        self.total_rewards = lambda trajectories: np.sum([trajectory["total_rewards"] for trajectory in trajectories])
        self.iter_mean_rewards = lambda trajectories: np.round(np.mean(self.self.total_rewards(trajectories)), 2)
        self.time = lambda: round(time.time() - self.start_time, 2) if self.start_time else -1

    def __get_trajectory(self, trajectory_len: int = 500) -> dict:
        state = self.env.reset()  # Инициализация среды и получения начального состояния агента
        trajectory = {
            "states": [],
            "actions": [],
            "total_rewards": 0
        }

        for _ in range(trajectory_len):
            trajectory["states"].append(state)
            action = self.agent.get_action(state)  # Агент совершает действие P(A | S)
            trajectory["actions"].append(action)
            state, reward, terminated, truncated = self.env.step(action)  # Среда обновляется
            trajectory["total_rewards"] += reward

            if self.render_last_iterations:
                if self.current_iteration >= self.render_last_iterations:
                    time.sleep(0.25)
                    self.env.render(self.render_mode)
            if terminated:
                break
        return trajectory

    def get_trajectories(self, trajectory_len: int, trajectory_n: int) -> list:
        return [
            self.__get_trajectory(trajectory_len)
            for _ in range(trajectory_n)
        ]

    def get_elite_trajectories(self, trajectories: list, q_param: Optional[float] = 0.85) -> list:
        total_rewards = self.total_rewards(trajectories)
        quantile = np.quantile(total_rewards, q_param)
        return [
            trajectory
            for trajectory in trajectories
            if trajectory["total_rewards"] > quantile
        ]

    def on_epoch_end(self,
                     iteration: int,
                     trajectories: list,
                     elite_trajectories: list,
                     q_param: float,
                     ) -> None:

        mean_reward = self.iter_mean_rewards(trajectories)
        elite_mean_reward = self.iter_mean_rewards(elite_trajectories)

        wandb.log({"mean_reward": mean_reward, "elite_mean_reward": elite_mean_reward})
        self.rewards["all"].append(mean_reward)
        self.rewards["elite"].append(elite_mean_reward)

        print(f"\n| Iteration {iteration} of {self.iterations} | Time: {self.time()} sec |")
        print(f"| Mean Reward: {mean_reward} | Elite Mean Reward: {elite_mean_reward} |")
        print(f"| Quantile: {q_param} |\n")

    def q_sheduler(self,
                   q_param,
                   threshold: Optional[float] = 0.9,
                   gamma: Optional[float] = 0.001
                   ) -> float:
        if q_param < threshold and self.len(rewards["all"]) > 0:
            delta = np.abs(self.rewards["all"][-1] - self.rewards["elite"][-1])
            q_param -= delta * gamma
        return q_param

    def train(self,
              q_param: Optional[float] = 0.9,
              threshold: float = 0.86,
              gamma: Optional[float] = 0.001,
              iterations: Optional[int] = 20,
              trajectory_len: Optional[int] = 1000,
              trajectory_n: Optional[int] = 50
              ):
        print(f"\n| Training initialized  🚀|\n{time.asctime()}\n")
        self.start_time = time.time()
        self.iterations = iterations
        if self.render_last_iterations:
            self.render_last_iterations = iterations - self.render_last_iterations

        for iteration in range(iterations):
            self.rewards["q_params"].append(q_param)
            self.current_iteration = iteration
            trajectories = self.get_trajectories(trajectory_len, trajectory_n)
            elite_trajectories = self.get_elite_trajectories(trajectories, q_param)
            self.agent.update(elite_trajectories)
            self.on_epoch_end(iteration, trajectories, elite_trajectories, q_param)
            if gamma:
                q_param = self.q_sheduler(q_param, threshold, gamma)

        self.env.close()
        print(f"\n| Training finished  🚀|\nTotal time: {self.time()} sec\n")


def seed_everything(seed: Optional[int] = 42) -> int:
    random.seed(seed)
    np.random.seed(seed)
    return seed


def check_sources():
    path = os.path.join(os.getcwd(), "models")
    if not os.path.exists(path):
        os.mkdir(path)


def main() -> None:
    agent = CrossEntropyAgent(
        state_dim=8,
        depth=[10, 10],
        action_n=4,
        dropout=None,
        batch_norm=False
    )

    # Создаём среду
    env = SandBoxEnvironment(
        agent,  # Агент
        env_name="LunarLander-v2",  # Имя среды (v2)
        render_last_iterations=1  # Отрисовывать только последние None итерации
    )

    # Запуск обучения
    env.train(
        iterations=5,  # Кол-во итераций 50
        trajectory_n=10,  # Кол-во траекторий 500
        trajectory_len=15,  # Длина траекторий 170
        q_param=0.55,  # Квантиль 0.53
    )


if __name__ == "__main__":
    filterwarnings("ignore")  # Игнорировать предупреждения
    # seed_everything(42)  # Фиксация датчика случайных чисел
    check_sources()
    main()  # Запуск обучения
