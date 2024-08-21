from typing import Optional, Any
from warnings import filterwarnings
import random
import time
import os
import gym
import gym_maze
import numpy as np
import matplotlib.pyplot as plt


class CrossEntropyAgent:
    def __init__(self, state_n: int, action_n: int):
        self.define_filepath = lambda filename: os.path.join(os.getcwd(), filename)
        self.state_n = state_n  # Кол-во состояний
        self.action_n = action_n  # Кол-во действий
        self.__shape = (self.state_n, self.action_n)  # Размеры матрицы P (policy)
        self.__uniform_value = 1 / self.action_n  # Значение равномерного распределения действий в состояниях
        self.__policy = self.__init_weights()  # Модель - матрица P (policy)

    def __empty_weights(self) -> np.ndarray:
        return np.zeros(self.__shape)

    def __init_weights(self) -> np.ndarray:
        # Равномерная инициализация начальной матрицы P (policy)
        return np.full(self.__shape, self.__uniform_value)

    def get_action(self, state: int) -> int:
        """
        Возвращает действие агента A при условии состояния S
        :param state: текущее состояние
        :return: P(A | S)
        """
        return np.random.choice(np.arange(self.action_n), p=self.__policy[state])

    def update(self, elite_trajectories: list) -> None:
        new_policy = self.__empty_weights()
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory["states"], trajectory["actions"]):
                new_policy[state][action] += 1
        for state in range(self.state_n):
            if np.sum(new_policy[state]) > 0:
                new_policy[state] /= np.sum(new_policy[state])
            else:
                new_policy[state] = self.__policy[state].copy()
        self.__policy = new_policy

    def save(self, filename: Optional[str] = "checkpoint_weights.npz") -> bool:
        filepath = self.define_filepath(filename)
        with open(filepath, "wb") as f:
            np.save(f, self.__policy)
        return True

    def load(self, filename: Optional[str] = "checkpoint_weights.npz") -> None:
        filepath = self.define_filepath(filename)
        with open(filepath, "rb") as f:
            self.__policy = np.load(f)


class SandBoxEnvironment:
    def __init__(self,
                 agent: Any | CrossEntropyAgent,
                 env_name: Optional[str] = "maze-sample-5x5-v0",
                 actions: Optional[list] = None,
                 render_mode: Optional[str] = "human",
                 render_threshold: Optional[float] = 0.0,
                 sleep: Optional[float] = 0.5):
        self.agent = agent
        self.actions = actions if actions else ["N", "E", "S", "W"]
        self.saved = None
        self.env = gym.make(env_name)
        self.render_mode = render_mode
        self.render_threshold = render_threshold
        self.sleep = sleep
        self.is_elite = lambda trajectory, quantile: trajectory if np.sum(trajectory["rewards"]) > quantile else None
        self.total_reward = lambda trajectories: list(map(lambda trajectory: np.sum(trajectory["rewards"]), trajectories))
        self.mean_total_reward = lambda trajectories: np.round(np.mean(self.total_reward(trajectories)), 2)
        self.start_time = None
        self.time = lambda: round(time.time() - self.start_time, 2) if self.start_time else -1
        self.rewards = {
            "elite": [],
            "all": []
        }

    def _init_env(self) -> int:
        observation = self.env.reset()
        return self.get_state(observation)

    def get_state(self, observation: np.ndarray | tuple) -> int:
        """
        Кодирует текущее состояние observation из вектора в скаляр,
        где observation - tuple(x, y),
        где x и y - координаты по оси абсцисс и ординат
        в среде (декартовой системе координат)
        :param observation: текущее состояние
        :return: vector(x, y) -> scalar
        """
        return int(np.sqrt(self.agent.state_n) * observation[0] + observation[1])

    def __get_trajectory(self, trajectory_len: int = 500) -> dict:
        state = self._init_env()  # Инициализация среды и получения начального состояния агента
        trajectory = {
            "states": [],
            "actions": [],
            "rewards": []
        }

        for _ in range(trajectory_len):
            action = self.agent.get_action(state)  # Агент совершает действие P(A | S)
            trajectory["states"].append(state)
            observation, reward, terminated, truncated = self.env.step(self.actions[action])  # Среда обновляется
            trajectory["actions"].append(action)
            state = self.get_state(observation)  # Переход агента в новое состояние
            trajectory["rewards"].append(reward)

            if self.render_mode and self.render_threshold:
                if self.render_threshold <= np.mean(self.rewards["all"][-5:]):
                    time.sleep(self.sleep)
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
        total_rewards = self.total_reward(trajectories)
        quantile = np.quantile(total_rewards, q_param)
        return [
            trajectory
            for trajectory in trajectories
            if self.is_elite(trajectory, quantile)
        ]

    def info(self,
             iteration: int,
             iterations: int,
             trajectories: list,
             elite_trajectories: list,
             save_threshold: Optional[float] = None
             ) -> None:
        mean_reward = self.mean_total_reward(trajectories)
        elite_mean_reward = self.mean_total_reward(elite_trajectories)
        self.rewards["all"].append(mean_reward)
        self.rewards["elite"].append(elite_mean_reward)
        print(f"\n| Iteration {iteration} of {iterations} | Time: {self.time()} sec |")
        print(f"| Mean Reward: {mean_reward} | Elite Mean Reward: {elite_mean_reward} |\n")
        if save_threshold and not self.saved:
            if mean_reward >= save_threshold:
                self.saved = self.agent.save(filename=f"checkpoint_mean_reward={mean_reward}_iteration={iteration}.npz")

    def plot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(self.rewards["all"])
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Mean rewards")

        ax2.plot(self.rewards["elite"])
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Mean elit rewards")
        plt.show()

    def train(self,
              q_param: Optional[float] = 0.9,
              iterations: Optional[int] = 20,
              trajectory_len: Optional[int] = 1000,
              trajectory_n: Optional[int] = 50,
              save_threshold: Optional[float] = None,
              plot: Optional[bool] = True
              ):
        self.start_time = time.time()
        print(f"\n| Training initialized  🚀|\n{time.asctime()}\n")
        for iteration in range(iterations):
            trajectories = self.get_trajectories(trajectory_len, trajectory_n)
            elite_trajectories = self.get_elite_trajectories(trajectories, q_param)
            self.agent.update(elite_trajectories)
            self.info(iteration, iterations, trajectories, elite_trajectories, save_threshold)
        print(f"\n| Training finished  🚀|\nTotal time: {self.time() // 60} min\n")
        if plot:
            self.plot()


def main() -> None:
    agent = CrossEntropyAgent(state_n=25, action_n=4)
    env = SandBoxEnvironment(agent, sleep=0.25, render_threshold=0)
    env.train(iterations=15, trajectory_n=100, q_param=0.85, trajectory_len=100, save_threshold=None)


if __name__ == "__main__":
    filterwarnings("ignore")
    main()
