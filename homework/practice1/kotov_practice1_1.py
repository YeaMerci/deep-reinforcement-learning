
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
    import gym
    import numpy as np
    import matplotlib.pyplot as plt
    from joblib import dump, load
except ModuleNotFoundError:
    os.system("pip install -r requirements.txt")
    print(
        "Приношу извинения за дополнительные библиотеки, "
        "если таковые были 🙃\n Зависимости установленны, перезапустите файл ✅"
    )


class CrossEntropyAgent:
    def __init__(self, state_n: int, action_n: int):
        self.define_filepath = lambda filename: os.path.join(os.getcwd(), "models", f"{filename}.joblib")
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
        self.__policy = new_policy.copy()

    def save(self, filename: Optional[str] = "checkpoint_weights") -> bool:
        filepath = self.define_filepath(filename)
        dump(self.__policy, filepath)
        return True

    def load(self, filename: Optional[str] = "checkpoint_weights") -> None:
        filepath = self.define_filepath(filename)
        self.__policy = load(filepath)


class SandBoxEnvironment:
    def __init__(self,
                 agent: Any | CrossEntropyAgent,
                 env_name: Optional[str] = "maze-sample-5x5-v0",
                 render_mode: Optional[str] = "human",
                 render_threshold: Optional[float] = None,
                 render_last_iterations: Optional[int] = None,
                 sleep: Optional[float] = 0.5):

        self.env = gym.make(env_name)
        self.state_n = self.env.observation_space.n
        self.action_n = self.env.action_space.n
        self.agent = agent(self.state_n, self.action_n)

        self.render_mode = render_mode
        self.render_last_iterations = render_last_iterations
        self.render_threshold = render_threshold
        self.current_iteration = 0
        self.sleep = sleep

        self.saved = None
        self.start_time = None
        self.rewards = {
            "elite": [],
            "all": [],
            "q_params": []
        }

        self.is_elite = lambda trajectory, quantile: True if np.sum(trajectory["rewards"]) > quantile else False
        self.total_reward = lambda trajectories: list(map(lambda trajectory: np.sum(trajectory["rewards"]), trajectories))
        self.mean_total_reward = lambda trajectories: np.round(np.mean(self.total_reward(trajectories)), 2)
        self.time = lambda: round(time.time() - self.start_time, 2) if self.start_time else -1

    def __get_trajectory(self, trajectory_len: int = 500) -> dict:
        state = self.env.reset()  # Инициализация среды и получения начального состояния агента
        trajectory = {
            "states": [],
            "actions": [],
            "rewards": []
        }

        for _ in range(trajectory_len):
            trajectory["states"].append(state)
            action = self.agent.get_action(state)  # Агент совершает действие P(A | S)
            trajectory["actions"].append(action)
            state, reward, terminated, truncated = self.env.step(action)  # Среда обновляется
            trajectory["rewards"].append(reward)

            if self.render_last_iterations:
                if self.current_iteration >= self.render_last_iterations:
                    time.sleep(self.sleep)
                    self.env.render(self.render_mode)
            elif len(self.rewards["all"]) > 0 and self.render_threshold:
                if self.rewards["all"][-1] >= self.render_threshold:
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

    def on_epoch_end(self,
                     iteration: int,
                     iterations: int,
                     trajectories: list,
                     elite_trajectories: list,
                     q_param: float,
                     save_threshold: Optional[float] = None
                     ) -> None:

        mean_reward = self.mean_total_reward(trajectories)
        elite_mean_reward = self.mean_total_reward(elite_trajectories)
        self.rewards["all"].append(mean_reward)
        self.rewards["elite"].append(elite_mean_reward)

        print(f"\n| Iteration {iteration} of {iterations} | Time: {self.time()} sec |")
        print(f"| Mean Reward: {mean_reward} | Elite Mean Reward: {elite_mean_reward} |")
        print(f"| Quantile: {q_param} |\n")

        if save_threshold and not self.saved:
            if mean_reward >= save_threshold:
                self.saved = self.agent.save(
                    filename=f"checkpoint_mean_reward={mean_reward}_iteration={iteration}"
                )

    def plot(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.plot(self.rewards["all"])
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Mean rewards")

        ax2.plot(self.rewards["elite"])
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Mean elit rewards")

        ax3.plot(self.rewards["q_params"])
        ax3.set_xlabel("Iterations")
        ax3.set_ylabel("Quantile")
        plt.show()

    def q_sheduler(self,
                   q_param,
                   threshold: Optional[float] = 0.9,
                   gamma: Optional[float] = 0.001
                   ) -> float:
        if q_param < threshold and (self.rewards["all"][-1] >= 0 and self.rewards["elite"][-1] >= 0):
            delta = self.rewards["all"][-1] - self.rewards["elite"][-1]
            q_param -= delta * gamma
        return q_param

    def train(self,
              q_param: Optional[float] = 0.9,
              iterations: Optional[int] = 20,
              trajectory_len: Optional[int] = 1000,
              trajectory_n: Optional[int] = 50,
              gamma: Optional[float] = 0.001,
              save_threshold: Optional[float] = None,
              plot: Optional[bool] = True
              ):
        print(f"\n| Training initialized  🚀|\n{time.asctime()}\n")
        self.start_time = time.time()

        if self.render_last_iterations:
            self.render_last_iterations = iterations - self.render_last_iterations

        for iteration in range(iterations):
            self.rewards["q_params"].append(q_param)
            self.current_iteration = iteration
            trajectories = self.get_trajectories(trajectory_len, trajectory_n)
            elite_trajectories = self.get_elite_trajectories(trajectories, q_param)
            self.agent.update(elite_trajectories)
            self.on_epoch_end(iteration, iterations, trajectories, elite_trajectories, q_param, save_threshold)
            if gamma:
                q_param = self.q_sheduler(q_param, gamma=gamma)

        self.env.close()
        print(f"\n| Training finished  🚀|\nTotal time: {self.time()} sec\n")
        if plot:
            self.plot()


def seed_everything(seed: Optional[int] = 42) -> int:
    random.seed(seed)
    np.random.seed(seed)
    return seed


def check_sources():
    path = os.path.join(os.getcwd(), "models")
    if not os.path.exists(path):
        os.mkdir(path)


def main() -> None:
    # Создаём среду
    env = SandBoxEnvironment(
        CrossEntropyAgent,  # Агент
        sleep=0.15,  # Время задержки между вывода картинки
        env_name="Taxi-v3",  # Имя среды
        render_last_iterations=1  # Отрисовывать только последние None итерации
    )

    # Загрузка policy
    # env.agent.load("practice1_1_mean_reward=7.15_iteration=19")

    # Запуск обучения
    env.train(
        iterations=50,  # Кол-во итераций 50
        trajectory_n=500,  # Кол-во траекторий 500
        trajectory_len=170,  # Длина траекторий 170
        q_param=0.53,  # Квантиль 0.53
        save_threshold=None  # Сохранять модель > 7. mean total reward
    )


if __name__ == "__main__":
    filterwarnings("ignore")  # Игнорировать предупреждения
    seed_everything(42)  # Фиксация датчика случайных чисел
    check_sources()
    main()  # Запуск обучения
