
"""
Реализовать алгоритм Кросс-Энтропии с двумя типами сглаживания,
указанными в лекции 1.
При выбранных в пункте 1 гиперпараметров сравнить их результаты с результатами алгоритма без сглаживания.
"""


from typing import Optional, Any, Literal
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

    def simple_update(self, elite_trajectories: list, alpha: Optional[int] = None) -> np.ndarray:
        new_policy = self.__empty_weights()

        for trajectory in elite_trajectories:
            for state, action in zip(trajectory["states"], trajectory["actions"]):
                new_policy[state][action] += 1

        for state in range(self.state_n):
            new_policy[state] += alpha if alpha else 0
            if np.sum(new_policy[state]) > 0:
                new_policy[state] /= np.sum(new_policy[state])
            else:
                new_policy[state] = self.__policy[state]
        return new_policy

    def policy_smoothing(self, elite_trajectories: list, decay: float) -> np.ndarray:
        new_policy = self.simple_update(elite_trajectories)
        return decay * self.__policy + (1 - decay) * new_policy

    def update(self,
               elite_trajectories: list,
               smoothing: Optional[Literal["laplas", "policy"]] = None,
               decay: Optional[float] = 0.7,
               alpha: Optional[int] = 1
               ) -> None:
        if not smoothing:
            self.__policy = self.simple_update(elite_trajectories)
        elif smoothing == "laplas":
            self.__policy = self.simple_update(elite_trajectories, alpha)
        elif smoothing == "policy":
            assert 0.0 <= decay <= 1.0
            self.__policy = self.policy_smoothing(elite_trajectories, decay)
        else:
            raise ValueError(
                f"Expected argument smoothing must be Literal['laplas', 'policy'], but got {smoothing}"
            )

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
        self.lower = None

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
                   q_param: float,
                   trajectory_len: int,
                   min_threshold: Optional[float] = 0.55,
                   max_threshold: Optional[float] = 0.9,
                   gamma: Optional[float] = 0.001
                   ) -> float:
        delta = (self.rewards["all"][-1] - self.rewards["elite"][-1]) / trajectory_len
        if delta <= 0 and q_param < max_threshold and not self.lower:
            q_param -= delta * gamma
        elif delta >= 0 and q_param > min_threshold:
            self.lower = True
            q_param -= delta * gamma
        else:
            self.lower = False
        return q_param

    def train(self,
              smoothing: Optional[Literal["laplas", "policy"]] = None,
              q_param: Optional[float] = 0.9,
              alpha: Optional[int] = 1,
              decay: Optional[float] = 0.7,
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
            alpha = 0 if iteration >= (iterations // 2) else alpha
            self.rewards["q_params"].append(q_param)
            self.current_iteration = iteration
            trajectories = self.get_trajectories(trajectory_len, trajectory_n)
            elite_trajectories = self.get_elite_trajectories(trajectories, q_param)
            self.agent.update(elite_trajectories, smoothing, decay, alpha)
            self.on_epoch_end(iteration, iterations, trajectories, elite_trajectories, q_param, save_threshold)
            q_param = self.q_sheduler(q_param, trajectory_len, gamma=gamma)

        self.env.close()
        print(f"\n| Training finished  🚀|\nTotal time: {self.time()} sec\n")
        if plot:
            self.plot()


def seed_everything(seed: Optional[int] = 42) -> int:
    random.seed(seed)
    np.random.seed(seed)
    return seed


def main(model: Optional[str]) -> None:
    smoothing = model[0]
    filename = model[1]
    print(f"\n[{smoothing if smoothing else 'Not'}] using smoothing ⚡️\n")

    # Создаём среду
    env = SandBoxEnvironment(
        CrossEntropyAgent,  # Агент
        sleep=0.25,  # Время задержки между вывода картинки
        env_name="Taxi-v3",  # Имя среды
        render_last_iterations=1  # Отрисовывать только последние K итерации
    )

    # env.agent.load(filename)  # Загрузка весов

    # Запуск обучения
    env.train(
        iterations=50,  # Кол-во итераций
        trajectory_n=500,  # Кол-во траекторий
        trajectory_len=170,  # Длина траекторий
        q_param=0.52,  # Квантиль
        save_threshold=None,
        smoothing=smoothing,
        decay=0.5,
        alpha=1
    )


if __name__ == "__main__":
    filterwarnings("ignore")  # Игнорировать предупреждения
    seed_everything(42)  # Фиксация датчика случайных чисел

    models = (
        ("laplas", "laplas_mean_reward=4.25_iteration=33"),  # С Лапласом
        ("policy", "policy_smooth_mean_reward=7.07_iter=31")  # С Policy
    )

    for model in models:
        main(model)  # Запуск обучения
