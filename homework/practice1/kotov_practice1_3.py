
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
        self.__stochastic_policy = self.__init_uniform_weights()  # Модель - матрица P (policy)
        self.deterministic_policy = self.__empty_weights()

    def __empty_weights(self) -> np.ndarray:
        return np.zeros(self.__shape)

    def __init_uniform_weights(self) -> np.ndarray:
        # Равномерная инициализация начальной матрицы P (policy)
        return np.full(self.__shape, self.__uniform_value)

    def get_action(self, state: int, mode: Literal["deterministic", "stochastic"]) -> int:
        """
        Возвращает действие агента A при условии состояния S
        :param state: текущее состояние
        :return: P(A | S)
        """
        if mode == "stochastic":
            action = np.random.choice(np.arange(self.action_n), p=self.__stochastic_policy[state])
        else:
            action = np.argmax(self.__deterministic_policy[state])
        return action

    @property
    def deterministic_policy(self) -> np.ndarray:
        return self.__deterministic_policy

    @deterministic_policy.setter
    def deterministic_policy(self, policy: np.ndarray) -> None:
        self.__deterministic_policy = policy

    def __get_deterministic_policy(self) -> np.ndarray:
        sample = self.__empty_weights()
        for state in range(self.state_n):
            action = self.get_action(state, "stochastic")
            sample[state][action] = 1
        return sample

    def sample_deterministic_policies(self, num: int) -> np.ndarray:
        policies = []
        for _ in range(num):
            sample = self.__get_deterministic_policy()
            policies.append(sample)
        return policies

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
                new_policy[state] = self.__stochastic_policy[state]
        return new_policy

    def policy_smoothing(self, elite_trajectories: list, decay: float) -> np.ndarray:
        new_policy = self.simple_update(elite_trajectories)
        return decay * self.__stochastic_policy + (1 - decay) * new_policy

    def update(self,
               elite_trajectories: list,
               smoothing: Optional[Literal["laplas", "policy"]] = None,
               decay: Optional[float] = 0.7,
               alpha: Optional[int] = 1
               ) -> None:
        if not smoothing:
            self.__stochastic_policy = self.simple_update(elite_trajectories)
        elif smoothing == "laplas":
            self.__stochastic_policy = self.simple_update(elite_trajectories, alpha)
        elif smoothing == "policy":
            assert 0.0 <= decay <= 1.0
            self.__stochastic_policy = self.policy_smoothing(elite_trajectories, decay)
        else:
            raise ValueError(
                f"Expected argument smoothing must be Literal['laplas', 'policy'], but got {smoothing}"
            )

    def save(self, filename: Optional[str] = "checkpoint_weights") -> bool:
        filepath = self.define_filepath(filename)
        dump(self.__stochastic_policy, filepath)
        return True

    def load(self, filename: Optional[str] = "checkpoint_weights") -> None:
        filepath = self.define_filepath(filename)
        self.__stochastic_policy = load(filepath)


class SandBoxEnvironment:
    def __init__(self,
                 agent: Any,
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
        self.rewards = []

        self.is_elite = lambda me, quantile: True if me > quantile else False
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
            action = self.agent.get_action(state, mode="deterministic")  # Агент совершает действие P(A | S)
            trajectory["actions"].append(action)
            state, reward, terminated, truncated = self.env.step(action)  # Среда обновляется
            trajectory["rewards"].append(reward)

            if self.render_last_iterations:
                if self.current_iteration >= self.render_last_iterations:
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

    def get_elite_trajectories(self, policies_trajectories: list, policies_me: list, q_param: Optional[float] = 0.85) -> list:
        quantile = np.quantile(policies_me, q_param)
        elite_trajectories = []
        elite_policies = [
            policy_trajectories
            for me_reward, policy_trajectories in zip(policies_me, policies_trajectories)
            if self.is_elite(me_reward, quantile)
        ]
        for trajectories in elite_policies:
            for trajectory in trajectories:
                elite_trajectories.append(trajectory)
        return elite_trajectories

    @staticmethod
    def policies_me_eval(policies_trajectories: list) -> list:
        policies_me_rewards = []
        for policy_trajectories in policies_trajectories:
            me_policy_reward = np.mean([np.sum(trajectory["rewards"]) for trajectory in policy_trajectories])
            policies_me_rewards.append(me_policy_reward)
        return policies_me_rewards

    def on_epoch_end(self,
                     iteration: int,
                     iterations: int,
                     policies_me: list,
                     q_param: float,
                     save_threshold: Optional[float] = None
                     ) -> None:
        policies_me = np.round(np.mean(policies_me), 2)
        self.rewards.append(policies_me)
        print(f"\n| Iteration {iteration} of {iterations} | Time: {self.time()} sec |")
        print(f"| Me[P] rewards: {policies_me} |")
        print(f"| Quantile: {q_param} |\n")

        if save_threshold and not self.saved:
            if policies_me >= save_threshold:
                self.saved = self.agent.save(
                    filename=f"policies_me={policies_me}_iteration={iteration}"
                )

    def plot(self):
        plt.plot(self.rewards)
        plt.xlabel("Iterations")
        plt.ylabel("Me[P] rewards")
        plt.show()

    def train(self,
              num_policies: int,
              smoothing: Optional[Literal["laplas", "policy"]] = None,
              q_param: Optional[float] = 0.9,
              alpha: Optional[int] = 1,
              decay: Optional[float] = 0.7,
              iterations: Optional[int] = 20,
              trajectory_len: Optional[int] = 1000,
              trajectory_n: Optional[int] = 50,
              save_threshold: Optional[float] = None,
              plot: Optional[bool] = True
              ):
        print(f"\n| Training initialized  🚀|\n{time.asctime()}\n")
        self.start_time = time.time()

        if self.render_last_iterations:
            self.render_last_iterations = iterations - self.render_last_iterations

        for iteration in range(iterations):
            policies_trajectories = []

            for policy in self.agent.sample_deterministic_policies(num_policies):
                self.agent.deterministic_policy = policy
                self.current_iteration = iteration
                trajectories = self.get_trajectories(trajectory_len, trajectory_n)
                policies_trajectories.append(trajectories)

            policies_me = self.policies_me_eval(policies_trajectories)
            elite_trajectories = self.get_elite_trajectories(policies_trajectories, policies_me, q_param)

            self.agent.update(elite_trajectories, smoothing, decay, alpha)
            self.on_epoch_end(iteration, iterations, policies_me, q_param, save_threshold)

        self.env.close()
        print(f"\n| Training finished  🚀|\nTotal time: {self.time()} sec\n")
        self.agent.save(
            filename=f"policies_me={np.round(np.mean(policies_me), 2)}_iteration={self.current_iteration}"
        )
        if plot:
            self.plot()


def seed_everything(seed: Optional[int] = 42) -> int:
    random.seed(seed)
    np.random.seed(seed)
    return seed


def main() -> None:
    # Создаём среду
    env = SandBoxEnvironment(
        CrossEntropyAgent,  # Агент
        sleep=0.35,  # Время задержки между вывода картинки
        env_name="Taxi-v3",  # Имя среды
        render_last_iterations=None  # Отрисовывать только последние N итерации
    )

    # Запуск обучения
    env.train(
        num_policies=800,
        iterations=150,  # Кол-во итераций
        trajectory_n=100,  # Кол-во траекторий
        trajectory_len=190,  # Длина траекторий
        q_param=0.6,  # Квантиль
        save_threshold=None,
        smoothing="policy",
        decay=0.5
    )


if __name__ == "__main__":
    filterwarnings("ignore")  # Игнорировать предупреждения
    seed_everything(42)  # Фиксация датчика случайных чисел
    main()  # Запуск обучения
