# -----------------------------
# cliffwalk_rl.py
# 从零实现：Cliff Walking + SARSA + Q-learning
# 仅用 numpy / matplotlib，不依赖 RL 库
# -----------------------------

import numpy as np
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional

# =========================================================
# 1) 环境建模：CliffWalkEnv
# =========================================================
class CliffWalkEnv:
    """
    经典 4x12 悬崖行走环境。
    状态用一维索引表示：s = row * W + col
    动作编码：0=上，1=右，2=下，3=左
    奖励：普通步 -1；踩到悬崖 -100 且终止；到达终点终止。
    """
    ACTIONS = 4
    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

    def __init__(self, H: int = 4, W: int = 12):
        # 基本尺寸检查：至少 2x3
        assert H >= 2 and W >= 3, "Grid must be at least 2x3."
        self.H = H
        self.W = W
        # 起点在左下角、终点在右下角
        self.start = (H - 1, 0)
        self.goal = (H - 1, W - 1)
        # 悬崖位置：底行的 1..W-2 列
        self.cliff_cols = set(range(1, W - 1))
        # 当前位置（row, col）
        self._pos = self.start

    @property
    def n_states(self) -> int:
        # 总状态数 = H*W
        return self.H * self.W

    @property
    def n_actions(self) -> int:
        # 动作数 = 4
        return self.ACTIONS

    def reset(self, seed: Optional[int] = None) -> int:
        # 复位到起点；可选设随机种子，保证复现实验
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._pos = self.start
        return self._to_state(self._pos)  # 返回一维状态索引

    def step(self, action: int) -> Tuple[int, int, bool]:
        """执行一个动作，返回：(下一个状态, 即时奖励, 是否终止)"""
        # 当前坐标
        r, c = self._pos

        # 根据动作更新坐标，并做边界裁剪（越界则贴边）
        if action == self.UP:
            r = max(r - 1, 0)
        elif action == self.RIGHT:
            c = min(c + 1, self.W - 1)
        elif action == self.DOWN:
            r = min(r + 1, self.H - 1)
        elif action == self.LEFT:
            c = max(c - 1, 0)
        else:
            raise ValueError("Invalid action")

        # 得到新位置与默认奖励
        new_pos = (r, c)
        reward = -1               # 每步代价
        done = (new_pos == self.goal)  # 到终点即终止

        # 悬崖检测：若在底行且列落在 cliff 区间（且不是终点）
        if r == self.H - 1 and c in self.cliff_cols and not done:
            reward = -100         # 踩到悬崖重罚
            done = True           # 本实现采用“踩悬崖=终止”的变体

        # 刷新位置并返回
        self._pos = new_pos
        return self._to_state(new_pos), reward, done

    def _to_state(self, pos: Tuple[int, int]) -> int:
        # (row, col) -> 一维索引
        r, c = pos
        return r * self.W + c

    def _from_state(self, s: int) -> Tuple[int, int]:
        # 一维索引 -> (row, col)
        return divmod(s, self.W)

    def render_policy(self, Q: np.ndarray) -> List[str]:
        """
        把 Q 表的贪心动作以网格箭头渲染出来：
        ^ > v < 表示动作；S/G/# 表示起点/终点/悬崖
        """
        arrows = {0: '^', 1: '>', 2: 'v', 3: '<'}
        lines = []
        for r in range(self.H):
            row_cells = []
            for c in range(self.W):
                if (r, c) == self.start:
                    row_cells.append('S')
                elif (r, c) == self.goal:
                    row_cells.append('G')
                elif r == self.H - 1 and c in self.cliff_cols:
                    row_cells.append('#')
                else:
                    s = self._to_state((r, c))
                    a = int(np.argmax(Q[s]))  # 贪心动作
                    row_cells.append(arrows[a])
            lines.append(' '.join(row_cells))
        return lines

# =========================================================
# 2) ε-贪心策略 与 ε 调度
# =========================================================
def epsilon_greedy(Q: np.ndarray, s: int, epsilon: float, n_actions: int) -> int:
    # 以概率 ε 随机探索，否则选择 Q 最大的动作
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return int(np.argmax(Q[s]))

@dataclass
class RunConfig:
    # 训练相关超参数
    episodes: int = 500       # 训练回合数
    alpha: float = 0.1        # 学习率
    gamma: float = 0.9        # 折扣因子
    epsilon_start: float = 0.1   # 初始 ε
    epsilon_end: float = 0.1     # 结束 ε（与 start 相等表示不衰减）
    epsilon_decay: str = "linear" # ε 衰减策略："none" | "linear" | "exp"
    seed: int = 42               # 随机种子
    max_steps_per_episode: int = 10_000  # 单回合最大步数防死循环

def make_epsilon_schedule(cfg: RunConfig):
    # 生成一个函数 eps_fn(ep) 按回合号给出 ε
    if cfg.epsilon_decay == "none" or cfg.epsilon_start == cfg.epsilon_end:
        return lambda _: cfg.epsilon_start
    if cfg.epsilon_decay == "linear":
        def eps_fn(ep):
            t = ep / max(1, cfg.episodes - 1)
            return cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * t
        return eps_fn
    if cfg.epsilon_decay == "exp":
        # 指数衰减：start -> end
        k = np.log((cfg.epsilon_end + 1e-12) / (cfg.epsilon_start + 1e-12)) / max(1, cfg.episodes - 1)
        return lambda ep: float(cfg.epsilon_start * np.exp(k * ep))
    raise ValueError("Unknown epsilon_decay")

# =========================================================
# 3) 训练循环（**算法核心**）
#    下面两段分别是 SARSA 与 Q-learning
# =========================================================
def sarsa_train(env: CliffWalkEnv, cfg: RunConfig) -> Tuple[np.ndarray, List[float]]:
    # --- 这段是 **SARSA（on-policy）** ---
    random.seed(cfg.seed); np.random.seed(cfg.seed)
    Q = np.zeros((env.n_states, env.n_actions), dtype=float)  # Q 表
    returns = []                                              # 每回合累计回报
    eps_fn = make_epsilon_schedule(cfg)                       # ε 调度

    for ep in range(cfg.episodes):
        epsilon = eps_fn(ep)          # 当前回合 ε
        s = env.reset()               # 初始状态
        a = epsilon_greedy(Q, s, epsilon, env.n_actions)  # 先选一个动作（on-policy）
        G = 0.0
        steps = 0
        done = False
        while not done and steps < cfg.max_steps_per_episode:
            # 与环境交互：执行 A，得到 S'、R、终止标志
            s_next, r, done = env.step(a)
            # **SARSA 关键**：在 S' 上再次按 ε-greedy 选 A'
            a_next = epsilon_greedy(Q, s_next, epsilon, env.n_actions)
            # SARSA TD 目标：R + γ * Q(S',A')   （若终止则不加未来项）
            td_target = r + cfg.gamma * Q[s_next, a_next] * (0.0 if done else 1.0)
            # SARSA 更新：Q(S,A) ← Q(S,A) + α[目标 - 估计]
            Q[s, a] += cfg.alpha * (td_target - Q[s, a])
            # 状态动作前移
            s, a = s_next, a_next
            G += r
            steps += 1
        returns.append(G)
    return Q, returns

def qlearning_train(env: CliffWalkEnv, cfg: RunConfig) -> Tuple[np.ndarray, List[float]]:
    # --- 这段是 **Q-learning（off-policy）** ---
    random.seed(cfg.seed); np.random.seed(cfg.seed)
    Q = np.zeros((env.n_states, env.n_actions), dtype=float)
    returns = []
    eps_fn = make_epsilon_schedule(cfg)

    for ep in range(cfg.episodes):
        epsilon = eps_fn(ep)
        s = env.reset()
        G = 0.0
        steps = 0
        done = False
        while not done and steps < cfg.max_steps_per_episode:
            # 行为策略仍然 ε-greedy（用于探索）
            a = epsilon_greedy(Q, s, epsilon, env.n_actions)
            s_next, r, done = env.step(a)
            # **Q-learning 关键**：目标用 max_a' Q(S', a')（与行为动作无关）
            best_next = np.max(Q[s_next])
            td_target = r + cfg.gamma * best_next * (0.0 if done else 1.0)
            # Q-learning 更新：Q(S,A) ← Q(S,A) + α[目标 - 估计]
            Q[s, a] += cfg.alpha * (td_target - Q[s, a])
            s = s_next
            G += r
            steps += 1
        returns.append(G)
    return Q, returns

# =========================================================
# 4) 评估与曲线辅助
# =========================================================
def greedy_episode_return(env: CliffWalkEnv, Q: np.ndarray, max_steps: int = 10_000) -> int:
    """用贪心策略（ε=0）跑 1 回合，返回累计回报（便于 sanity check）。"""
    s = env.reset()
    total = 0
    for _ in range(max_steps):
        a = int(np.argmax(Q[s]))   # 纯贪心
        s, r, done = env.step(a)
        total += r
        if done:
            break
    return total

def moving_average(x: List[float], k: int = 20) -> np.ndarray:
    """简单的后向窗口移动平均，用于平滑学习曲线。"""
    if len(x) < 1: return np.array([])
    k = max(1, k)
    cumsum = np.cumsum(np.insert(np.array(x, dtype=float), 0, 0.0))
    ma = (cumsum[k:] - cumsum[:-k]) / k
    pad_left = np.full(k - 1, ma[0])
    return np.concatenate([pad_left, ma])

# =========================================================
# 5) 主流程：配置参数、训练、可视化
# =========================================================
def main():
    env = CliffWalkEnv(H=4, W=12)  # 经典 4x12

    # 关键超参数（与你最终实验一致）
    cfg = RunConfig(
        episodes=10000,
        alpha=0.1,
        gamma=0.9,
        epsilon_start=0.1,
        epsilon_end=0.1,      # 固定 ε=0.1（不衰减）
        epsilon_decay="none",
        seed=0
    )

    # 分别训练 SARSA 与 Q-learning
    Q_sarsa, ret_sarsa = sarsa_train(env, cfg)
    Q_ql, ret_ql = qlearning_train(env, cfg)

    # 贪心测试回报（单回合），常见值：Q-learning ~ -13，SARSA ~ -16~-18
    g_sarsa = greedy_episode_return(env, Q_sarsa)
    g_ql = greedy_episode_return(env, Q_ql)
    print(f"[Greedy single-episode return] SARSA: {g_sarsa} | Q-learning: {g_ql}")

    # 打印贪心策略网格
    print("\nGreedy policy (SARSA):")
    print("\n".join(env.render_policy(Q_sarsa)))
    print("\nGreedy policy (Q-learning):")
    print("\n".join(env.render_policy(Q_ql)))

    # 画移动平均曲线（MA@50）
    ma_sarsa = moving_average(ret_sarsa, 50)
    ma_ql = moving_average(ret_ql, 50)

    plt.figure(figsize=(9, 5))
    plt.plot(ma_sarsa, label="SARSA (MA@50)", linewidth=2)
    plt.plot(ma_ql, label="Q-learning (MA@50)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Average Return (50-episode moving average)")
    plt.title("Cliff Walking – Smoothed Learning Curve (MA@50)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
