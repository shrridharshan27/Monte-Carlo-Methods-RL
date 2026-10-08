"""
Monte Carlo Methods for Value Estimation:
First-Visit Monte Carlo vs. Every-Visit Monte Carlo
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

def run_infographic_example(gamma: float = 0.9) -> Dict[str, float]:
    episode = [
        ("S0", "a0", -1),
        ("S1", "a1", -1),
        ("S2", "a2", +2),
        ("S3", "a3", +5),
    ]
    rewards = [r for _, _, r in episode]
    returns = {}
    for t in range(len(episode)):
        G = sum((gamma ** (k - t)) * rewards[k] for k in range(t, len(rewards)))
        state = episode[t][0]
        returns[state] = round(G, 3)
    return returns

class MonteCarloSimulation:
    def __init__(self, rows: int, cols: int, terminal_state: str, gamma: float = 1.0):
        self.rows = rows
        self.cols = cols
        self.gamma = gamma
        self.terminal_state = terminal_state.upper()
        self.state_positions = {}
        self.position_states = {}
        self.episodes = []
        self.initialize_grid()

    def initialize_grid(self) -> None:
        state_num = 0
        for i in range(self.rows):
            for j in range(self.cols):
                state = f"S{state_num}"
                self.state_positions[state] = (i, j)
                self.position_states[(i, j)] = state
                state_num += 1

    def set_episodes(self, episodes: List[List[Tuple[str, str, float]]]) -> None:
        self.episodes = [[(s.upper(), a, r) for s, a, r in ep] for ep in episodes]

    def calculate_returns(self, episode: List[Tuple[str, str, float]], start_idx: int) -> Tuple[float, str]:
        G = 0.0
        rewards = []
        for i in range(start_idx + 1, len(episode)):
            _, _, reward = episode[i]
            G = self.gamma * G + reward
            rewards.append(str(reward))
            if episode[i][0] == self.terminal_state:
                break
        breakdown = " + ".join(rewards) if rewards else "0"
        return G, f"G = {breakdown} = {G}"

    def first_visit_mc(self) -> Tuple[Dict[str, Optional[float]], Dict[str, List[str]]]:
        returns = defaultdict(list)
        calcs = defaultdict(lambda: ["NULL"] * len(self.episodes))
        for ep_idx, episode in enumerate(self.episodes):
            visited = set()
            for i, (state, _, _) in enumerate(episode):
                if state in visited or state == self.terminal_state:
                    continue
                visited.add(state)
                G, breakdown = self.calculate_returns(episode, i)
                returns[state].append(G)
                calcs[state][ep_idx] = breakdown
        V = {s: round(float(np.mean(returns[s])), 2) if returns[s] else None for s in self.state_positions}
        V[self.terminal_state] = None
        return V, calcs

    def every_visit_mc(self) -> Tuple[Dict[str, Optional[float]], Dict[str, List[str]]]:
        returns = defaultdict(list)
        calcs = defaultdict(lambda: ["NULL"] * len(self.episodes))
        for ep_idx, episode in enumerate(self.episodes):
            for i, (state, _, _) in enumerate(episode):
                if state == self.terminal_state:
                    continue
                G, breakdown = self.calculate_returns(episode, i)
                returns[state].append(G)
                calcs[state][ep_idx] = breakdown
        V = {s: round(float(np.mean(returns[s])), 2) if returns[s] else None for s in self.state_positions}
        V[self.terminal_state] = None
        return V, calcs

    def plot_heatmap(self, V_first: Dict[str, Optional[float]], V_every: Dict[str, Optional[float]], title_prefix: str, save_path: str):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.suptitle(f"{title_prefix} — Value Heatmaps", fontsize=13, fontweight='bold')
        for ax, V, method in zip(axes, [V_first, V_every], ["First-Visit MC", "Every-Visit MC"]):
            grid = np.full((self.rows, self.cols), np.nan)
            for state, pos in self.state_positions.items():
                val = V.get(state)
                if val is not None:
                    grid[pos] = val
            im = ax.imshow(grid, cmap='RdYlGn', interpolation='nearest')
            plt.colorbar(im, ax=ax)
            ax.set_title(method, fontweight='bold')
            for i in range(self.rows):
                for j in range(self.cols):
                    state = self.position_states.get((i, j), "")
                    val = V.get(state)
                    label = f"{state}\n{val:.2f}" if val is not None else f"{state}\nNULL"
                    ax.text(j, i, label, ha='center', va='center', fontsize=9, fontweight='bold', color='black')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
