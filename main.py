"""
Main Execution Script for Monte Carlo Methods
Shrri Dharshan D R - 23BPS1090
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from monte_carlo.simulator import run_infographic_example, MonteCarloSimulation

def compare_methods(V_first, V_every, grid_name):
    print(f"\n  ── Comparison: First-Visit vs Every-Visit ({grid_name}) ──")
    print(f"  {'State':<8} {'First-Visit':<15} {'Every-Visit':<15} {'Same?'}")
    print(f"  {'-'*50}")
    for s in sorted(V_first.keys(), key=lambda x: int(x[1:])):
        fv = V_first[s]
        ev = V_every[s]
        same = "Match" if fv == ev else "DIFF"
        print(f"  {s:<8} {str(fv):<15} {str(ev):<15} {same}")

def main():
    print("=" * 60)
    print("   REINFORCEMENT LEARNING LAB 5: MONTE CARLO METHODS")
    print("=" * 60)

    # Part 1: Infographic
    info_returns = run_infographic_example(gamma=0.9)
    print("\n[Part 1: Infographic Example (gamma = 0.9)]")
    print(f"  {'State':<8} {'Return G':<12} {'New V(s)'}")
    print(f"  {'-'*35}")
    for s, g in info_returns.items():
        print(f"  {s:<8} {g:<12} V({s}) = {g}")

    # Part 2: 2x3 GridWorld
    episodes_2x3 = [
        [("s4","U",-1), ("s1","R",-1), ("s2","L",-3), ("s1","stop",-1)],
        [("s3","U",-1), ("s0","stop",50)],
        [("s5","L",-4), ("s4","U",-2), ("s1","R",-1), ("s2","stop",-3)],
    ]
    mc1 = MonteCarloSimulation(rows=2, cols=3, terminal_state="S0", gamma=1.0)
    mc1.set_episodes(episodes_2x3)
    V1_fv, _ = mc1.first_visit_mc()
    V1_ev, _ = mc1.every_visit_mc()
    plot1_path = os.path.join("assets", "2x3_grid_heatmap.png")
    mc1.plot_heatmap(V1_fv, V1_ev, "2x3 Grid World", plot1_path)
    compare_methods(V1_fv, V1_ev, "2x3 Grid")

    # Part 3: 3x3 GridWorld
    episodes_3x3 = [
        [("S4","R",-1), ("S5","D",-1), ("S8","STOP",30)],
        [("S1","R",-1), ("S2","D",-1), ("S5","L",-1), ("S4","D",-1), ("S7","R",-1), ("S8","STOP",30)],
        [("S0","D",-1), ("S3","D",-1), ("S6","R",-1), ("S7","U",-1),
         ("S4","U",-1), ("S1","R",-1), ("S2","D",-1), ("S5","D",-1), ("S8","STOP",30)],
        [("S2","D",-1), ("S5","L",-1), ("S4","D",-1), ("S7","R",-1), ("S8","STOP",30)],
    ]
    mc2 = MonteCarloSimulation(rows=3, cols=3, terminal_state="S8", gamma=1.0)
    mc2.set_episodes(episodes_3x3)
    V2_fv, _ = mc2.first_visit_mc()
    V2_ev, _ = mc2.every_visit_mc()
    plot2_path = os.path.join("assets", "3x3_grid_heatmap.png")
    mc2.plot_heatmap(V2_fv, V2_ev, "3x3 Grid World", plot2_path)
    compare_methods(V2_fv, V2_ev, "3x3 Grid")

    print(f"\nHeatmap visualizations saved to {plot1_path} and {plot2_path}")

if __name__ == "__main__":
    main()
