# 📁 evaluate.py — Evaluate DQN, A2C, PPO trên 2 map Beacon & Mineral

import os
import sys
import numpy as np
import pandas as pd

# ===== CONFIG =====
EPISODES_EVAL = 3000
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

OUTPUT_BEACON   = os.path.join(LOG_DIR, "evaluate_beacon.csv")
OUTPUT_MINERAL  = os.path.join(LOG_DIR, "evaluate_mineral.csv")
OUTPUT_ALL      = os.path.join(LOG_DIR, "evaluate_all_maps.csv")

AGENTS = ["dqn", "a2c", "ppo"]
MAPS   = ["beacon", "mineral"]

# ===== POSSIBLE COLUMN NAMES =====
POSSIBLE_REWARD_COLS = [
    "totalreward", "total_reward", "reward", "rewards", "episode_reward",
    "total reward", "ep_reward", "return", "EpisodeReward"
]

def find_reward_col(df: pd.DataFrame):
    """Tìm cột chứa reward không phân biệt hoa/thường."""
    cols_lower = [c.lower().replace(" ", "") for c in df.columns]
    for idx, c in enumerate(cols_lower):
        for cand in POSSIBLE_REWARD_COLS:
            if cand.replace("_", "") == c:
                return df.columns[idx]
    for idx, c in enumerate(cols_lower):
        if "reward" in c or "total" in c:
            return df.columns[idx]
    return None

def load_rewards_array(agent, map_name):
    filename = f"{agent}_{map_name}_rewards.csv"
    path = os.path.join(LOG_DIR, filename)
    if not os.path.exists(path):
        print(f"  ❌ Không tìm thấy file: {path}")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  ❌ Lỗi đọc CSV {path}: {e}")
        return None

    reward_col = find_reward_col(df)
    if reward_col is None:
        print(f"  ❌ Không tìm thấy cột reward trong {filename}. Có các cột: {list(df.columns)}")
        return None

    rewards = pd.to_numeric(df[reward_col], errors="coerce").dropna().values
    if len(rewards) == 0:
        print(f"  ⚠ File {filename} không có dữ liệu reward hợp lệ.")
        return np.array([])

    used_n = min(EPISODES_EVAL, len(rewards))
    if len(rewards) < EPISODES_EVAL:
        print(f"  ⚠ {filename} chỉ có {len(rewards)} episode — dùng {used_n} episode thực tế.")
    else:
        print(f"  ✔ {filename}: {len(rewards)} reward — dùng {used_n} episode đầu để đánh giá.")

    return rewards[:used_n]

def compute_metrics(arr):
    if arr is None or len(arr) == 0:
        return dict(EpisodesUsed=0, MeanReward=np.nan, MedianReward=np.nan,
                    StdReward=np.nan, MinReward=np.nan, MaxReward=np.nan)

    return dict(
        EpisodesUsed=len(arr),
        MeanReward=float(np.mean(arr)),
        MedianReward=float(np.median(arr)),
        StdReward=float(np.std(arr)),
        MinReward=float(np.min(arr)),
        MaxReward=float(np.max(arr))
    )

def evaluate_all(visualize=False):
    all_results = []

    for map_name in MAPS:
        print(f"\n🚩 ĐÁNH GIÁ TRÊN MAP: {map_name.upper()}")
        rows = []

        for agent in AGENTS:
            print(f" ▶ {agent.upper()}...")
            rewards = load_rewards_array(agent, map_name)
            metrics = compute_metrics(rewards)

            row = {"Agent": agent.upper(), "Map": map_name, **metrics}
            rows.append(row)
            all_results.append(row)

        df = pd.DataFrame(rows)
        out_file = OUTPUT_BEACON if map_name == "beacon" else OUTPUT_MINERAL
        df.to_csv(out_file, index=False)
        print(f" ✅ Lưu kết quả → {out_file}")

    df_all = pd.DataFrame(all_results)
    df_all.to_csv(OUTPUT_ALL, index=False)
    print(f"\n✅ Lưu kết quả tổng hợp → {OUTPUT_ALL}")

if __name__ == "__main__":
    print(f"Start Evaluate (EPISODES_EVAL = {EPISODES_EVAL})")
    evaluate_all(visualize=False)
    print("Finished ✅")
