import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random
import vizdoom as vzd
import time
import sys
from collections import deque
import heapq
import torch.optim as optim
import torch.nn.functional as F

# 設置 OpenMP 環境變數
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 定義配置文件和地圖文件路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "scenarios", "defend_the_center.cfg")
wad_path = os.path.join(current_dir, "scenarios", "defend_the_center.wad")

# 檢查配置文件和地圖文件是否存在
if not os.path.exists(config_path) or not os.path.exists(wad_path):
    raise FileNotFoundError(f"Config or WAD file not found: {config_path}, {wad_path}")

# 定義超參數
LEARNING_RATE = 1e-5  # 降低學習率
GAMMA = 0.99
NUM_EPISODES = 2000000
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 50000
UPDATE_FREQ = 4
EPSILON_START = 1.0
EPSILON_END = 0.2
EPSILON_DECAY = 0.9995
MAX_HEALTH = 100
FORWARD_REWARD = 0.1  # 增加前進獎勵
BACKWARD_PENALTY = 0.02  # 增加後退懲罰
HEALTH_REWARD = 1.0  # 增加生命值獎勵
DAMAGE_PENALTY = 1.0  # 增加受傷懲罰
KILL_REWARD = 500.0  # 增加擊殺獎勵
SURVIVAL_REWARD = 0.2  # 增加生存獎勵
ROTATION_PENALTY = 0.00001  # 大幅降低旋轉懲罰
ATTACK_REWARD = 5.0  # 增加攻擊獎勵
DODGE_REWARD = 2.0  # 增加閃避獎勵
HEALTH_THRESHOLD = 30
DEATH_PENALTY = 100.0  # 增加死亡懲罰

# 新增評估參數
EVAL_INTERVAL = 100
EVAL_EPISODES = 3
PERFORMANCE_THRESHOLD = -300
MAX_NO_IMPROVE = 3

# 選擇設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 導入 DSACV2
sys.path.append(os.path.join(current_dir, 'DSAC-v2'))
from dsac_v2 import DSAC_V2
from env_gym.gym_vizdoom_data import VizDoomEnv

# 定義經驗回放緩衝區
class ReplayBuffer:
    def __init__(self, capacity, action_size):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.action_size = action_size

    def push(self, state, action, reward, next_state, done):
        # 強制轉為 numpy array 並 reshape
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_size,):
            print(f"[警告] action shape 不正確: {action.shape}，自動 reshape")
            action = action.reshape(-1)[:self.action_size]
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        # 將 action 轉成 numpy array 再轉 tensor，確保 shape 為 [batch_size, action_size]
        action = np.stack(action)
        return {
            "obs": torch.stack(state).to(device),
            "act": torch.tensor(action, dtype=torch.float32).to(device),
            "rew": torch.tensor(reward, dtype=torch.float32).to(device),
            "obs2": torch.stack(next_state).to(device),
            "done": torch.tensor(done, dtype=torch.float32).to(device)
        }

    def __len__(self):
        return len(self.buffer)

# 繪製獎勵曲線圖
def plot_rewards(rewards, window=10):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Reward")
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, label=f"{window}-ep Moving Avg")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.legend()
    plt.grid()
    plt.savefig("reward_plot15.png")
    plt.close()

# 計算獎勵函數
def calculate_reward(reward, info, last_angle, step_count, last_kills):
    # rotation: 旋轉越多越好
    current_angle = info.get('angle', 0)
    rotation = abs(current_angle - last_angle)
    rotation_reward = rotation * 0.1  # 旋轉越多越好
    # attack: 擊殺才給獎勵
    kills = info.get('kills', 0)
    attack_reward = 5 if kills > last_kills else 0
    # survival: 每步只加 1 分
    survival_reward = 1
    total_reward = rotation_reward + attack_reward + survival_reward
    reward_detail = {
        'rotation': rotation_reward,
        'attack': attack_reward,
        'survival': survival_reward
    }
    return total_reward, reward_detail, kills

# 訓練主函數
def train():
    print("開始訓練...")
    torch.set_default_dtype(torch.float32)
    
    # 初始化環境
    env = VizDoomEnv(config_path=config_path, wad_path=wad_path, render=True)
    action_size = env.action_space.shape[0]
    print(f"動作空間大小: {action_size}")
    
    # 初始化 DSACV2 agent
    dsac_kwargs = {
        'obsv_dim': 60 * 80,
        'act_dim': action_size,
        'gamma': GAMMA,
        'tau': 0.005,
        'auto_alpha': True, 
        'alpha': 0.2,
        'delay_update': 2,
        'value_func_type': 'MLP',  
        'policy_func_type': 'MLP',
        'value_func_name': 'ActionValueDistri',
        'policy_func_name': 'StochaPolicy',
        'value_hidden_sizes': [256, 256],
        'policy_hidden_sizes': [256, 256],
        'value_hidden_activation': 'relu',
        'policy_hidden_activation': 'relu',
        'value_output_activation': 'linear',
        'policy_output_activation': 'linear',
        'action_type': 'continu',
        'value_learning_rate': LEARNING_RATE,
        'policy_learning_rate': LEARNING_RATE,
        'alpha_learning_rate': 1e-4,  # 降低 alpha 學習率
        'cnn_shared': False,
        'action_dim': action_size,
        'action_high_limit': [1.0] * action_size,
        'action_low_limit': [-1.0] * action_size,
        'policy_act_distribution': 'GaussDistribution',
    }
    agent = DSAC_V2(**dsac_kwargs)
    agent.networks.to(device)
    
    # 初始化訓練狀態
    start_episode = 0
    reward_history = []
    print("從頭開始訓練")
    
    print("DSACV2 agent 初始化完成")
    
    # 初始化經驗回放緩衝區
    memory = ReplayBuffer(REPLAY_BUFFER_SIZE, action_size)
    print("經驗回放緩衝區初始化完成")
    
    # 初始化 epsilon
    epsilon = EPSILON_START
    
    reward_history = []
    performance_history = []
    no_improve_count = 0
    best_performance = float('-inf')
    
    print(f"開始訓練，總回合數: {NUM_EPISODES}")
    
    MAX_STEPS_PER_EPISODE = 1000  # 避免死循環
    try:
        with open("log.txt", "w") as log_file:
            for episode in range(start_episode, NUM_EPISODES):
                start_time = time.time()
                state = env.reset()
                episode_reward = 0
                done = False
                step_count = 0
                last_health = MAX_HEALTH
                last_position = np.array([0, 0, 0])
                last_angle = 0
                last_attack_time = 0
                last_dodge_time = 0
                consecutive_hits = 0
                total_damage_taken = 0
                total_damage_dealt = 0
                total_kills = 0
                total_distance = 0
                min_health = MAX_HEALTH
                last_reward = 0
                reward_history_episode = []
                last_kills = 0
                
                while not done and step_count < MAX_STEPS_PER_EPISODE:
                    # 選擇動作
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    if random.random() < epsilon:
                        action = np.random.uniform(-1.0, 1.0, size=action_size).astype(np.float32)
                    else:
                        with torch.no_grad():
                            action = agent.networks.policy(state_tensor).cpu().numpy()[0]
                            action = action[:action_size]  # 只取 mean 部分
                    # 執行動作
                    next_state, reward, done, info = env.step(action)
                    #print("info:", info)  # Debug: 印出 info 內容
                    # 記錄最低生命值
                    if info.get('health', MAX_HEALTH) < min_health:
                        min_health = info.get('health', MAX_HEALTH)
                    # 計算獎勵
                    reward, reward_detail, current_kills = calculate_reward(reward, info, last_angle, step_count, last_kills)
                    last_kills = current_kills
                    
                    # 更新狀態
                    last_health = info.get('health', MAX_HEALTH)
                    last_position = info.get('position', np.array([0, 0, 0]))
                    last_angle = info.get('angle', 0)
                    last_attack_time = info.get('attack_time', 0)
                    last_dodge_time = info.get('dodge_time', 0)
                    consecutive_hits = info.get('consecutive_hits', 0)
                    total_damage_taken = info.get('damage_taken', 0)
                    total_damage_dealt = info.get('damage_dealt', 0)
                    total_kills = info.get('kills', 0)
                    last_reward = reward
                    
                    # 儲存經驗
                    memory.push(state, action, reward, next_state, done)
                    
                    # 更新狀態
                    state = next_state
                    episode_reward += reward
                    step_count += 1
                    
                    # 訓練模型
                    if len(memory) > BATCH_SIZE and step_count % UPDATE_FREQ == 0:
                        batch = memory.sample(BATCH_SIZE)
                        data = {
                            "obs": batch['obs'],
                            "act": batch['act'],
                            "rew": batch['rew'],
                            "obs2": batch['obs2'],
                            "done": batch['done']
                        }
                        agent.local_update(data, step_count)
                    
                    # 更新 epsilon
                    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
                    
                    # 記錄獎勵
                    reward_history_episode.append(reward)
                    
                    # 檢查是否死亡
                    if info.get('health', MAX_HEALTH) <= 0:
                        done = True
                
                # 更新獎勵歷史
                reward_history.append(episode_reward)
                # Print reward breakdown
                print(f"回合 {episode + 1}/{NUM_EPISODES} | 總獎勵: {episode_reward:.2f} (rotation:{reward_detail['rotation']:.2f}, attack:{reward_detail['attack']:.2f}, survival:{reward_detail['survival']:.2f}) | ε: {epsilon:.3f}")
                log_file.write(f"回合 {episode + 1}/{NUM_EPISODES} | 總獎勵: {episode_reward:.2f} (rotation:{reward_detail['rotation']:.2f}, attack:{reward_detail['attack']:.2f}, survival:{reward_detail['survival']:.2f}) | ε: {epsilon:.3f}\n")
                
                # 每 100 回合顯示一次訓練進度並保存檢查點
                if (episode + 1) % 100 == 0:
                    avg_reward = np.mean(reward_history[-100:])
                    max_reward = np.max(reward_history[-100:])
                    log_file.write(f"\\n=== 訓練進度報告 ===\\n")
                    log_file.write(f"回合 {episode + 1}/{NUM_EPISODES}\\n")
                    log_file.write(f"平均獎勵: {avg_reward:.2f}\\n")
                    log_file.write(f"最大獎勵: {max_reward:.2f}\\n")
                    log_file.write(f"探索率: {epsilon:.2%}\\n")
                    log_file.write(f"==================\\n\\n")
                    print(f"\\n=== 訓練進度報告 ===")
                    print(f"回合 {episode + 1}/{NUM_EPISODES}")
                    print(f"平均獎勵: {avg_reward:.2f}")
                    print(f"最大獎勵: {max_reward:.2f}")
                    print(f"探索率: {epsilon:.2%}")
                    print("==================\n")

                # 每 1000 回合保存檢查點
                if (episode + 1) % 1000 == 0:
                    checkpoint_path = os.path.join(current_dir, "checkpoints", f"{episode + 1}.pth")
                    torch.save({
                        'model_state_dict': agent.networks.state_dict(),
                        # 修改為儲存 Actor 和 Critic 優化器狀態
                        'q1_optimizer_state_dict': agent.networks.q1_optimizer.state_dict(),
                        'q2_optimizer_state_dict': agent.networks.q2_optimizer.state_dict(),
                        'policy_optimizer_state_dict': agent.networks.policy_optimizer.state_dict(),
                        'alpha_optimizer_state_dict': agent.networks.alpha_optimizer.state_dict(),
                        'episode': episode,
                        'reward_history': reward_history,
                    }, checkpoint_path)
                    log_file.write(f"模型已保存: {checkpoint_path}\\n")
                    print(f"模型已保存: {checkpoint_path}")
                
                # 計算最近 10 回合的統計資訊
                recent_rewards = reward_history[-10:] if len(reward_history) >= 10 else reward_history
                avg_reward = np.mean(recent_rewards)
                
                # 計算當前回合的統計資訊
                current_stats = {
                    'kills': 0,
                    'damage_taken': 0,
                    'distance': 0,
                    'shots': 0,
                    'hits': 0,
                    'health': 100,
                    'ammo': 0
                }
                
                # 收集當前回合的統計資訊
                state = next_state
                while not done:
                    _, _, done, info = env.step(action)
                    current_stats['kills'] = max(current_stats['kills'], info['kills'])
                    current_stats['damage_taken'] = max(current_stats['damage_taken'], info['damage_taken'])
                    current_stats['distance'] = max(current_stats['distance'], info['distance_traveled'])
                    current_stats['shots'] = max(current_stats['shots'], info['shots_fired'])
                    current_stats['hits'] = max(current_stats['hits'], info['hits'])
                    current_stats['health'] = min(current_stats['health'], info['health'])
                    current_stats['ammo'] = max(current_stats['ammo'], info['ammo'])
                
                # print(f"回合 {episode + 1}/{NUM_EPISODES} | 獎勵: {episode_reward:.2f} | 擊殺: {current_stats['kills']} | 移動: {current_stats['distance']:.1f} | 生命: {current_stats['health']} | 平均: {avg_reward:.2f} | ε: {epsilon:.3f}")
    
    except KeyboardInterrupt:
        print("\n訓練被用戶中斷")
        torch.save(agent.state_dict(), 'model_checkpoint_interrupted.pth')
    finally:
        env.close()
        if reward_history:
            print(f"\n訓練總結:")
            print(f"回合數: {len(reward_history)}")
            print(f"平均獎勵: {np.mean(reward_history):.2f}")
            print(f"最大獎勵: {np.max(reward_history):.2f}")
            print(f"最小獎勵: {np.min(reward_history):.2f}")
            plot_rewards(reward_history)
            print("最終獎勵圖表已保存")

if __name__ == "__main__":
    train()