import sys
sys.path.append("..")
sys.path.append(".")
from typing import DefaultDict
import pickle
import os
SAVE_LOG = False

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

try:
    import pygame
except ImportError:
    pygame = None

from env.vehicle import Status
from configs import *

def _save_episode_frame(env, log_path: str, episode_idx: int):
    """Save a snapshot of the current pygame screen for qualitative inspection."""
    if log_path is None:
        return
    frames_dir = os.path.join(log_path, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    filename = os.path.join(frames_dir, f"episode_{episode_idx:04d}.png")

    # Try to grab the pygame surface directly (best quality)
    if pygame is not None:
        try:
            surface = getattr(getattr(env, "env", env), "screen", None)
            if surface is not None:
                pygame.image.save(surface, filename)
                return
        except Exception:
            pass

    # Fallback: render an rgb frame via the environment and save the image observation
    try:
        raw_obs = getattr(getattr(env, "env", env), "render")("rgb_array")
        img = raw_obs.get("img", None)
        if img is not None:
            # img is (C,W,H) after wrapper; transpose to HWC for saving
            if img.shape[0] in (1,3):
                img = np.transpose(img, (1,2,0))
            plt.imsave(filename, img.astype(np.uint8))
    except Exception:
        pass

def _save_episode_signals(log_path: str, episode_idx: int, speed_history, steer_history):
    """Save velocity and steering traces for the episode."""
    if log_path is None or (len(speed_history) == 0 and len(steer_history) == 0):
        return
    signals_dir = os.path.join(log_path, "signals")
    os.makedirs(signals_dir, exist_ok=True)
    filename = os.path.join(signals_dir, f"episode_{episode_idx:04d}.png")
    steps = np.arange(1, len(speed_history) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    axes[0].plot(steps, speed_history, color="tab:blue")
    axes[0].set_ylabel("Speed")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].plot(steps, steer_history, color="tab:orange")
    axes[1].set_ylabel("Steering")
    axes[1].set_xlabel("Step")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)

def eval(env, agent, episode=2000, log_path='', multi_level=False, post_proc_action=True, save_signal_plots=True):

    succ_rate_case = DefaultDict(list)
    if multi_level:
        succ_rate_level = DefaultDict(list)
        step_num_level = DefaultDict(list)
        path_length_level = DefaultDict(list)
    reward_case = DefaultDict(list)
    reward_record = []
    succ_record = []
    success_step_record = []
    step_record = DefaultDict(list)
    path_length_record = DefaultDict(list)
    eval_record = []

    for i in trange(episode):
        obs = env.reset(i+1)
        agent.reset()
        done = False
        total_reward = 0
        step_num = 0
        path_length = 0
        last_xy = (env.vehicle.state.loc.x, env.vehicle.state.loc.y)
        last_obs = obs['target']
        speed_history = []
        steer_history = []
        while not done:
            step_num += 1
            if post_proc_action:
                action, _ = agent.choose_action(obs)
            else:
                action, _ = agent.get_action(obs)
            if (last_obs == obs['target']).all():
                action = env.action_space.sample()
            last_obs = obs['target']
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            obs = next_obs
            path_length += np.linalg.norm(np.array(last_xy)-np.array((env.vehicle.state.loc.x, env.vehicle.state.loc.y)))
            last_xy = (env.vehicle.state.loc.x, env.vehicle.state.loc.y)
            speed_history.append(env.vehicle.state.speed)
            steer_history.append(env.vehicle.state.steering)
            
            if info['path_to_dest'] is not None:
                agent.set_planner_path(info['path_to_dest'])
            if done:
                if info['status']==Status.ARRIVED:
                    succ_record.append(1)
                else:
                    succ_record.append(0)

        reward_record.append(total_reward)
        succ_rate_case[env.map.case_id].append(succ_record[-1])
        if step_num < 200:
            path_length_record[env.map.case_id].append(path_length)
        reward_case[env.map.case_id].append(reward_record[-1])
        if multi_level:
            succ_rate_level[env.map.map_level].append(succ_record[-1])
            if step_num < 200:
                path_length_level[env.map.map_level].append(path_length)
            step_num_level[env.map.map_level].append(step_num)
        if info['status']==Status.OUTBOUND:
            step_record[env.map.case_id].append(200)
        else:
            step_record[env.map.case_id].append(step_num)
        if succ_record[-1] == 1:
            success_step_record.append(step_num)
        eval_record.append({'case_id':env.map.case_id,
                            'status':info['status'],
                            'step_num':step_num,
                            'reward':total_reward,
                            'path_length':path_length,
                            })
        _save_episode_frame(env, log_path, i+1)
        if save_signal_plots:
            _save_episode_signals(log_path, i+1, speed_history, steer_history)

    print('#'*15)
    print('EVALUATE RESULT:')
    print('success rate: ', np.mean(succ_record))
    print('average reward: ', np.mean(reward_record))
    print('-'*10)
    print('success rate per case: ')
    case_ids = [int(k) for k in succ_rate_case.keys()]
    case_ids.sort()
    if len(case_ids) < 10:
        print('-'*10)
        print('average reward per case: ')
        for k in case_ids:
            env.reset(k)
            # Map-level classification is bypassed; report case ID only
            print('case %s :' % k, np.mean(succ_rate_case[k]))
        for k in case_ids:
            print('case %s :'%k, np.mean(reward_case[k]), np.mean(step_record[k]), '+-(%s)'%np.std(step_record[k]))

    if multi_level:
        print('success rate per level: ')
        for k in succ_rate_level.keys():
            print('%s (case num %s):'%(k, len(succ_rate_level[k])) + '%s '%np.mean(succ_rate_level[k]))
    
    if log_path is not None:
        def plot_time_ratio(node_list):
            max_node = TOLERANT_TIME
            raw_len = len(node_list)
            filtered_node_list = []
            for n in node_list:
                if n != max_node:
                    filtered_node_list.append(n)
            filtered_node_list.sort()
            ratio_list = [i/raw_len for i in range(1,len(filtered_node_list)+1)]
            plt.plot(filtered_node_list, ratio_list)
            plt.xlabel('Search node')
            plt.ylabel('Accumulate success rate')
            fig = plt.gcf()
            fig.savefig(log_path+'/success_rate.png')
            plt.close()
        all_step_record = []
        for k in step_record.keys():
            all_step_record.extend(step_record[k])
        plot_time_ratio(all_step_record)

        # save eval result
        f_record = open(log_path+'/record.data', 'wb')
        pickle.dump(eval_record, f_record)
        f_record.close()

        f_record_txt = open(log_path+'/result.txt', 'w', newline='')
        f_record_txt.write('success rate: %s\n'%np.mean(succ_record))
        f_record_txt.write('step num: %s '%np.mean(success_step_record)+'+-(%s)\n'%np.std(success_step_record))
        if multi_level:
            f_record_txt.write('\n')
            for k in succ_rate_level.keys():
                f_record_txt.write('%s (case num %s):'%(k, len(succ_rate_level[k])) + '%s \n'%np.mean(succ_rate_level[k]))
                f_record_txt.write('step num: %s '%np.mean(step_num_level[k])+'+-(%s)\n'%np.std(step_num_level[k]))
                f_record_txt.write('path length: %s '%np.mean(path_length_level[k])+'+-(%s)\n'%np.std(path_length_level[k]))
        if len(case_ids) < 10:
            for k in case_ids:
                f_record_txt.write('\ncase %s : '%k + 'success rate: %s \n'%np.mean(succ_rate_case[k]))
                f_record_txt.write('step num: %s '%np.mean(step_record[k])+'+-(%s)\n'%np.std(step_record[k]))
                f_record_txt.write('path length: %s '%np.mean(path_length_record[k])+'+-(%s)\n'%np.std(path_length_record[k]))
        f_record_txt.close()
    
    return np.mean(succ_record)
