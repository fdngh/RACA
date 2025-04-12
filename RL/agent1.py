import numpy as np
import torch
from RL.PPO import ppo
# import rl_utils
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import json
import os


class env:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        # 加载JSON文件
        self.patch_data = self.load_patch_data()

    def load_patch_data(self):
        """从JSON文件加载patch数据"""
        json_path = os.path.join(os.path.dirname(__file__), 'patch_data.json')
        with open(json_path, 'r') as f:
            return json.load(f)

    def get_not_in_patch(self, text_em_name, action):
        B, a_dim = action.shape
        device = action.device
        # 创建一个全1的掩码
        mask = torch.ones(a_dim, dtype=torch.bool, device=device)

        # 从JSON数据中获取不在patch的索引
        if self.dataset_name == 'iu_xray':
            Not_in_patch = self.patch_data['iu_xray'][text_em_name]
        else:
            Not_in_patch = self.patch_data['default'][text_em_name]

        # 转换为PyTorch张量并应用掩码
        not_in_patch_mask = torch.tensor(Not_in_patch, dtype=torch.long, device=device)
        not_in_patch_bool = torch.zeros(a_dim, dtype=torch.bool, device=device)
        not_in_patch_bool[not_in_patch_mask] = True

        # 使用布尔掩码更新mask
        mask = mask & ~not_in_patch_bool

        # 创建all_reward并应用掩码
        all_reward = mask.float().unsqueeze(0).expand(B, -1)
        return all_reward

    def reward_model(self, action, text_em_name):
        all_reward = self.get_not_in_patch(text_em_name, action)

        is_same = action == all_reward
        negative_one_tensor = torch.full_like(action, -1, dtype=action.dtype)

        result = torch.where(
            is_same,
            torch.ones_like(action),  # 相同位置为1
            torch.where(all_reward == 0, all_reward, negative_one_tensor)  # 不同且tensor2为0的位置为0，否则为-1
        )

        # reward_a = action * all_reward
        reward = result.sum()  # dim=1, keepdim=True

        return reward

    def state(self, text_em, patch_feature, action):
        B, _ = action.shape
        feature = torch.sum(patch_feature, dim=2)
        text = text_em.repeat(B, 1)
        state = torch.cat([feature, action, text], dim=1)
        return state

    def next_state(self, text_em, patch_feature, action):
        return self.state(text_em, patch_feature, action)


class local_agent(nn.Module):
    def __init__(self, args, agent=None):
        super(local_agent, self).__init__()
        self.env = env(args.dataset_name)
        self.ppo = agent
        self.organs = ['Lung fields', 'Airways', "Heart", "Pulmonary vasculature", 'Mediastinum', 'Spine', 'Abdomen',
                       'Diaphragm',
                       'Shoulders', 'Ribs', 'Clavicles', 'Costophrenic angles']

        self.transition_dict = None

    def init_transition_dict(self):
        self.transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': []
        }

    def ss1(self, transition_dict):
        detached_dict = {
            'states': [s.detach() for s in transition_dict['states']],
            'actions': [a.detach() for a in transition_dict['actions']],
            'next_states': [ns.detach() for ns in transition_dict['next_states']],
            'rewards': [r.detach() for r in transition_dict['rewards']],
            'dones': [d.detach() for d in transition_dict['dones']]
        }
        self.ppo.update(detached_dict)

    def select_action1(self, text_em, image_feature, action1):
        state = self.env.next_state(text_em, image_feature, action1)

        return self.ppo.take_action(state)

    def forward(self, em, image_feature):  # , transition_dict
        if self.transition_dict is None:
            self.init_transition_dict()
        B, d, _ = image_feature.shape

        action = torch.zeros(B, 49).to(image_feature.device)

        local_action = []
        selected_patches = []
        for i in range(len(self.organs)):
            text_em_name = self.organs[i]
            text_em = em[i, :]
            state = self.env.next_state(text_em, image_feature, action)
            action_list = self.ppo.take_action(state)
            next_state = self.env.next_state(text_em, image_feature, action)
            # print(next_state)
            reward = self.env.reward_model(action, text_em_name)
            done = torch.tensor(text_em_name == 'Costophrenic angles', dtype=torch.float)
            action = action_list

            self.transition_dict['states'].append(state)
            self.transition_dict['actions'].append(action)
            self.transition_dict['next_states'].append(next_state)
            self.transition_dict['dones'].append(done)
            self.transition_dict['rewards'].append(reward.unsqueeze(0))
            local_action.append(action_list)
            # print(action_list)
            selected_patches.append(action_list.nonzero().squeeze().tolist())

        return local_action, self.transition_dict, selected_patches