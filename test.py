def calculate_final_reward(state, reward_weights):
    final_reward = 0.0

    for key, value in state.items():
        if key in reward_weights:
            final_reward += value * reward_weights[key]
            print(f'{key} :{value * reward_weights[key]}')

    return final_reward


# 状态字典
state = {'hp_point': 0.0, 'tower_hp_point': 0.007833333333333359, 'money': 100, 'exp': 0.0, 'ep_rate': 0.0054466230936819175, 'death': 0, 'kill': 0, 'last_hit': 0.0, 'forward': 1.0, 'hurt_hero': 0.5, 'kill_income': 0, 'cakes': -0.3150896619744794, 'buff_skills': 3, 'atk_range': 1, 'win': 0, 'no_action': 1, 'time_punish': 2.0869444444444445, 'monster': 0.0, 'reward_sum': 0.02490085796546542}

# 奖励权重字典
REWARD_WEIGHT_DICT = {
    "hp_point": 2.0,
    "tower_hp_point": 6.0,
    "money": 0.008,
    "exp": 0.006,
    "ep_rate": 0.65,
    "death": -1.0,
    "kill": -0.6,
    "last_hit": 0.6,
    "forward": 0.035,

    "hurt_hero": 0.015,
    "kill_income": 0.0005,
    "cakes": 0.040,
    "buff_skills": 0.001,
    "atk_range": 0.001,
    "win": 40,
    "no_action": -0.0008,
    "time_punish": -0.008,
    "monster": 0.06,
}
# 计算最终奖励
final_reward = calculate_final_reward(state, REWARD_WEIGHT_DICT)
print(final_reward)
