import argparse
import gymnasium as gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Cart Pole
# 倒立摆（CartPole）。

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


render_mode = "human" if args.render else None
env = gym.make('CartPole-v1', render_mode=render_mode)
env.reset(seed=args.seed)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    # 在同一模型中实现 Actor 和 Critic。
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)

        # actor's layer
        # Actor，即策略分支的层。
        self.action_head = nn.Linear(128, 2)

        # critic's layer
        # Critic，即价值评估分支的层。
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        # 动作和奖励缓冲区。
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        # Actor 和 Critic 的前向传播。
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # Actor 根据状态 s_t 选择动作，
        # by returning probability of each action
        # 返回各个动作的概率。
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        # Critic 评估处于状态 s_t 的价值。
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 以二元组返回 Actor 和 Critic 的结果：
        # 1. a list with the probability of each action over the action space
        # 1．动作空间中各动作的概率列表；
        # 2. the value from state s_t
        # 2．状态 s_t 的价值。
        return action_prob, state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    # 根据动作概率列表创建类别分布，
    m = Categorical(probs)

    # and sample an action using the distribution
    # 并从分布中采样一个动作。
    action = m.sample()

    # save to action buffer
    # 保存到动作缓冲区。
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    # 要执行的动作：向左或向右。
    return action.item()


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    # 训练代码：计算 Actor 和 Critic 损失，并执行反向传播。
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    # 保存 Actor（策略）损失的列表。
    value_losses = [] # list to save critic (value) loss
    # 保存 Critic（价值）损失的列表。
    returns = [] # list to save the true values
    # 保存真实回报目标的列表。

    # calculate the true value using rewards returned from the environment
    # 根据环境返回的奖励计算回报目标。
    for r in model.rewards[::-1]:
        # calculate the discounted value
        # 计算折扣回报。
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        # 计算 Actor（策略）损失。
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        # 使用平滑 L1 损失计算 Critic（价值）损失。
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    # 清空梯度。
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    # 将 policy_losses 和 value_losses 中的所有值相加。
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    # 执行反向传播。
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    # 清空奖励和动作缓冲区。
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    running_reward = 10

    # run infinitely many episodes
    # 持续运行多个回合。
    for i_episode in count(1):

        # reset environment and episode reward
        # 重置环境和当前回合奖励。
        state, _ = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # 每个回合最多运行 9999 步，
        # infinite loop while learning
        # 避免学习过程中陷入无限循环。
        for t in range(1, 10000):

            # select action from policy
            # 根据策略选择动作。
            action = select_action(state)

            # take the action
            # 执行动作。
            state, reward, terminated, truncated, _ = env.step(action)

            model.rewards.append(reward)
            ep_reward += reward
            if terminated or truncated:
                break

        # update cumulative reward
        # 更新累计奖励。
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        # 执行反向传播。
        finish_episode()

        # log results
        # 记录结果。
        if i_episode % args.log_interval == 0:
            print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}')

        # check if we have "solved" the cart pole problem
        # 检查是否已解决倒立摆任务。
        if running_reward > env.spec.reward_threshold:
            print(f"Solved! Running reward is now {running_reward} and "
                  f"the last episode runs to {t} time steps!")
            break


if __name__ == '__main__':
    main()
