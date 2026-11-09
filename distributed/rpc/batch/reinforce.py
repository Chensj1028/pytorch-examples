import argparse
import gymnasium as gym
import os
import threading
import time
import warnings

# Suppress deprecated ProcessGroup warning
# 屏蔽已弃用的 ProcessGroup 警告。
warnings.filterwarnings("ignore", message="You are using a Backend.*ProcessGroup")

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote
from torch.distributions import Categorical

# demonstrating using rpc.functions.async_execution to speed up training
# 演示如何使用 rpc.functions.async_execution 加速训练。

NUM_STEPS = 500
AGENT_NAME = "agent"
OBSERVER_NAME = "observer{}"

parser = argparse.ArgumentParser(description='PyTorch RPC Batch RL example')
parser.add_argument('--gamma', type=float, default=1.0, metavar='G',
                    help='discount factor (default: 1.0)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--num-episode', type=int, default=10, metavar='E',
                    help='number of episodes (default: 10)')
parser.add_argument('--max-world-size', type=int, default=3, metavar='W',
                    help='maximum world size to test (default: 3)')
args = parser.parse_args()

torch.manual_seed(args.seed)


class Policy(nn.Module):
    r"""
    Borrowing the ``Policy`` class from the Reinforcement Learning example.
    Copying the code to make these two examples independent.
    See https://github.com/pytorch/examples/tree/main/reinforcement_learning
    """
    # 借用强化学习示例中的 Policy 类。
    # 复制代码是为了让两个示例相互独立。
    # 原示例见上方仓库链接。
    def __init__(self, batch=True):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)
        self.dim = 2 if batch else 1

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=self.dim)


class Observer:
    r"""
    An observer has exclusive access to its own environment. Each observer
    captures the state from its environment, and send the state to the agent to
    select an action. Then, the observer applies the action to its environment
    and reports the reward to the agent.

    It is true that CartPole-v1 is a relatively inexpensive environment, and it
    might be an overkill to use RPC to connect observers and trainers in this
    specific use case. However, the main goal of this tutorial to how to build
    an application using the RPC API. Developers can extend the similar idea to
    other applications with much more expensive environment.
    """
    # 每个观察者独占自己的环境，读取环境状态并发送给智能体，
    # 由智能体选择动作；观察者执行该动作，再向智能体报告奖励。
    # CartPole-v1 的环境开销较小，在此场景中通过 RPC 连接观察者与训练进程可能显得复杂。
    # 本教程的主要目标是演示 RPC API 的应用结构，这种思路可扩展到环境开销更大的应用。
    def __init__(self, batch=True):
        self.id = rpc.get_worker_info().id - 1
        self.env = gym.make('CartPole-v1')
        self.env.reset(seed=args.seed)
        self.select_action = Agent.select_action_batch if batch else Agent.select_action

    def run_episode(self, agent_rref, n_steps):
        r"""
        Run one episode of n_steps.

        Args:
            agent_rref (RRef): an RRef referencing the agent object.
            n_steps (int): number of steps in this episode
        """
        # 运行一个包含 n_steps 步的回合。
        # 参数 agent_rref（RRef）：指向智能体对象的远程引用。
        # 参数 n_steps（int）：当前回合的步数。
        state, _ = self.env.reset()
        ep_reward = NUM_STEPS
        rewards = torch.zeros(n_steps)
        start_step = 0
        for step in range(n_steps):
            state = torch.from_numpy(state).float().unsqueeze(0)
            # send the state to the agent to get an action
            # 将状态发送给智能体，以获取动作。
            action = rpc.rpc_sync(
                agent_rref.owner(),
                self.select_action,
                args=(agent_rref, self.id, state)
            )

            # apply the action to the environment, and get the reward
            # 在环境中执行动作，并获得奖励。
            state, reward, terminated, truncated, _ = self.env.step(action)
            rewards[step] = reward

            if terminated or truncated or step + 1 >= n_steps:
                curr_rewards = rewards[start_step:(step + 1)]
                R = 0
                for i in range(curr_rewards.numel() -1, -1, -1):
                    R = curr_rewards[i] + args.gamma * R
                    curr_rewards[i] = R
                state, _ = self.env.reset()
                if start_step == 0:
                    ep_reward = min(ep_reward, step - start_step + 1)
                start_step = step + 1

        return [rewards, ep_reward]


class Agent:
    def __init__(self, world_size, batch=True):
        self.ob_rrefs = []
        self.agent_rref = RRef(self)
        self.rewards = {}
        self.policy = Policy(batch).cuda()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.running_reward = 0

        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))
            self.ob_rrefs.append(remote(ob_info, Observer, args=(batch,)))
            self.rewards[ob_info.id] = []

        self.states = torch.zeros(len(self.ob_rrefs), 1, 4)
        self.batch = batch
        # With batching, saved_log_probs contains a list of tensors, where each
        # 启用批处理时，saved_log_probs 是张量列表，列表中的每个
        # tensor contains probs from all observers in one step.
        # 张量包含同一步中所有观察者的概率。
        # Without batching, saved_log_probs is a dictionary where the key is the
        # 未启用批处理时，saved_log_probs 是字典，键为
        # observer id and the value is a list of probs for that observer.
        # 观察者编号，值为该观察者的概率列表。
        self.saved_log_probs = [] if self.batch else {k:[] for k in range(len(self.ob_rrefs))}
        self.future_actions = torch.futures.Future()
        self.lock = threading.Lock()
        self.pending_states = len(self.ob_rrefs)

    @staticmethod
    @rpc.functions.async_execution
    def select_action_batch(agent_rref, ob_id, state):
        r"""
        Batching select_action: In each step, the agent waits for states from
        all observers, and process them together. This helps to reduce the
        number of CUDA kernels launched and hence speed up amortized inference
        speed.
        """
        # 批量选择动作：每一步等待所有观察者的状态，再统一处理。
        # 这样可减少启动 CUDA 内核的次数，提高摊销后的推理速度。
        self = agent_rref.local_value()
        self.states[ob_id].copy_(state)
        future_action = self.future_actions.then(
            lambda future_actions: future_actions.wait()[ob_id].item()
        )

        with self.lock:
            self.pending_states -= 1
            if self.pending_states == 0:
                self.pending_states = len(self.ob_rrefs)
                probs = self.policy(self.states.cuda())
                m = Categorical(probs)
                actions = m.sample()
                self.saved_log_probs.append(m.log_prob(actions).t()[0])
                future_actions = self.future_actions
                self.future_actions = torch.futures.Future()
                future_actions.set_result(actions.cpu())
        return future_action

    @staticmethod
    def select_action(agent_rref, ob_id, state):
        r"""
        Non-batching select_action, return the action right away.
        """
        # 非批量动作选择：立即返回动作。
        self = agent_rref.local_value()
        probs = self.policy(state.cuda())
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs[ob_id].append(m.log_prob(action))
        return action.item()

    def run_episode(self, n_steps=0):
        r"""
        Run one episode. The agent will tell each oberser to run one episode
        with n_steps. Then it collects all actions and rewards, and use those to
        train the policy.
        """
        # 运行一个回合：智能体要求每个观察者运行 n_steps 步，
        # 然后收集所有动作和奖励，用于训练策略。
        futs = []
        for ob_rref in self.ob_rrefs:
            # make async RPC to kick off an episode on all observers
            # 通过异步 RPC，让所有观察者开始一个回合。
            futs.append(ob_rref.rpc_async().run_episode(self.agent_rref, n_steps))

        # wait until all obervers have finished this episode
        # 等待所有观察者完成当前回合。
        rets = torch.futures.wait_all(futs)
        rewards = torch.stack([ret[0] for ret in rets]).cuda().t()
        ep_rewards = sum([ret[1] for ret in rets]) / len(rets)

        if self.batch:
            probs = torch.stack(self.saved_log_probs)
        else:
            probs = [torch.stack(self.saved_log_probs[i]) for i in range(len(rets))]
            probs = torch.stack(probs)

        policy_loss = -probs * rewards / len(rets)
        policy_loss.sum().backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # reset variables
        # 重置变量。
        self.saved_log_probs = [] if self.batch else {k:[] for k in range(len(self.ob_rrefs))}
        self.states = torch.zeros(len(self.ob_rrefs), 1, 4)

        # calculate running rewards
        # 计算运行中的奖励统计。
        self.running_reward = 0.5 * ep_rewards + 0.5 * self.running_reward
        return ep_rewards, self.running_reward


def run_worker(rank, world_size, n_episode, batch, print_log=True):
    r"""
    This is the entry point for all processes. The rank 0 is the agent. All
    other ranks are observers.
    """
    # 所有进程的入口：rank 0 为智能体，其余进程为观察者。
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if rank == 0:
        # rank0 is the agent
        # rank 0 为智能体。
        rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)

        agent = Agent(world_size, batch)
        for i_episode in range(n_episode):
            last_reward, running_reward = agent.run_episode(n_steps=NUM_STEPS)

            if print_log:
                print(f'Episode {i_episode}\tLast reward: {last_reward:.2f}\tAverage reward: {running_reward:.2f}')
    else:
        # other ranks are the observer
        # 其余进程为观察者。
        rpc.init_rpc(OBSERVER_NAME.format(rank), rank=rank, world_size=world_size)
        # observers passively waiting for instructions from agents
        # 观察者被动等待智能体的指令。
    rpc.shutdown()


def main():
    for world_size in range(2, args.max_world_size):
        delays = []
        for batch in [True, False]:
            tik = time.time()
            mp.spawn(
                run_worker,
                args=(world_size, args.num_episode, batch),
                nprocs=world_size,
                join=True
            )
            tok = time.time()
            delays.append(tok - tik)

        print(f"{world_size}, {delays[0]}, {delays[1]}")


if __name__ == '__main__':
    main()
