#!/usr/bin/env python3

from baselines.common import tf_util as U
from baselines import logger
import pposgd_ex
from mlp_policy_ex import MlpPolicy
from sdn_env import sdn_simulator
from baselines.common import set_global_seeds
from collections import deque

iterNum = 2  # number of iteration for each environment setting
envNum = 2  # number of environment settings
timesteps_per_actorbatch = 1024

def make_sdn_env(num, seed):
    set_global_seeds(seed)
    return sdn_simulator(timesteps_per_actorbatch, num, seed)


def train(seed):
    U.make_session(num_cpu=1).__enter__()

    def policy_fn(name, vf_ob_space, ac_space):
        return MlpPolicy(name=name, vf_ob_space=vf_ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    end_timesteps = deque([])
    env_list = deque([])

    start_timestep = 0
    newround = deque([0]*envNum*iterNum)  # indicator of a new iteration for all settings

    for i in range(envNum):
        newround[i*iterNum] = 1
        for j in range(iterNum):
            env = make_sdn_env(i, seed)
            env_list.append(env)
            end_timestep = env.tot_pktNum + start_timestep
            # end_timestep = timesteps_per_actorbatch * (sum(env.set.pktRate) * (env.set.maxSimTime)//timesteps_per_actorbatch) + start_timestep
            end_timesteps.append(end_timestep)
            start_timestep = end_timestep

    pposgd_ex.learn(env_list, policy_fn,
        max_timesteps=end_timesteps[-1],
        timesteps_per_actorbatch=timesteps_per_actorbatch,
        clip_param=0.2, entcoeff=0.0,
        optim_epochs=5, optim_stepsize=3e-4, optim_batchsize=128,
        gamma=0.99, lam=0.95, schedule='linear', end_timesteps=end_timesteps, newround=newround)

    while len(env_list):
        env = env_list.popleft()
        env.close()


def main():
    seed = 1
    logger.configure()
    train(seed=seed)


if __name__ == '__main__':
    main()
