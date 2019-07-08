from baselines import logger
import baselines.common.tf_util as U
import numpy as np
import time
import collections
import struct
from mlp_policy_recover import MlpPolicy
# from mlp_policy_ex import MlpPolicy
from sdn_env import sdn_simulator
from baselines.common import set_global_seeds


def traj_segment_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    vf_ob, ac_ob = env.reset()

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    # Initialize history arrays
    vf_obs = np.array([vf_ob for _ in range(horizon)])
    ac_obs = np.array([ac_ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()
    resp_times = np.zeros(horizon, 'float32')
    pkt_nums = np.zeros(horizon, 'int32')

    while True:
        prevac = ac
        vpred = pi.vf_pred(vf_ob)
        # ==================Define your policy=======================================
        if env.set.algo == 'RL':
            ac = pi.act(stochastic, ac_ob)
        elif env.set.algo == 'RANDOM':
            ac = np.array([np.random.randint(env.set.ctlNum)])
        elif env.set.algo == 'WEIGHTED_RANDOM':
            pr = np.array(env.set.ctlRate)  # capacity-based probability
            pr = pr.astype(float)
            pr /= pr.sum()
            ac = np.array([np.random.choice(range(env.ctlNum), replace=True, p = pr)])
        elif env.set.algo == 'OPTIMAL':
            ac = np.array([0])
        else:
            print("wrong scheduling algorithms !!!!!!!!!!!!!!!")
            break
        # ac = np.array([np.random.choice(range(3), replace=False, p= [0.35, 0.47, 0.18])])
        # ac = np.array([np.random.choice(range(4), replace=False, p=[0.145, 0.38, 0.19, 0.285])])
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"vf_ob": vf_obs, "ac_ob": ac_obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens, "resptime": resp_times, "pktnum": pkt_nums}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        vf_obs[i] = vf_ob
        ac_obs[i] = ac_ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        vf_ob, ac_ob, rew, resptime, pktnum, new, _ = env.step(ac)
        rews[i] = rew
        resp_times[i] = resptime
        pkt_nums[i] = pktnum

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            vf_ob, ac_ob = env.reset()
        t += 1


def learn(env, policy_fn, *,
          timesteps_per_actorbatch,  # timesteps per actor per update
          clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
          optim_epochs, optim_stepsize, optim_batchsize,  # optimization hypers
          gamma, lam,  # advantage estimation
          max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
          callback=None,  # you can do anything in the callback, since it takes locals(), globals()
          adam_epsilon=1e-5,
          schedule='constant'  # annealing for stepsize parameters (epsilon and adam)
          ):
    # Open a file to record the accumulated rewards
    #rewFile = open("reward/%s-%d.txt" % (env.set.algo, env.seed), "ab")
    #resptimeFile = open("respTime/%s-%d.txt" % (env.set.algo, env.seed), "ab")
    #pktnumFile = open("pktNum/%s-%d.txt" % (env.set.algo, env.seed), "ab")
    # Setup losses and stuff
    # ----------------------------------------
    vf_ob_space = env.vf_observation_space
    ac_space = env.action_space
    pi = policy_fn("pi1", vf_ob_space, ac_space)  # Construct network for new policy

    U.initialize()



    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    resptime = []
    pktnum = []

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            #rewFile.close()
            #resptimeFile.close()
            #pktnumFile.close()
            print(sum(resptime)/sum(pktnum), sum(pktnum))
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        logger.log("********** Iteration %i ************" % iters_so_far)

        seg = seg_gen.__next__()

        # record_reward(rewFile, sum(seg["rew"]))
        # record_reward(resptimeFile, sum(seg["resptime"]))
        # record_reward(pktnumFile, sum(seg["pktnum"]))
        resptime.append(sum(seg["resptime"]))
        pktnum.append(sum(seg["pktnum"]))
        print("average response time: %s, num of pkts: %s" % (
        sum(seg["resptime"]) / sum(seg["pktnum"]), sum(seg["pktnum"])))
        # print("total rewards for Iteration %s: %s" % (iters_so_far, sum(seg["rew"])))
        prob = collections.Counter(seg["ac"])  # a dict where elements are stored as dictionary keys and their counts are stored as dictionary values.
        for key in prob:
            prob[key] = prob[key]/len(seg["ac"])
        print("percentage of choosing each controller: %s" % (prob))

        lens = seg["ep_lens"]

        iters_so_far += 1
        timesteps_so_far += sum(lens)


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


def record_reward(file, num):
    num = struct.pack("d", num)
    file.write(num)
    file.flush()


def make_sdn_env(timesteps_per_actorbatch, num, seed):
    set_global_seeds(seed)
    return sdn_simulator(timesteps_per_actorbatch, num, seed)


def train(seed, j):

    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, vf_ob_space, ac_space):
        return MlpPolicy(name=name, vf_ob_space=vf_ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    timesteps_per_actorbatch = 1024
    env = make_sdn_env(timesteps_per_actorbatch, j, seed)
    num_timesteps = timesteps_per_actorbatch * (sum(env.set.pktRate) * (env.set.maxSimTime) // timesteps_per_actorbatch)
    learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=timesteps_per_actorbatch,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=5, optim_stepsize=3e-4, optim_batchsize=128,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()


def main():
    seed = 0
    j = 5
    logger.configure()
    train(seed=seed, j=j)


if __name__ == '__main__':
    main()