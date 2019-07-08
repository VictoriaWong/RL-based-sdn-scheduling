from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
from distribution_ex import CategoricalPd
import math, collections
import struct, pickle


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
        ac = pi.act(stochastic, ac_ob)
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


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def learn(env_list, policy_fn, *,
          timesteps_per_actorbatch,  # timesteps per actor per update
          clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
          optim_epochs, optim_stepsize, optim_batchsize,  # optimization hypers
          gamma, lam,  # advantage estimation
          max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
          callback=None,  # you can do anything in the callback, since it takes locals(), globals()
          adam_epsilon=1e-5,
          schedule='constant',  # annealing for stepsize parameters (epsilon and adam)
          end_timesteps,
          newround
          ):

    env = env_list.popleft()
    # Open a file to record the accumulated rewards
    rewFile = open("reward/%d.txt" % (env.seed), "ab")
    resptimeFile = open("respTime/%d.txt" % (env.seed), "ab")
    pktnumFile = open("pktNum/%d.txt" % (env.seed), "ab")

    # Setup losses and stuff
    # ----------------------------------------
    vf_ob_space = env.vf_observation_space
    # ac_ob_space = env.ac_observation_space
    ac_space = env.action_space
    pi = policy_fn("pi1", vf_ob_space, ac_space)  # Construct network for new policy
    oldpi = policy_fn("oldpi", vf_ob_space, ac_space)  # Network for old policy
    atarg = tf.placeholder(name="atarg", dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(name="ret", dtype=tf.float32, shape=[None])  # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])  # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult  # Annealed clipping parameter epislon

    vf_ob = U.get_placeholder_cached(name="vf_ob")
    nn_in = U.get_placeholder_cached(name="nn_in")  # placeholder for nn input
    ac = pi.pdtype.sample_placeholder([None])

    # kloldnew = oldpi.pd.kl(pi.pd)
    # ent = pi.pd.entropy()
    pb_old_holder = tf.placeholder(name="pd_old", dtype=tf.float32, shape=[None, ac_space.n])
    pb_new_holder = tf.placeholder(name="pd_new", dtype=tf.float32, shape=[None, ac_space.n])
    oldpd = CategoricalPd(pb_old_holder)
    pd = CategoricalPd(pb_new_holder)
    kloldnew = oldpd.kl(pd)
    ent = pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    # ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # pnew / pold
    ratio = tf.placeholder(dtype=tf.float32, shape=[None])
    surr1 = ratio * atarg  # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg  #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    vf_var_list = [v for v in var_list if v.name.split("/")[1].startswith("vf")]
    pol_var_list = [v for v in var_list if v.name.split("/")[1].startswith("pol")]

    vf_grad = U.function([vf_ob, ret], U.flatgrad(vf_loss, vf_var_list))  # gradient of value function
    pol_nn_grad = U.function([nn_in], U.flatgrad(pi.nn_out, pol_var_list))
    vf_adam = MpiAdam(vf_var_list, epsilon=adam_epsilon)
    pol_adam = MpiAdam(pol_var_list, epsilon=adam_epsilon)
    clip_para = U.function([lrmult], [clip_param])

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in
                                                    zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([vf_ob, atarg, ret, lrmult, ratio, pb_new_holder, pb_old_holder], losses)

    U.initialize()
    vf_adam.sync()
    pol_adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)

    end_timestep = end_timesteps.popleft()
    new = newround.popleft()
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=10)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=10)  # rolling buffer for episode rewards
    env_so_far = 1

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            rewFile.close()
            resptimeFile.close()
            pktnumFile.close()
            para = {}
            for vf in range(len(vf_var_list)):
                # para[vf_var_list[vf].name] = vf_var_list[vf].eval()
                para[vf] = vf_var_list[vf].eval()
            for pol in range(len(pol_var_list)):
                # para[pol_var_list[pol].name] = pol_var_list[pol].eval()
                para[pol + len(vf_var_list)] = pol_var_list[pol].eval()
            f = open("network/%d-%d.txt" % (env.seed, timesteps_so_far), "wb")
            pickle.dump(para, f)
            f.close()
            print("============================= policy is stored =================================")
            break
        elif end_timestep and timesteps_so_far >= end_timestep:
            env = env_list.popleft()
            seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)
            end_timestep = end_timesteps.popleft()
            new = newround.popleft()
            env_so_far += 1
            if True:
                para = {}
                for vf in range(len(vf_var_list)):
                    # para[vf_var_list[vf].name] = vf_var_list[vf].eval()
                    para[vf] = vf_var_list[vf].eval()
                for pol in range(len(pol_var_list)):
                    # para[pol_var_list[pol].name] = pol_var_list[pol].eval()
                    para[pol + len(vf_var_list)] = pol_var_list[pol].eval()
                f = open("network/%d-%d.txt" % (env.seed, timesteps_so_far), "wb")
                pickle.dump(para, f)
                f.close()
            print("======================== new environment (%s network settings left) ===========================" % len(env_list))
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break
        elif timesteps_so_far == 0:
            para = {}
            for vf in range(len(vf_var_list)):
                # para[vf_var_list[vf].name] = vf_var_list[vf].eval()
                para[vf] = vf_var_list[vf].eval()
            for pol in range(len(pol_var_list)):
                # para[pol_var_list[pol].name] = pol_var_list[pol].eval()
                para[pol + len(vf_var_list)] = pol_var_list[pol].eval()
            f = open("network/%d-%d.txt" % (env.seed, timesteps_so_far), "wb")
            pickle.dump(para, f)
            f.close()

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i, Environment %i ************" % (iters_so_far, env_so_far))

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # for vf in range(len(vf_var_list)):
        #     print(vf_var_list[vf].name, vf_var_list[vf].eval())
        # for pol in range(len(pol_var_list)):
        #     print(pol_var_list[pol].name, pol_var_list[pol].eval())

        record_reward(rewFile, sum(seg["rew"]))
        record_reward(resptimeFile, sum(seg["resptime"]))
        record_reward(pktnumFile, sum(seg["pktnum"]))
        print("total rewards for Iteration %s: %s" % (iters_so_far, sum(seg["rew"])))
        print("average response time: %s, num of pkts: %s" % (sum(seg["resptime"])/sum(seg["pktnum"]), sum(seg["pktnum"])))
        prob = collections.Counter(seg["ac"])  # a dict where elements are stored as dictionary keys and their counts are stored as dictionary values.
        for key in prob:
            prob[key] = prob[key]/len(seg["ac"])
        print("percentage of choosing each controller: %s" % (prob))

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        vf_ob, ac_ob, ac, atarg, tdlamret = seg["vf_ob"], seg['ac_ob'], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"]  # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
        d = Dataset(dict(vf_ob=vf_ob, ac_ob=ac_ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or vf_ob.shape[0]

        # if hasattr(pi, "vf_ob_rms"): pi.vf_ob_rms.update(vf_ob)  # update running mean/std for policy
        # if hasattr(pi, "nn_in_rms"):
        #     temp = ac_ob.reshape(-1,ac_ob.shape[2])
        #     pi.nn_in_rms.update(temp)

        assign_old_eq_new()  # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = []  # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                # calculate the value function gradient
                vf_g = vf_grad(batch["vf_ob"], batch["vtarg"])
                vf_adam.update(vf_g, optim_stepsize * cur_lrmult)

                # calculate the policy gradient
                pol_g = []
                ratios = []
                pbs_new_batch = []
                pbs_old_batch = []
                e = clip_para(cur_lrmult)[0]
                for sample_id in range(optim_batchsize):
                    sample_ac_ob = batch["ac_ob"][sample_id]
                    sample_ac = batch["ac"][sample_id]
                    probs_new = pi.calculate_ac_prob(sample_ac_ob)
                    prob_new = probs_new[sample_ac]
                    probs_old = oldpi.calculate_ac_prob(sample_ac_ob)
                    prob_old = probs_old[sample_ac]
                    if prob_old == 0:
                        logger.error("pi_old = 0 in %s th iteration %s th epoch %s th sample..." % (iters_so_far, _, sample_id))
                    r = prob_new / prob_old
                    ratios.append(r)
                    pbs_new_batch.append(probs_new)
                    pbs_old_batch.append(probs_old)
                    if (r > 1.0 + e and batch["atarg"][sample_id] > 0) or (r < 1.0 - e and batch["atarg"][sample_id] < 0) or r == 0:
                        dnn_dtheta = pol_nn_grad(sample_ac_ob[0].reshape(1, -1))
                        pol_g.append(0.*dnn_dtheta)
                    else:
                        nn = pi.calculate_ac_value(sample_ac_ob)
                        denominator = np.power(sum(nn), 2)
                        sorted_ind = np.argsort(nn)  # sort the array in ascending order
                        if len(probs_new) == 2:
                            if sample_ac == 0:
                                numerator1 = nn[1]*pol_nn_grad(sample_ac_ob[0].reshape(1,-1))
                                numerator2 = nn[0] * pol_nn_grad(sample_ac_ob[1].reshape(1, -1))
                                dpi_dtheta = -(numerator1-numerator2)/denominator
                            else:
                                numerator1 = nn[1]*pol_nn_grad(sample_ac_ob[0].reshape(1,-1))
                                numerator2 = nn[0]*pol_nn_grad(sample_ac_ob[1].reshape(1,-1))
                                dpi_dtheta = -(numerator2 - numerator1)/denominator

                            # numerator1 = nn[sorted_ind[0]]*pol_nn_grad(sample_ac_ob[sorted_ind[1]].reshape(1,-1))
                            # numerator2 = nn[sorted_ind[1]]*pol_nn_grad(sample_ac_ob[sorted_ind[0]].reshape(1,-1))
                            # dpi_dtheta = (numerator1-numerator2)/denominator

                        elif len(probs_new) == 3:
                            if sample_ac == sorted_ind[0]:
                                # the controller with lowest probability will still possible to be chosen because the probability is not zero
                                dnn_dtheta = pol_nn_grad(sample_ac_ob[0].reshape(1, -1))
                                pol_g.append(0. * dnn_dtheta)
                            else:
                                numerator1 = sum(nn) * (pol_nn_grad(sample_ac_ob[sample_ac].reshape(1,-1)) + 0.5 * pol_nn_grad(
                                    sample_ac_ob[sorted_ind[0]].reshape(1, -1)))
                                numerator2 = (nn[sample_ac] + 0.5 * nn[sorted_ind[0]]) * pol_nn_grad(sample_ac_ob)
                                dpi_dtheta = -(numerator1 - numerator2) / denominator
                        else:
                            if sample_ac == sorted_ind[-1] or sample_ac == sorted_ind[-2]:
                                numerator1 = sum(nn) * (pol_nn_grad(sample_ac_ob[sample_ac] .reshape(1,-1))+0.5*pol_nn_grad(sample_ac_ob[sorted_ind[0:-2]]))
                                numerator2 = (nn[sample_ac]+0.5*sum(nn[sorted_ind[0:-2]])) * pol_nn_grad(sample_ac_ob)
                                dpi_dtheta = -(numerator1 - numerator2) / denominator
                            else:
                                dnn_dtheta = pol_nn_grad(sample_ac_ob[0].reshape(1, -1))
                                pol_g.append(0. * dnn_dtheta)
                        pol_g.append(batch["atarg"][sample_id] * dpi_dtheta / prob_old)

                pol_g_mean = np.mean(np.array(pol_g), axis=0)
                pol_adam.update(pol_g_mean, optim_stepsize * cur_lrmult)

                newlosses = compute_losses(batch["vf_ob"], batch["atarg"], batch["vtarg"],
                                           cur_lrmult, np.array(ratios), np.array(pbs_new_batch), np.array(pbs_old_batch))

                # adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        # losses = []
        # for batch in d.iterate_once(optim_batchsize):
        #     newlosses = compute_losses(batch["vf_ob"], batch["ac_ob"], batch["ac"], batch["atarg"], batch["vtarg"],
        #                                cur_lrmult)
        #     losses.append(newlosses)
        meanlosses, _, _ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_" + name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        if len(lenbuffer) == 0:
            logger.record_tabular("EpLenMean", 0)
            logger.record_tabular("EpRewMean", 0)
        else:
            logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.dump_tabular()


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


def record_reward(file, num):
    num = struct.pack("d", num)
    file.write(num)
    file.flush()