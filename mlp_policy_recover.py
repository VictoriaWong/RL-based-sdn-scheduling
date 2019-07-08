from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from distribution_ex import make_pdtype
import numpy as np
import pickle, os, logging

def readFile(i,j):
    filename = "network/%d-%d.txt" % (i,j)
    if not os.path.exists(filename):
        logging.warning("no %s"%filename)
        return None
    with open(filename, "rb") as f:
        dict = pickle.load(f)
    return dict


class MlpPolicy(object):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, vf_ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(vf_ob_space, gym.spaces.Box)

        para = readFile(0, 716800)
        # para = readFile(0, 430080)
        # para = readFile(0, 239616)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        vf_ob = U.get_placeholder(name="vf_ob", dtype=tf.float32, shape=[sequence_length]+list(vf_ob_space.shape))
        nn_in = U.get_placeholder(name="nn_in", dtype=tf.float32, shape=[sequence_length]+list(vf_ob_space.shape))
        nn_outs = U.get_placeholder(name="nn_outs", dtype=tf.float32, shape=[ac_space.n])

        with tf.variable_scope("obfilter_vf"):
            self.vf_ob_rms = RunningMeanStd(shape=vf_ob_space.shape)

        with tf.variable_scope("obfilter_ac"):
            # self.ac_ob_rms = RunningMeanStd(shape=ac_ob_space.shape)
            self.nn_in_rms = RunningMeanStd(shape=vf_ob_space.shape)

        with tf.variable_scope('vf'):
            # vf_obz = tf.clip_by_value((vf_ob - self.vf_ob_rms.mean) / self.vf_ob_rms.std, -5.0, 5.0)
            vf_last_out = vf_ob
            for i in range(num_hid_layers):
                vf_last_out = tf.nn.tanh(tf.layers.dense(vf_last_out, hid_size, name="fc%i" % (i+1), kernel_initializer=tf.constant_initializer(para[i*2]), bias_initializer=tf.constant_initializer(para[i*2+1])))
            self.vpred = tf.layers.dense(vf_last_out, 1, name='final', kernel_initializer=tf.constant_initializer(para[num_hid_layers*2]), bias_initializer=tf.constant_initializer(para[num_hid_layers*2+1]))[:,0]

        with tf.variable_scope('pol'):
            # ac_obz = tf.clip_by_value((nn_in - self.nn_in_rms.mean) / self.nn_in_rms.std, -5.0, 5.0)
            ac_last_out = nn_in
            for i in range(num_hid_layers):
                ac_last_out = tf.nn.tanh(tf.layers.dense(ac_last_out, hid_size, name='fc%i' % (i + 1), kernel_initializer=tf.constant_initializer(para[(i+num_hid_layers+1)*2]), bias_initializer=tf.constant_initializer(para[(i+num_hid_layers+1)*2+1])))
            self.nn_out = tf.nn.softplus(tf.layers.dense(ac_last_out, 1, name='mid_final', kernel_initializer=tf.constant_initializer(para[num_hid_layers*5]), bias_initializer=tf.constant_initializer(para[num_hid_layers*5+1])))[:,0]  # action value prediction/ performance of each controller

        # M1: calculate the probability: remain the two largest elements
        with tf.variable_scope('prob'):
            epsilon = 0.00001  # assign a small probability value (epsilon/ctlNum) instead of 0 to avoid log(0)
            sum = tf.reduce_sum(nn_outs)
            ac_in2 = nn_outs / tf.reshape(sum, (-1,1))  # normalize the input within range [0,1], shape:[None, CtlNum]
            val = tf.map_fn(lambda x:tf.nn.top_k(x, 2)[0], ac_in2)  # shape: [None, k(i.e.,2)]
            ind = tf.map_fn(lambda x:tf.nn.top_k(x, 2)[1], ac_in2, dtype=tf.int32)  # shape: [None, k]
            lam = 0.5*(1-tf.reduce_sum(val, axis=1)-epsilon*(ac_space.n-2))  # shape: a list with length=None
            mask = tf.map_fn(lambda x: tf.one_hot(x, ac_space.n, on_value=1.0, off_value=0.), ind, dtype=tf.float32)  # shape: [None, k, CtlNum]
            mask = tf.reduce_sum(mask, axis=1)  # shape: [None, 16]
            self.prob = (ac_in2 + tf.reshape(lam, (-1, 1)))*mask  # assign 0 to probabilities except the 2 largest probabilites
            mask0 = tf.ones(shape=mask.shape)
            mask0 = (mask0 - mask)*epsilon  # assign epsilon instead of 0 to the probabilities
            self.prob = self.prob + mask0

        # M2: calculate the probability using softmax
        # self.prob = tf.nn.softmax(nn_outs)

        probability = U.get_placeholder(name="logits", dtype=tf.float32, shape=[ac_space.n])
        self.pd = pdtype.pdfromflat(probability)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())  # mean value of the distribution

        self._vpred = U.function([vf_ob], [self.vpred])
        self._nneval = U.function([nn_in],[self.nn_out])
        self._calprob = U.function([nn_outs], [self.prob])
        self._act = U.function([stochastic, probability], [ac])

    def calculate_ac_value(self, ac_ob):  # input:(CtlNum, 9), output:(1, CtlNum)
        ac_vpred = self._nneval(ac_ob)[0]  # shape: (CtlNum, 1)
        ac_vpred = ac_vpred.reshape((1, -1))[0]  # shape: (1, CtlNum)
        if sum(ac_vpred) == 0:
            print("all policy neural network outputs are 0 !!!!!!!!!!")
        return ac_vpred

    def calculate_ac_prob(self, ac_ob):  # input: (CtlNum, 9), output:(1, CtlNum)
        ac_vpred = self.calculate_ac_value(ac_ob)
        proba = self._calprob(ac_vpred)
        # return proba[0]
        return proba[0][0]

        # epsilon = 0.0  # assign a small probability value (epsilon/ctlNum) instead of 0 to avoid log(0)
        # ac_vpred = ac_vpred / np.sum(ac_vpred)  # normalization
        # sorted_ind = np.argsort(ac_vpred)  # sort the array in ascending order
        # mask = np.zeros(sorted_ind.shape)
        # for i in range(2):
        #     mask[sorted_ind[-1 - i]] = 1
        # temp = 0.5 * (1 - np.sum(np.multiply(mask, ac_vpred))-epsilon)  # sum the elements except the largest 2
        # proba = np.multiply(mask, ac_vpred + temp)
        # proba += epsilon/Setting.ctlNum
        # proba[sorted_ind[-1]] += 1-sum(proba)  # guarantee the probability sum is 1
        # if sum(proba)-1 != 0:
        #     print("sum(proba) not equal to 1")
        # return proba

    def vf_pred(self, vf_ob):
        vpred1 = self._vpred(vf_ob[None])
        return vpred1[0]

    def act(self, stochastic, ac_ob):
        prob = self.calculate_ac_prob(ac_ob)
        ac1 = self._act(stochastic, prob)
        if prob[ac1] == 0:
            print("The action with 0 probability is selected")
        return np.array(ac1)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []