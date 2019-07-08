# Reimplement the Categorical distribution:
# Initialized parameters: probability


import tensorflow as tf
import numpy as np
import baselines.common.tf_util as U


class Pd(object):
    """
    A particular probability distribution
    """

    def flatparam(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def logp(self, x):
        return - self.neglogp(x)


class PdType(object):
    """
    Parametrized family of probability distributions
    """

    def pdclass(self):
        raise NotImplementedError

    def pdfromflat(self, flat):
        return self.pdclass()(flat)

    def param_shape(self):
        raise NotImplementedError

    def sample_shape(self):
        raise NotImplementedError

    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape + self.param_shape(), name=name)

    def sample_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape + self.sample_shape(), name=name)


class CategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat  # variable length

    def pdclass(self):
        return CategoricalPd

    def param_shape(self):
        return [self.ncat]

    def sample_shape(self):
        return []

    def sample_dtype(self):
        return tf.int32


class CategoricalPd(Pd):
    def __init__(self, prob):
        self.prob = prob

    def flatparam(self):
        return self.prob

    def mode(self):
        return tf.argmax(self.prob, axis=-1)

    def neglogp(self, x):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        one_hot_actions = tf.one_hot(x, self.prob.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits(
            logits=tf.log(self.prob),
            labels=one_hot_actions)

    def kl(self, other):
        return tf.reduce_sum(self.prob * (tf.log(self.prob) - tf.log(other.prob)), axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.prob * (-tf.log(self.prob)), axis=-1)

    def sample(self):  # Sampling via the Gumbel distribution based on wikipedia
        u = tf.random_uniform(tf.shape(self.prob))
        return tf.argmax(tf.log(self.prob) - tf.log(-tf.log(u)), axis=-1)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


def make_pdtype(ac_space):
    from gym import spaces
    if isinstance(ac_space, spaces.Discrete):
        return CategoricalPdType(ac_space.n)
    else:
        raise NotImplementedError


def shape_el(v, i):
    maybe = v.get_shape()[i]
    if maybe is not None:
        return maybe
    else:
        return tf.shape(v)[i]


@U.in_session
def test_probtypes():
    np.random.seed(0)

    pdparam_categorical = np.array([-.2, .3, .5])
    categorical = CategoricalPdType(pdparam_categorical.size)  # pylint: disable=E1101
    validate_probtype(categorical, pdparam_categorical)


def validate_probtype(probtype, pdparam):
    N = 100000
    # Check to see if mean negative log likelihood == differential entropy
    Mval = np.repeat(pdparam[None, :], N, axis=0)
    M = probtype.param_placeholder([N])
    X = probtype.sample_placeholder([N])
    pd = probtype.pdfromflat(M)
    calcloglik = U.function([X, M], pd.logp(X))
    calcent = U.function([M], pd.entropy())
    Xval = tf.get_default_session().run(pd.sample(), feed_dict={M: Mval})
    logliks = calcloglik(Xval, Mval)
    entval_ll = - logliks.mean()  # pylint: disable=E1101
    entval_ll_stderr = logliks.std() / np.sqrt(N)  # pylint: disable=E1101
    entval = calcent(Mval).mean()  # pylint: disable=E1101
    assert np.abs(entval - entval_ll) < 3 * entval_ll_stderr  # within 3 sigmas

    # Check to see if kldiv[p,q] = - ent[p] - E_p[log q]
    M2 = probtype.param_placeholder([N])
    pd2 = probtype.pdfromflat(M2)
    q = pdparam + np.random.randn(pdparam.size) * 0.1
    Mval2 = np.repeat(q[None, :], N, axis=0)
    calckl = U.function([M, M2], pd.kl(pd2))
    klval = calckl(Mval, Mval2).mean()  # pylint: disable=E1101
    logliks = calcloglik(Xval, Mval2)
    klval_ll = - entval - logliks.mean()  # pylint: disable=E1101
    klval_ll_stderr = logliks.std() / np.sqrt(N)  # pylint: disable=E1101
    assert np.abs(klval - klval_ll) < 3 * klval_ll_stderr  # within 3 sigmas
    print('ok on', probtype, pdparam)
