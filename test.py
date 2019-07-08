import os
import logging
import struct
import traceback
import matplotlib.pyplot as plt
import numpy as np


def readFile(seed, f):
    filename = "%s%d.txt" % (f, seed)
    data = []
    if not os.path.exists(filename):
        logging.warning("no %s" % filename)
        return None
    with open(filename, "rb") as file:
        while True:
            try:
                temp = file.read(8)
                temp = struct.unpack("d",temp)[0]
                data.append(temp)
            except:
                traceback.print_exc()
                return data


def smooth(x, window_len=11, window='hanning'):
    # if x.ndim != 1:
    #     raise ValueError, "smooth only accepts 1 dimension arrays."
    # if x.size < window_len:
    #     raise ValueError, "Input vector needs to be bigger than window size."
    if window_len < 3:
        return x
    # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #     raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s = np.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len:-window_len + 1]

plt.figure()
filename = "respTime/"
resp = readFile(0, filename)
pktnum = readFile(0, "pktNum/")
reward = readFile(0, "reward/")
length = len(reward)
print(len(pktnum))
avg = []
avgrew = []
begin = 0
end = length//4
for i in range(4):
    avg.append(sum(resp[begin:end])/sum(pktnum[begin:end]))
    avgrew.append(sum(reward[begin:end]))
    begin = end
    end += length //4


# interval = 117
# end = interval
# for i in range(5):
#     avg.append(sum(resp[begin:end])/sum(pktnum[begin:end]))
#     avgrew.append(sum(reward[begin:end]))
#     begin = end
#     end += interval
# interval = 93
# for i in range(5):
#     avg.append(sum(resp[begin:end]) / sum(pktnum[begin:end]))
#     avgrew.append(sum(reward[begin:end]))
#     begin = end
#     end += interval

plt.plot(range(1,len(avg)+1),avg)
plt.xlabel("Iteration")
plt.ylabel("Response time")
plt.xticks(np.arange(1, len(avg)+1, step=1))
plt.title("Training performance")
# plt.savefig("figure/Training_per1.pdf", bbox_inches='tight')

plt.figure()
plt.plot(range(1,len(avgrew)+1),avgrew)
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.title("Training performance")
# plt.savefig("figure/Training_per2.pdf", bbox_inches='tight')


plt.figure()
resp = np.array(resp)
pktnum = np.array(pktnum)
avgresp = np.divide(resp, pktnum)
plt.plot((smooth(avgresp)))
plt.xlabel("Learning step")
plt.ylabel("Response time")
plt.title("Training performance")
# plt.savefig("figure/Training_per3.pdf", bbox_inches='tight')


# plt.figure()
# array = [0.0049901, 0.0037396, 0.0037356, 0.00363046, 0.00394329, 0.00404382, 0.00486709, 0.00501858, 0.00500536, 0.005023, 0.00503625]
# plt.bar(range(len(array)), array)
# plt.xlabel("Network at different iteration")
# plt.ylabel("Response time")
# plt.title("Testing performance")
# plt.savefig("figure/test_per.pdf", bbox_inches='tight')

plt.show()