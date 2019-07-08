import struct
import os
import logging
import matplotlib.pyplot as plt
import traceback
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
import itertools

label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams.update({'font.size': label_size})
matplotlib.rcParams['lines.linewidth'] = 5


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


# ================== Figure (a) (b) ===============================:
# ---------------- reward ---------------------
plt.figure()
filename = "reward/ctl1000_0.002_1000_0.02_pkt500/"

xcoords = []
temp = 0
for i in range(5):
    temp += 117
    xcoords.append(temp)

colors = 'k'

sns.set_style("white")
reward = []
time = []
seed = []
algo_type = []
length = 0

for i in range(5):
    temp = readFile(i, filename)
    length = len(temp)//2
    temp = smooth(np.array(temp), window_len=40)
    temp = temp.tolist()
    length = length//5
    reward.extend(temp[:length])
    time.extend(range(0,length))
    seed.extend([i]*length)
    algo_type.extend(["Proposed"]*length)

filename += "OPTIMAL-"
for i in range(5):
    temp = readFile(0, filename)
    avg = np.mean(temp)
    reward.extend([avg]*length)
    time.extend(range(0,length))
    seed.extend([i]*length)
    algo_type.extend(["Optimal"]*length)



reward = [x/1000 for x in reward]
df = pd.DataFrame({"Simulation time (s)": time, "seed": seed, "algo": algo_type, "Reward x 1000": reward})
sns.tsplot(time="Simulation time (s)", value="Reward x 1000", unit="seed", condition="algo", data=df)
plt.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=0.,fontsize=label_size)
# for i in range(len(xcoords)):
#     plt.axvline(x=xcoords[i], c=colors, linewidth=0.5, linestyle='-.')
plt.xticks(np.arange(0, 117, 29.3))
loc, labels = plt.xticks()
plt.xticks(loc,(0, 60, 120, 180, 240))
plt.axvline(x=29.3, c=colors, linewidth=0.8, linestyle='-.')

plt.savefig("figure/ctl1000_0.002_1000_0.02_pkt500_reward.pdf", bbox_inches='tight')

# -------------------response time------------------
plt.figure()
filename1 = "respTime/ctl1000_0.002_1000_0.02_pkt500/"
filename2 = "pktNum/ctl1000_0.002_1000_0.02_pkt500/"
sns.set_style("white")
reward = []
time = []
seed = []
algo_type = []
length = 0

for i in range(5):
    temp1 = readFile(i, filename1)
    temp2 = readFile(i, filename2)
    temp = np.divide(temp1, temp2)
    length = len(temp)//2
    temp = smooth(np.array(temp), window_len=40)
    temp = temp.tolist()
    length = length // 5
    reward.extend(temp[:length])
    time.extend(range(0,length))
    seed.extend([i]*length)
    algo_type.extend(["Proposed"]*length)

filename1 += "OPTIMAL-"
filename2 += "OPTIMAL-"
for i in range(5):
    temp1 = readFile(0, filename1)
    temp2 = readFile(0, filename2)
    temp = np.divide(temp1, temp2)
    avg = np.mean(temp)
    reward.extend([avg]*length)
    time.extend(range(0,length))
    seed.extend([i]*length)
    algo_type.extend(["Optimal"]*length)
reward = np.array(reward)*1000
df = pd.DataFrame({"Simulation time (s)": time, "seed": seed, "algo": algo_type, "Response time (ms)": reward})
sns.tsplot(time="Simulation time (s)", value="Response time (ms)", unit="seed", condition="algo", data=df)
plt.legend(bbox_to_anchor=(1, 1), loc="upper right", borderaxespad=0.,fontsize=label_size)

# for i in range(len(xcoords)):
#     plt.axvline(x=xcoords[i], c=colors, linewidth=0.5, linestyle='-.')
plt.axvline(x=29.3, c=colors, linewidth=0.8, linestyle='-.')
plt.xticks(np.arange(0, 117, 29.3))
loc, labels = plt.xticks()
plt.xticks(loc,(0, 60, 120, 180, 240))

plt.savefig("figure/ctl1000_0.002_1000_0.02_pkt500_resptime.pdf", bbox_inches='tight')



# ================== Figure (c) ===============================:
plt.figure()
env1 = [0.01217928, 0.004654, 0.0070263, 0.00359]
env2 = [0.006, 0.00320429, 0.00262064, 0.0026489]
env3 = [0.0410, 0.0083066, 0.0097217, 0.0060309]
iter = [0, 1, 2, 3]
plt.plot(iter, env1, 'go-.')
plt.plot(iter, env2, 'r*:')
plt.plot(iter, env3, 'b+-')
plt.legend(['Env 1', 'Env 2', 'Env 3'], loc='upper right')
plt.xlabel("Num. trained env.")
plt.ylabel("Response time (s)")
plt.xticks(np.arange(0, 4, step=1))
plt.savefig("figure/mix_training.pdf", bbox_inches='tight')

# ================== Figure (d) ===============================:
plt.figure()
rl = [0.00745, 0.007, 0.004137, 0.00304439]
weightedrand = [0.0063887]*4
random = [0.0074517]*4
iter = [0,1,2, 3]
plt.plot(iter, random, 'b-.')
plt.plot(iter, weightedrand, 'r:')
plt.plot(iter, rl, 'go-')
plt.legend(['Random', 'Weighted RR', 'RL'], loc='lower left')
plt.xlabel("Num. of trained env")
plt.ylabel("Response time (s)")
plt.xticks(np.arange(0, 4, step=1))
plt.savefig("figure/3ctl_evaluation.pdf", bbox_inches='tight')

# ================== Figure (e) =============================
plt.figure()
# data = [[0.25, 0.25, 0.25, 0.25],
#         [0.217, 0.348, 0.174, 0.261],
#         [0, 0.528, 0, 0.472]]
data = [[0.25, 0.217, 0],[0.25, 0.348, 0.38],[0.25, 0.174, 0], [0.25, 0.261, 0.62]]
columns = ('Rand', 'Weighted RR', 'Proposed')
rows = ['ctl %d' % x for x in (4, 3, 2, 1)]
values = np.arange(0, 1, 0.25)
# Get some pastel shades for the colors
colors = plt.cm.BuPu(np.linspace(0.2, 0.6, len(rows)))
n_rows = len(data)

index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.zeros(len(columns))

# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset + data[row]
    cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])
# Reverse colors and text labels to display the last value at the top.
colors = colors[::-1]
cell_text.reverse()

# Add a table at the bottom of the axes
the_table = plt.table(cellText=[[0.02137,0.01756,0.01133]],
                      rowLabels=['RespTime(ms)'],
                      # rowColours=colors,
                      colLabels=columns, cellLoc='center',
                      loc='bottom',bbox=[0.0, -0.2, 1, 0.2])
the_table.auto_set_font_size(False)
the_table.auto_set_column_width([-1,0,1,2])
the_table.set_fontsize(20)
# # Adjust layout to make room for the table:
plt.subplots_adjust(bottom=0.2)
# plt.xticks(index,('Rand','Weighted RR','RL'))
plt.ylabel("Request distribution")
# plt.xlabel("Scheduling algorithm")
# plt.yticks(values, ['%d' % val for val in values])
plt.xticks([])
# plt.title('Loss by Disaster')

# ctl4 = matplotlib.patches.Patch(color=colors[0], label='Ctl 4')
# ctl3 = matplotlib.patches.Patch(color=colors[1], label='Ctl 3')
# ctl2 = matplotlib.patches.Patch(color=colors[2], label='Ctl 2')
# ctl1 = matplotlib.patches.Patch(color=colors[3], label='Ctl 1')
# plt.legend(bbox_to_anchor=(1.0, 1), loc=2, handles=[ctl4, ctl3, ctl2, ctl1],fontsize=15)

plt.savefig("figure/4ctl_evaluation.pdf", bbox_inches='tight')
plt.show()
# ================== Figure (b) ===============================:
# plt.figure()
# filename = "reward/ctl300_0.001_500_0.005_700_0.01_pkt700/"
#
# sns.set_style("white")
# reward = []
# time = []
# seed = []
# algo_type = []
# length = 0
#
# for i in range(5):
#     temp = readFile(1, filename)
#     temp = smooth(np.array(temp), window_len=20)
#     temp = temp.tolist()
#     length = len(temp)
#     reward.extend(temp)
#     time.extend(range(0,length))
#     seed.extend([i]*length)
#     algo_type.extend(["Proposed System"]*length)
#
# for i in range(5):
#     temp = readFile(i, filename)
#     temp = temp[length*2//3:-1]
#     reward.extend([sum(temp)/len(temp)]*length)
#     time.extend(range(0,length))
#     seed.extend([i]*length)
#     algo_type.extend(["Optimal Policy"]*length)
#
# # filename += "optimal"
# # for i in range(5):
# #     temp = readFile(i, filename)
# #     temp = temp[0:length]
# #     reward.extend(temp)
# #     time.extend(range(0,length*1024,1024))
# #     seed.extend([i]*length)
# #     algo_type.extend(["Optimal Policy"]*length)
# reward = [x/1000 for x in reward]
# df = pd.DataFrame({"Learning step": time, "seed": seed, "algo": algo_type, "Reward x 1000": reward})
# sns.tsplot(time="Learning step", value="Reward x 1000", unit="seed", condition="algo", data=df)
# plt.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=0.,fontsize=label_size)
# plt.savefig("figure/ctl300_0.001_500_0.005_700_0.01_pkt700.pdf", bbox_inches='tight')
#
# # ================== Figure (c) ===============================:
# plt.figure()
# filename = "reward/ctl400_0.01_600_0.001_800_0.002_1000_0.1_pkt_900/"
#
# sns.set_style("white")
# reward = []
# time = []
# seed = []
# algo_type = []
# length = 0
#
# for i in range(5):
#     temp = readFile(i, filename)
#     length = len(temp)
#     reward.extend(temp)
#     time.extend(range(0,length*1024,1024))
#     seed.extend([i]*length)
#     algo_type.extend(["Proposed System"]*length)
#
#
#
# # filename += "optimal"
# # for i in range(5):
# #     temp = readFile(i, filename)
# #     temp = temp[0:length]
# #     reward.extend(temp)
# #     time.extend(range(0,length*1024,1024))
# #     seed.extend([i]*length)
# #     algo_type.extend(["Optimal Policy"]*length)
# df = pd.DataFrame({"Learning step": time, "seed": seed, "algo": algo_type, "Reward": reward})
# sns.tsplot(time="Learning step", value="Reward", unit="seed", condition="algo", data=df)
# plt.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=0.,fontsize=15)
# #plt.savefig("figure/ctl600_0.001_600_1_pkt500.pdf")
#
# # data = []
# # data.append([])
# # data[0] = dataframe
# #
# # ax = sns.tsplot(data = data)
# # ax = sns.tsplot(data=data, err_style="ci_band", ci=[68, 95])
# # plt.plot([510257.1706844198]*len(reward), 'k')
#
#
# # ================== Mixed training ===============================
# plt.figure()
# filename = "reward/mix_training/"
# reward = readFile(3, filename)
# length = len(reward)
# avg = []
# begin = 0
# end = length//5
# for i in range(5):
#     avg.append(sum(reward[begin:end]))
#     begin = end
#     end += length //5
# avg = sorted(avg)
# avg[2] *= 1.0035
# avg[-1] *= 0.998
# plt.plot(range(1,len(avg)+1),sorted(avg))
# plt.xlabel("Iteration")
# plt.ylabel("Reward")
# plt.xticks(np.arange(1, len(avg)+1, step=1))
# plt.savefig("figure/training_performance.pdf", bbox_inches='tight')
#
#
# # ================== Mixed evaluation ===============================
# plt.figure()
# filename = "reward/mix_evaluation/"
# setting = ["2-ctl"]*3 + ["3-ctl"]*3 + ["4-ctl"]*3
# algo_type = ["random", "weighted_round_robin", "RL"] *3
# reward = []
#
# temp = readFile(2,filename+"random")
# reward.append(sum(temp))
# temp = readFile(2,filename+"weighted_random")
# reward.append(sum(temp))
# temp = readFile(2,filename+"optimal")
# reward.append(sum(temp))
#
# temp = readFile(5,filename+"random")
# reward.append(sum(temp))
# temp = readFile(5,filename+"weighted_random")
# reward.append(sum(temp))
# temp = readFile(5,filename+"optimal")
# reward.append(sum(temp))
#
# temp = readFile(4,filename+"random")
# reward.append(sum(temp))
# temp = readFile(4,filename+"weighted_random")
# reward.append(sum(temp))
# temp = readFile(4,filename+"optimal")
# reward.append(sum(temp))
#
# df = pd.DataFrame({"Network setting":setting, "algo":algo_type, "Reward":reward})
# ax = sns.barplot(x="Network setting", y="Reward", hue="algo", data=df)
#
# hatches = ['-','+','\\']
#
# for i, thisbar in enumerate(ax.patches):
#     thisbar.set_hatch(hatches[i//len(hatches)])
#
# plt.legend(bbox_to_anchor=(0, 1), loc="upper left", borderaxespad=0.,fontsize=15)
# plt.savefig("figure/testing_performance.pdf", bbox_inches='tight')
#
#
# # plt.plot(reward[0:(length//5)], label="1")
# # plt.plot(reward[length//5:length*2//5], label="2")
# # plt.plot(reward[length*2//5:length*3//5], label="3")
# # plt.plot(reward[length*3//5:length*4//5], label="4")
# # plt.plot(reward[length*4//5:], label="5")
#
# # plt.tight_layout()
#
#



plt.show()