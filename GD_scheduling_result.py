import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd

label_size = 25
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams.update({'font.size': label_size})
matplotlib.rcParams['lines.linewidth'] = 5
matplotlib.rcParams["legend.markerscale"] = 0.8

# arrival_rate = range(15, 26, 2)
# arrival_rate.extend([26, 27])

color = [(0.75, 0.75, 0.75), (0.5, 0.5, 0.5), (0., 0., 0.), (0.0, 0.83, 1.0), (1.0, 0.6, 0.0), 'r']
linestyle = ['-', '-', '-', '-.', ':', '--']
markers = ["", "D", "o", "v", "", "^", "p", ">"]
temp = -1


def calculate_stat_in_arr(vec):
    avg = []
    for i in range(len(vec)//3):
        temp = vec[i*3:i*3+3]
        avg.append(np.mean(temp))
    return avg


def draw_line(t, y, col, ls, label):
    global temp
    temp = (temp + 1) % len(markers)
    plt.plot(t, y, color=col, ls=ls, label=label, marker=markers[temp], markersize=15)


def generate_arrival_rate_vector(arrival_rate):
    for i in range(15, 26, 2):
        arrival_rate.extend([i] * 3)
    arrival_rate.extend([26] * 3)
    arrival_rate.extend([27] * 3)
    return arrival_rate


def generate_algorithm_vector(arr, alg, oneTypelen):
    for i in range(len(alg)):
        arr.extend([alg[i]] * oneTypelen)
    return arr



# overall average response time
rand_tot_resp = [0.16653492890052926, 0.16654499864168212, 0.16647510100232848, 0.16725935068211834,
                 0.16722315413591468,
                 0.16724699143864094, 2.2171767514511194, 2.1616097606977918, 2.1611705037870252, 5.362145908319148,
                 5.313420253435226, 5.285790773325493, 7.537931106980123, 7.500336449973662, 7.503629685335224,
                 9.081815108515347, 9.120351862201385, 9.06605462077843, 9.691400161175812, 9.75432572344135,
                 9.675218976476149, 10.227810123702413, 10.286009865986347, 10.215182921049431]
wrr_tot_resp = [0.17427201076864215, 0.17420745227805431, 0.17427040928467377, 0.17432377614785322, 0.1743651191737322,
                0.17425646832571087, 0.17434762720637342, 0.1743212917454128, 0.1743412795507568, 0.17444693247226545,
                0.17462020456075414, 0.174485618361112, 0.17476040457950665, 0.17476721664677475, 0.174747583579365,
                0.17548361100147483, 0.17549175355721106, 0.17551164725454596, 0.17707289792815034, 0.17696950648917129,
                0.1769325284121431, 0.28845127379890984, 0.24765248549083727, 0.29409354751608935]
gd_tot_resp = [0.11792116233783909, 0.11789071597023432, 0.11792746896923786, 0.11921432221937636, 0.11919952913227508,
               0.11926996191138865, 0.11707048054670173, 0.11707309926604764, 0.11709028082972875, 0.11528303784692374,
               0.11532942713573936, 0.11533180555897708, 0.11363318989973191, 0.11366968347089858, 0.11366137850031809,
               0.1197888613845835, 0.11987169880660956, 0.1199783221641246, 0.1636732770609535, 0.163630315985288,
               0.1635617060549714, 0.4610808752904897, 0.34807327538779614, 0.4131092243184982]

# overall throughput
rand_tot_throughput = [3598679, 3598724, 3598781, 4078530, 4078638, 4078595, 4479469, 4478881, 4486622, 4796484,
                       4805367,
                       4805583, 5117836, 5119964, 5117746, 5437443, 5436300, 5436067, 5597120, 5604467, 5597452,
                       5758831,
                       5764267, 5758655]
wrr_tot_throughput = [3598668, 3598645, 3598676, 4078491, 4078571, 4078550, 4558347, 4558331, 4558352, 5038219, 5038238,
                      5038136, 5518081, 5518042, 5518056, 5997871, 5997821, 5997759, 6237778, 6237777, 6237723, 6474213,
                      6477683, 6475312]
gd_tot_throughput = [3599104, 3599088, 3599098, 4078968, 4079027, 4078943, 4558886, 4558904, 4558905, 5038853, 5038815,
                     5038776,
                     5518772, 5518720, 5518719, 5998562, 5998493, 5998489, 6237962, 6237944, 6237897, 6463058, 6477865,
                     6467202]
rand_tot_throughput = np.array(rand_tot_throughput) / 240.0
wrr_tot_throughput = np.array(wrr_tot_throughput) / 240.0
gd_tot_throughput = np.array(gd_tot_throughput) / 240.0

algo = []
algo = generate_algorithm_vector(algo, ["Rand", "WeightRR", "GD"], len(rand_tot_resp))
algo = generate_algorithm_vector(algo, ["Rand", "WeightRR", "GD"], len(rand_tot_resp))

arrival_rate = []
for i in range(6):
    arrival_rate = generate_arrival_rate_vector(arrival_rate)

value = rand_tot_resp+wrr_tot_resp+gd_tot_resp
throughput = []
throughput.extend(rand_tot_throughput)
throughput.extend(wrr_tot_throughput)
throughput.extend(gd_tot_throughput)
value += throughput
print(len(rand_tot_throughput), len(wrr_tot_throughput), len(gd_tot_throughput), len(value), len(algo), len(arrival_rate))

metric = ["response Time"]*len(throughput)
metric.extend(["throughput"]*len(throughput))

df = pd.DataFrame({"metric": metric, "algo":algo, "arrival_rate":arrival_rate, "value":value})
sns.relplot(x="arrival_rate", y="value", col="metric", hue="algo", style="algo", kind='line', data=df)

# calculate the mean value of response time and throughput
rand_tot_resp_avg = calculate_stat_in_arr(rand_tot_resp)
wrr_tot_resp_avg = calculate_stat_in_arr(wrr_tot_resp)
gd_tot_resp_avg = calculate_stat_in_arr(gd_tot_resp)

rand_tot_throughput_avg = calculate_stat_in_arr(rand_tot_throughput)
wrr_tot_throughput_avg = calculate_stat_in_arr(wrr_tot_throughput)
gd_tot_throughput_avg = calculate_stat_in_arr(gd_tot_throughput)


plt.show()


