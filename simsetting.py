# total pkt arrival rate = 1e7
# same controller capacity 700K
# scheduler to controller latency sample from a linear distribution from 1 to 50ms


class Setting(object):

    # RANDOM = 1123
    # WEIGHTED_RANDOM = 882
    # ROUND_ROBIN = 213
    # WEIGHTED_ROUND_ROBIN = 3213
    # RESP_RANDOM = 12435
    # RESP_WEIGHTED = 61958
    # IMPROVE_RANDOM = 589643
    # GLOBAL_WEIGHTED_RANDOM = 5215
    # GRADIENT_WEIGHTED = 34214
    # CAPACITY_WEIGHT = 59872
    # REINFORCEMENT_LEARNING = 12345

    def __init__(self, num):
        algorithm = ['RANDOM', 'WEIGHTED_RANDOM', 'RL', 'OPTIMAL']
        self.algo = algorithm[0]
        self.history_len = 3  # the number of history data (response time & utilization) for each variable will be used
        self.respTime_update_interval = 0.5  # (sec) the time interval used in averaging the response time
        self.util_update_interval = 0.5
        self.timeStep = 0.01
        self.probe_overhead = 0
        # self.algorithm = REINFORCEMENT_LEARNING

        self.maxSimTime = 240
        self._init(num)

    # ================================================================================
    # setting 0: used in RL scheduling test
    def _init(self, num):
        if num == 1:
            self.ctlNum = 2
            self.schNum = 1
            self.pktRate = [500]  # packet arrival rate for each scheduler
            self.ctlRate = [1000, 1000]  # controller: number of packets processed by the controller within one second
            # Round Trip Time(RTT) 0.1ms-1ms
            self.sch2ctlLink = {0: [0.002, 0.02]}

        elif num == 0:
            self.ctlNum = 2
            self.schNum = 1
            self.pktRate = [400]  # packet arrival rate for each scheduler
            self.ctlRate = [1000, 500]  # controller: number of packets processed by the controller within one second
            # Round Trip Time(RTT) 0.1ms-1ms
            self.sch2ctlLink = {0: [0.001, 0.001]}

        elif num == 2:
            self.ctlNum = 2
            self.schNum = 1
            self.pktRate = [600]
            self.ctlRate = [900, 900]
            self.sch2ctlLink = {0: [0.005, 0.08]}
        #
        # elif num == 3:
        #     self.ctlNum = 2
        #     self.schNum = 1
        #     self.pktRate = [800]
        #     self.ctlRate = [1000, 600]
        #     self.sch2ctlLink = {0: [0.005, 0.05]}

        # elif num == 4:
        #     self.ctlNum = 2
        #     self.schNum = 1
        #     self.pktRate = [900]
        #     self.ctlRate = [800, 1000]
        #     self.sch2ctlLink = {0:[0.001, 0.01]}

        # if num == 0:
        #     self.ctlNum = 2
        #     self.schNum = 1
        #     self.pktRate = [500]  # packet arrival rate for each scheduler
        #     self.ctlRate = [600, 600]  # controller: number of packets processed by the controller within one second
        #     # Round Trip Time(RTT) 0.1ms-1ms
        #     self.sch2ctlLink = {0: [0.001, 0.01]}
        # elif num == 1:
        #     self.ctlNum = 2
        #     self.schNum = 1
        #     self.pktRate = [600]  # packet arrival rate for each scheduler
        #     self.ctlRate = [600, 600]  # controller: number of packets processed by the controller within one second
        #     # Round Trip Time(RTT) 0.1ms-1ms
        #     self.sch2ctlLink = {0: [0.001, 0.001]}
        # elif num == 2:
        #     self.ctlNum = 2
        #     self.schNum = 1
        #     self.pktRate = [400]  # packet arrival rate for each scheduler
        #     self.ctlRate = [900, 600]  # controller: number of packets processed by the controller within one second
        #     # Round Trip Time(RTT) 0.1ms-1ms
        #     self.sch2ctlLink = {0: [0.001, 0.001]}
        # elif num == 3:
        #     self.ctlNum = 2
        #     self.schNum = 1
        #     self.pktRate = [600]
        #     self.ctlRate = [300, 800]
        #     self.sch2ctlLink = {0: [0.001, 0.005]}
        # elif num == 4:
        #     self.ctlNum = 2
        #     self.schNum = 1
        #     self.pktRate = [700]  # packet arrival rate for each scheduler
        #     self.ctlRate = [400, 800]  # controller: number of packets processed by the controller within one second
        #     # Round Trip Time(RTT) 0.1ms-1ms
        #     self.sch2ctlLink = {0: [0.005, 0.001]}
        #
        # else:
        #     self.ctlNum = 2
        #     self.schNum = 1
        #     self.pktRate = [2000]  # packet arrival rate for each scheduler
        #     self.ctlRate = [1000, 1500]  # controller: number of packets processed by the controller within one second
        #     # Round Trip Time(RTT) 0.1ms-1ms
        #     self.sch2ctlLink = {0: [0.001, 0.01]}
        # else:
        #     self.ctlNum = 3
        #     self.schNum = 3
        #     self.pktRate = [300, 300, 300]  # packet arrival rate for each scheduler
        #         # [150, 150, 150, 150, 150, 150]
        #         # [100, 100, 100, 100, 100, 100, 100, 100, 100]
        #         # [75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75]
        #
        #     self.ctlRate = [1000, 600, 500]  # controller: number of packets processed by the controller within one second
        #     # Round Trip Time(RTT) 0.1ms-1ms
        #     self.sch2ctlLink = {0: [0.0003, 0.0004, 0.0005], 1: [0.0001, 0.0002, 0.0005], 2: [0.0002, 0.0004, 0.0003]}
        #         # {0: [0.001, 0.01, 0.005], 1: [0.001, 0.001, 0.01], 2: [0.005, 0.001, 0.01]}
        #         # {0: [0.001, 0.01, 0.005], 1: [0.001, 0.001, 0.02], 2: [0.005, 0.001, 0.02], 3: [0.002, 0.01, 0.001], 4: [0.001, 0.001, 0.001], 5: [0.001, 0.01, 0.03]}
        #         # {0: [0.001, 0.01, 0.005], 1: [0.001, 0.001, 0.02], 2: [0.005, 0.001, 0.02], 3: [0.002, 0.01, 0.001], 4: [0.001, 0.01, 0.001], 5: [0.001, 0.01, 0.03], 6: [0.001, 0.02, 0.01], 7: [0.002, 0.03, 0.02], 8: [0.001, 0.01, 0.001]}
        #         # {0: [0.001, 0.01, 0.005], 1: [0.001, 0.001, 0.02], 2: [0.001, 0.005, 0.02], 3: [0.002, 0.01, 0.001], 4: [0.001, 0.01, 0.001], 5: [0.001, 0.005, 0.03], 6: [0.001, 0.02, 0.01], 7: [0.002, 0.03, 0.02], 8: [0.001, 0.01, 0.001], 9: [0.001, 0.01, 0.02], 10: [0.001, 0.01, 0.01], 11: [0.001, 0.01, 0.03]}
        #         # {0: [0.001, 0.01, 0.005]}
        # else:
        #     self.ctlNum = 4
        #     self.schNum = 1
        #     self.pktRate = [1000]  # packet arrival rate for each scheduler
        #     self.ctlRate = [500, 800, 400, 600]  # controller: number of packets processed by the controller within one second
        #     # Round Trip Time(RTT) 0.1ms-1ms
        #     self.sch2ctlLink = {0: [0.01, 0.01, 0.05, 0.001]}
        else:
            self.ctlNum = 3
            self.schNum = 6
            self.pktRate = [3333, 3333, 3333, 3333, 3334, 3334]
                # [2500, 2500, 2500, 2500, 2500, 2500]
                # [1785, 1786, 1785, 1786, 1785, 1786, 1785, 1786, 1786, 1786, 1786, 1786, 1786, 1786]
                # [1071, 1072, 1071, 1072, 1071, 1072, 1071, 1072, 1071, 1072, 1071, 1072, 1071, 1072]
                # [6000, 6000, 6000]  #[500, 500, 500]
            # self.pktRate = [250, 250, 250, 250, 250, 250]
            # self.pktRate = [166, 167, 166, 167, 166, 167, 166, 167, 166]
            # self.pktRate = [125,125,125,125,125,125,125,125,125,125,125,125]  # packet arrival rate for each scheduler
            self.ctlRate = [6000, 9000, 12000]  # controller: number of packets processed by the controller within one second
            # Round Trip Time(RTT) 0.1ms-1ms
            self.sch2ctlLink = {0:[0.0,0.038,0.092],
1:[0.038,0.0,0.086],
2:[0.078,0.072,0.0175],
3:[0.078,0.072,0.015],
4:[0.075,0.069,0.017],
5:[0.092,0.086,0.0]}
#                 {0:[0.0,0.0375,0.053,0.026],
# 1:[0.0175,0.055,0.036,0.0125],
# 2:[0.0105,0.03,0.0655,0.037],
# 3:[0.0375,0.0,0.091,0.0635],
# 4:[0.024,0.015,0.0775,0.05],
# 5:[0.0125,0.0505,0.0665,0.039],
# 6:[0.053,0.091,0.0,0.0475],
# 7:[0.037,0.0755,0.022,0.032],
# 8:[0.0435,0.083,0.011,0.0405],
# 9:[0.0405,0.0775,0.025,0.035],
# 10:[0.02,0.0575,0.038,0.0155],
# 11:[0.0345,0.0725,0.02,0.03],
# 12:[0.0275,0.0185,0.08,0.0535],
# 13:[0.026,0.0635,0.0475,0.0]}
                # {0: [0.01, 0.001, 0.1, 0.001], 1:[0.001, 0.4, 0.3, 0.002], 2:[0.01, 0.001, 0.001, 0.001]}
                # {0: [0.01, 0.001, 0.1, 0.001], 1: [0.001, 0.4, 0.3, 0.002], 2: [0.01, 0.001, 0.001, 0.001], 3: [0.01, 0.001, 0.1, 0.001], 4: [0.001, 0.4, 0.3, 0.002], 5: [0.05, 0.1, 0.001, 0.1]}

                # {0: [0.01, 0.001, 0.1, 0.001], 1:[0.001, 0.4, 0.3, 0.002], 2:[0.01, 0.001, 0.001, 0.001]}
                # {0: [0.01, 0.001, 0.1, 0.001], 1:[0.001, 0.4, 0.3, 0.002], 2:[0.01, 0.001, 0.001, 0.001], 3: [0.01, 0.001, 0.1, 0.001], 4: [0.001, 0.4, 0.3, 0.002], 5: [0.05, 0.1, 0.001, 0.1], 6: [0.01, 0.001, 0.1, 0.001], 7: [0.001, 0.4, 0.3, 0.002], 8: [0.05, 0.1, 0.001, 0.1]}
                # {0: [0.01, 0.001, 0.1, 0.001],
                #                 1:[0.001, 0.4, 0.3, 0.002],
                #                 2:[0.01, 0.001, 0.001, 0.001],
                #                 3: [0.01, 0.001, 0.1, 0.001],
                #                 4: [0.001, 0.4, 0.3, 0.002],
                #                 5: [0.05, 0.1, 0.001, 0.1],
                #                 6: [0.01, 0.001, 0.1, 0.001],
                #                 7: [0.001, 0.4, 0.3, 0.002],
                #                 8: [0.5, 0.001, 0.001, 0.4],
                #                 9: [0.001, 0.001, 0.1, 0.1],
                #                 10: [0.07, 0.004, 0.003, 0.2],
                #                 11: [0.005, 0.001, 0.1, 0.5]
                #                 }












    # ================================================================================
    # setting 1: within a data center same capacity, latency < 1ms
    # self.ctlNum = 3
    # self.schNum = 2
    # self.pktRate = [80000, 54000]  # packet arrival rate for each scheduler
    # self.schRate = [50000000] * self.schNum  # scheduler: number of packets processed by the scheduler within one second
    # self.ctlRate = [45000] * self.ctlNum  # controller: number of packets processed by the controller within one second
    # # Round Trip Time(RTT) 0.1ms-1ms
    # self.sch2ctlLink = {0: [0.0005, 0.0003, 0.0009], 1: [0.0003, 0.0002, 0.0005]}
    # self.arrivalRate = 134000

    # ================================================================================
    # setting 2: within a data center different capacity, latency < 1ms
    # ctlNum = 4
    # schNum = 2
    # pktRate = [70000, 55000]  # packet arrival rate for each scheduler
    # schRate = [
    #                        50000000] * schNum  # scheduler: number of packets processed by the scheduler within one second
    # ctlRate = [
    #                        45000, 45000, 30000, 30000 ]  # controller: number of packets processed by the controller within one second
    # # Round Trip Time(RTT) 0.1ms-1ms
    # sch2ctlLink = {0: [0.0002, 0.0001, 0.0002, 0.0002], 1: [0.0002, 0.0001, 0.0002, 0.0002]}

    # ================================================================================
    # setting 3: in a cloud different capacity, latency 1-50ms
    # ctlNum = 16
    # schNum = 10
    #     # pkt_in rate: 720000, 700000, 680000, 660000, 640000, 620000, 600000
    #     # self.pktRate = [47670, 26731, 93693, 30112, 12079, 125053, 95847, 50168, 132612, 126035]  # 740000
    # pktRate = [22000] * schNum
    #     # self.pktRate = [70116, 24421, 67667, 54294, 31180, 47150, 76220, 40787, 11780, 25488, 30150, 67244, 51570, 82820, 37163, 1950]  # 720000
    #     # self.pktRate = [61950, 4816, 27565, 46235, 62340, 9122, 49410, 12685, 57012, 35060, 57792, 48510, 54891, 63171, 59751, 49690] # 700000
    #     # self.pktRate = [11604, 83940, 106760, 57880, 6055, 25471, 65923, 4590, 3187, 25204, 61869, 8406, 51953, 38158, 49713, 79287]  #680000
    #     # self.pktRate = [30737, 53430, 18324, 9486, 50236, 2192, 41423, 74280, 4985, 18212, 85070, 91714, 10680, 2476, 83410, 83345]  # 660000
    #     # self.pktRate = [43045, 11896, 80706, 39306, 55435, 35949, 20029, 17180, 25981, 55958, 1880, 40133, 61854, 74093, 68908, 7647] # 640000
    #     # self.pktRate = [9411, 1555, 47424, 37719, 54297, 67040, 17788, 32117, 67723, 52078, 8577, 64304, 55952, 15940, 52626, 35449] # 620000
    #     # self.pktRate = [19932, 20174, 56333, 24351, 31474, 75798, 19105, 55385, 5806, 65549, 3209, 15234, 82165, 72589, 1907, 50989]  # 600000
    #     # self.pktRate = [45599, 5173, 44662, 26704, 23785, 57342, 54913, 2332, 54549, 8335, 23767, 15980, 56650, 35856, 18606, 25747]  # 500000
    #     # self.pktRate = [20314, 44752, 20428, 29182, 22771, 14983, 39678, 28021, 36220, 24034, 44446, 10075, 10283, 14348, 12841, 27624]  # 400000
    #     # self.pktRate = [14212, 3889, 383, 28826, 21413, 29630, 40665, 38367, 30013, 20056, 38034, 10519, 18250, 16983, 35094, 3666] # 350000
    #
    # schRate = [50000] * schNum  # scheduler: number of packets processed by the scheduler within one second
    # ctlRate = [45000, 45000, 30000, 30000, 30000, 60000, 60000, 60000, 60000, 90000, 90000, 90000, 15000, 15000, 15000,
    #          15000]  # controller: number of packets processed by the controller within one second
    #     # Round Trip Time(RTT) 0.1ms-1ms
    # sch2ctlLink = {0: [0.001, 0.011, 0.003, 0.013, 0.02, 0.009, 0.004, 0.005, 0.007, 0.012, 0.009, 0.002, 0.002, 0.006, 0.015, 0.009],
    #                         1: [0.001, 0.011, 0.003, 0.013, 0.02, 0.009, 0.004, 0.005, 0.007, 0.012, 0.009, 0.002, 0.002, 0.006, 0.015, 0.009],
    #                         2: [0.001, 0.011, 0.003, 0.013, 0.02, 0.009, 0.004, 0.005, 0.007, 0.012, 0.009, 0.002, 0.002, 0.006, 0.015, 0.009],
    #                         3: [0.001, 0.011, 0.003, 0.013, 0.02, 0.009, 0.004, 0.005, 0.007, 0.012, 0.009, 0.002, 0.002, 0.006, 0.015, 0.009],
    #                         4: [0.001, 0.011, 0.003, 0.013, 0.02, 0.009, 0.004, 0.005, 0.007, 0.012, 0.009, 0.002, 0.002, 0.006, 0.015, 0.009],
    #                         5: [0.001, 0.011, 0.003, 0.013, 0.02, 0.009, 0.004, 0.005, 0.007, 0.012, 0.009, 0.002, 0.002, 0.006, 0.015, 0.009],
    #                         6: [0.001, 0.011, 0.003, 0.013, 0.02, 0.009, 0.004, 0.005, 0.007, 0.012, 0.009, 0.002, 0.002, 0.006, 0.015, 0.009],
    #                         7: [0.001, 0.011, 0.003, 0.013, 0.02, 0.009, 0.004, 0.005, 0.007, 0.012, 0.009, 0.002, 0.002, 0.006, 0.015, 0.009],
    #                         8: [0.001, 0.011, 0.003, 0.013, 0.02, 0.009, 0.004, 0.005, 0.007, 0.012, 0.009, 0.002, 0.002, 0.006, 0.015, 0.009],
    #                         9: [0.001, 0.011, 0.003, 0.013, 0.02, 0.009, 0.004, 0.005, 0.007, 0.012, 0.009, 0.002, 0.002, 0.006, 0.015, 0.009]}

        # setting 2: within a cloud
        # self.ctlNum = 18
        # self.schNum = 10
        # self.pktRate = [477717, 1005139, 777284, 1014652, 615509, 1488495, 1495887, 1015201, 984546,
        #                 1125570]  # packet arrival rate for each scheduler
        # self.schRate = [
        #                    50000000] * self.schNum  # scheduler: number of packets processed by the scheduler within one second
        # self.ctlRate = [
        #                    700000] * self.ctlNum  # controller: number of packets processed by the controller within one second
        # self.sw2ctlLink = [0.002, 0.001, 0.001, 0.004, 0.002]  # Round Trip Time(RTT)
        # self.sch2ctlLink = {
        #     0: [0.011, 0.010, 0.015, 0.019, 0.045, 0.041, 0.007, 0.030, 0.035, 0.034, 0.027, 0.014, 0.014, 0.010, 0.035,
        #         0.010, 0.033, 0.021],
        #     1: [0.017, 0.031, 0.010, 0.011, 0.045, 0.002, 0.001, 0.042, 0.030, 0.044, 0.040, 0.038, 0.025, 0.003, 0.048,
        #         0.047, 0.022, 0.049],
        #     2: [0.003, 0.048, 0.038, 0.008, 0.045, 0.014, 0.017, 0.041, 0.033, 0.014, 0.030, 0.034, 0.039, 0.044, 0.038,
        #         0.024, 0.034, 0.021],
        #     3: [0.003, 0.022, 0.020, 0.029, 0.001, 0.016, 0.043, 0.027, 0.014, 0.037, 0.017, 0.047, 0.005, 0.015, 0.012,
        #         0.046, 0.003, 0.010],
        #     4: [0.038, 0.034, 0.045, 0.048, 0.015, 0.042, 0.010, 0.029, 0.038, 0.041, 0.007, 0.025, 0.008, 0.001, 0.018,
        #         0.024, 0.024, 0.049],
        #     5: [0.044, 0.032, 0.011, 0.013, 0.008, 0.004, 0.036, 0.022, 0.026, 0.028, 0.028, 0.047, 0.031, 0.033, 0.021,
        #         0.032, 0.006, 0.048],
        #     6: [0.012, 0.039, 0.020, 0.037, 0.040, 0.001, 0.038, 0.035, 0.020, 0.041, 0.046, 0.045, 0.017, 0.028, 0.019,
        #         0.014, 0.008, 0.045],
        #     7: [0.032, 0.046, 0.034, 0.044, 0.025, 0.017, 0.002, 0.046, 0.046, 0.004, 0.038, 0.014, 0.035, 0.026, 0.018,
        #         0.037, 0.014, 0.008],
        #     8: [0.09, 0.031, 0.029, 0.046, 0.034, 0.005, 0.008, 0.027, 0.035, 0.036, 0.031, 0.028, 0.041, 0.044, 0.043,
        #         0.021, 0.007, 0.049],
        #     9: [0.045, 0.018, 0.015, 0.023, 0.012, 0.025, 0.048, 0.008, 0.025, 0.042, 0.020, 0.022, 0.010, 0.001, 0.032,
        #         0.025, 0.043, 0.046]}

    # def __init__(self):
    #     self.timeStep = 0.01
    #     self.maxSimTime = 100
    #     self.ctlNum = 5
    #     self.schNum = 3
    #     self.pktRate = [600, 800, 800]  # packet arrival rate for each scheduler
    #     self.schRate = [
    #                        500000] * self.schNum  # scheduler: number of packets processed by the scheduler within one second
    #     self.ctlRate = [600, 300, 600, 400,
    #                     800]  # controller: number of packets processed by the controller within one second
    #     self.sw2ctlLink = [0.002, 0.001, 0.001, 0.004, 0.002]  # Round Trip Time(RTT)
    #     self.sch2ctlLink = {0: [0.004, 0.001, 0.002, 0.003, 0.002], 1: [0.001, 0.001, 0.002, 0.004, 0.001],
    #                         2: [0.001, 0.002, 0.004, 0.003, 0.002]}
    #     self.probe_overhead = 0
    #     self.algorithm = self.RESP_WEIGHTED