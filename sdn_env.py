import numpy as np
import random
from stats import Stats
from simsetting import Setting
from packet import Packet
from simqueue import SimQueue
import gym
import logging
from collections import deque

def sample_possion(rate, time):  # Sample from a possion process
    pos_array = []
    current = 0
    while True:
        pos = -(np.log(1 - random.random())) / rate
        current += pos
        if current < time:
            pos_array.append(current)
        else:
            return pos_array


class sdn_simulator(object):


    # vf_ob = [tot_ctlRate, tot_latency, tot_arrRate, tot_respTime(history_len), tot_util(history_len)]
    # ac_ob = [ [ctlRate, latency, sch_arrRate, ctlResp(history_len), ctl_util(history_len)],  [], [], ...]

    def __init__(self, ts_per_actorbatch, num, envseed):
        self.envid = num
        self.seed = envseed
        self.set = Setting(self.envid)
        self.ts_per_actorbatch = ts_per_actorbatch
        # variables used for regularization
        # self.vf_sum = np.array([0.] * 9)
        # self.vf_sumsq = np.array([0.] * 9)
        # self.vf_count = np.array([0.] * 9)
        # self.ac_sum = np.array([0.] * 9)
        # self.ac_sumsq = np.array([0.] * 9)
        # self.ac_count = np.array([0.] * 9)
        self.vf_observation_space = gym.spaces.Box(0, 100, (3 + self.set.history_len * 2,))
        # ac_observation_space = gym.spaces.Box(0, 100, (Setting.ctlNum, 3 + Setting.history_len * 2))
        self.action_space = gym.spaces.Discrete(self.set.ctlNum)
        self._init()

    def close(self):
        print("End of the episode")
        # reward = 0.
        #
        # # handle the remaining packets
        # remainPktNum = []
        # for i in range(0, self.ctlNum):
        #     remainPktNum.append(self.ctl_queues[i].queue.qsize())
        #
        # while sum(remainPktNum) != 0:
        #     for i in range(0, self.ctlNum):
        #         remainPktNum[i] = self.ctl_queues[i].queue.qsize()
        #         if remainPktNum[i] == 0:
        #             continue
        #         else:
        #             pktLeaveTime = sample_possion(Setting.ctlRate[i], Setting.timeStep)
        #             pktLeaveTime = [x + self.currentTime for x in pktLeaveTime]
        #             for x in pktLeaveTime:
        #                 firstPktTime = self.ctl_queues[i].getFirstPktTime()
        #                 if firstPktTime is None:
        #                     break
        #                 elif x < firstPktTime:
        #                     continue
        #                 else:
        #                     pkt = self.ctl_queues[i].dequeue()
        #                     t = pkt.generateTime
        #                     reward -= x-t
        #                     self.stat.add_response_time(x, pkt.scheduler, i, x - t)
        #                     logging.info("Handling remaining pkts from ctl %s" % (i))
        #                     # print("Controller response time: %s" % (x - t))
        #     self.currentTime += Setting.timeStep

    def _init(self):
        self.tot_pktNum = self.ts_per_actorbatch * (sum(self.set.pktRate) * self.set.maxSimTime //self.ts_per_actorbatch)   # used to indicate the end of an episode
        self.stat = Stats(self.set)
        self.currentTime = 0.
        self.sch_queues = []
        self.ctl_queues = []
        self.ctl_pktLeaveTime = []  # nested list, [[list]*ctlNum]
        self.schNum = self.set.schNum
        self.ctlNum = self.set.ctlNum
        for i in range(self.schNum):
            self.sch_queues.append(SimQueue())
        for i in range(self.ctlNum):
            self.ctl_queues.append(SimQueue())
            self.ctl_pktLeaveTime.append(deque([]))
        self.remainPktNum = 0
        self.tot_ctlRate = sum(self.set.ctlRate)  # used in vf_ob
        self.tot_arrRate = sum(self.set.pktRate)  # used in vf_ob
        self.tot_latency = 0.  # used in vf_ob
        for i in range(self.schNum):
            for j in range(self.ctlNum):
                self.tot_latency += self.set.pktRate[i] * self.set.sch2ctlLink[i][j] / self.tot_arrRate
        self.pkt_departure_generator()
        self.pkt_generator()
        self.sch = self.mapping_firstPkt2scheduler()  # the scheduler that has the earliest packet
        self._init_norm_para()

    def _init_ob(self):
        ac_ob = []
        for ctl in range(self.ctlNum):
            ac_ob.append([])
            ac_ob[ctl] = [self.set.ctlRate[ctl], self.set.sch2ctlLink[self.sch][ctl], self.set.pktRate[self.sch]]
            ac_ob[ctl] += [0.] * self.set.history_len * 2
        vf_ob = [self.tot_ctlRate, self.tot_latency, self.tot_arrRate]
        vf_ob += [0.] * self.set.history_len * 2
        vf_ob, ac_ob = self.min_max_norm(np.array(vf_ob), np.array(ac_ob))
        return vf_ob, ac_ob

    def _init_norm_para(self):
        maxLatency = 0.
        for sch in range(len(self.set.sch2ctlLink)):
            temp = max(self.set.sch2ctlLink[sch])
            if maxLatency < temp:
                maxLatency = temp
        # self.vf_ob_max = np.array([self.tot_ctlRate, self.tot_latency, self.tot_arrRate] + [1.1*maxLatency, 1.1*maxLatency, 1.1*maxLatency] + [1., 1., 1.])
        self.vf_ob_max = np.array([2000.0, 0.1, 1000.0] + [1., 1., 1.] + [1.0, 1.0, 1.0])
        self.vf_ob_min = np.array([500., 0., 300.] + [0., 0., 0.] + [0., 0., 0.])
        self.vf_ob_delta = self.vf_ob_max - self.vf_ob_min
        # self.ac_ob_max = np.array([max(self.set.ctlRate), maxLatency, max(self.set.pktRate)] + [1.1*maxLatency, 1.1*maxLatency, 1.1*maxLatency] + [1., 1., 1.])
        self.ac_ob_max = np.array([1000.0, 0.1, 1000.0] + [1., 1., 1.] + [1., 1., 1.])
        self.ac_ob_min = np.array([300., 0., 300.] + [0., 0., 0.] + [0., 0., 0.])
        self.ac_ob_delta = self.ac_ob_max - self.ac_ob_min

    def min_max_norm(self, vf_ob, ac_ob):
        vf_ob = (vf_ob - self.vf_ob_min)/self.vf_ob_delta
        for ctl in range(self.ctlNum):
            ac_ob[ctl] = (ac_ob[ctl] - self.ac_ob_min)/self.ac_ob_delta
        return vf_ob, ac_ob

    # def update_RunningMeanStd(self, vf_ob, ac_ob):
    #     self.vf_sum += vf_ob
    #     self.vf_sumsq += np.power(vf_ob, 2)
    #     self.vf_count += 1
    #     vf_mean = self.vf_sum/self.vf_count
    #     temp = np.maximum(self.vf_sumsq/self.vf_count-np.power(vf_mean, 2), 1e-2)
    #     vf_std = np.sqrt(temp)
    #     self.ac_sum += np.sum(ac_ob, axis=0)
    #     self.ac_sumsq += np.sum(np.power(ac_ob, 2), axis=0)
    #     self.ac_count += len(ac_ob)
    #     ac_mean = self.ac_sum / self.ac_count
    #     temp = np.maximum(self.ac_sumsq / self.ac_count - np.power(ac_mean, 2), 1e-2)
    #     ac_std = np.sqrt(temp)
    #     vf_obz = np.clip((vf_ob - vf_mean) / vf_std, -5.0, 5.0)
    #     ac_obz = np.clip((ac_ob - ac_mean) / ac_std, -5.0, 5.0)
    #     return vf_obz, ac_obz

    def pkt_generator(self):
        for i in range(self.schNum):
            nextArrivalTime = []
            while len(nextArrivalTime) == 0:
                logging.info("Packets are generated from sch")
                nextArrivalTime = sample_possion(self.set.pktRate[i], self.set.timeStep)
                nextArrivalTime = [x + self.currentTime for x in nextArrivalTime]
            logging.info("%s packets are generated" % (len(nextArrivalTime)))
            self.remainPktNum += len(nextArrivalTime)
            for x in nextArrivalTime:
                pkt = Packet(x, i)
                self.sch_queues[i].enqueue(pkt, x)
                logging.info("Put the packet %s into the scheduler queue %s" % (pkt.enqueueTime, i))
        self.currentTime += self.set.timeStep

    def pkt_departure_generator(self):
        for i in range(0, self.ctlNum):
            pktLeaveTime = []
            while len(pktLeaveTime) == 0:
                logging.info("Packets' leaving time is generated from ctl")
                pktLeaveTime = sample_possion(self.set.ctlRate[i], self.set.timeStep)
                pktLeaveTime = [x + self.currentTime for x in pktLeaveTime]
            self.ctl_pktLeaveTime[i].extend(pktLeaveTime)

    def mapping_firstPkt2scheduler(self):
        firstPktTime = []
        for i in range(self.schNum):
            temp = self.sch_queues[i].getFirstPktTime()
            if temp is None:
                logging.info("No packet in the scheduler")
                temp = 100000
            firstPktTime.append(temp)
        return firstPktTime.index(min(firstPktTime))

    def reset(self):
        self._init()
        vf_ob, ac_ob = self._init_ob()
        return vf_ob, ac_ob
        # return self.vf_observation_space.sample(), self.ac_observation_space.sample()

    # Function prototype is vf_ob, ac_ob, rew, new, _ = env.step(ac)
    def step(self, action):
        if type(action).__module__ == np.__name__:
            schDecision = action[0]
        else:
            schDecision = action
        done = False
        # Take Action: distribute the first packet from scheduler to controller
        pkt = self.sch_queues[self.sch].dequeue()
        self.remainPktNum -= 1
        self.tot_pktNum -= 1
        # todo: communication delay between switches and schedulers can be added later
        enqueueTime = pkt.generateTime + self.set.sch2ctlLink[self.sch][schDecision]  # Communication latency between schedulers and controllers
        self.ctl_queues[schDecision].enqueue(pkt, enqueueTime)
        logging.info("Put the packets into the controller queue %s" % (schDecision))

        # generate new packets for schedulers and generate packet departure time for controllers
        if self.remainPktNum == 0:
            self.pkt_departure_generator()
            self.pkt_generator()

        # update self.sch for next state and next action
        tempsch = self.sch
        self.sch = self.mapping_firstPkt2scheduler()  # the scheduler that has the earliest packet

        time = self.sch_queues[self.sch].getFirstPktTime()  # time for next state
        reward = 0.
        resp_time = 0.
        pkt_num = 0

        # remove packets from controllers from current state to next state
        for i in range(0, self.ctlNum):
            while len(self.ctl_pktLeaveTime[i]) != 0:
                x = self.ctl_pktLeaveTime[i][0]
                firstPktTime = self.ctl_queues[i].getFirstPktTime()
                if firstPktTime is None:  # no packets in ctl_queues[i]
                    break
                elif firstPktTime > time:  # finish processing packets for this time period
                    break
                elif x > time:
                    break
                elif x < firstPktTime:
                    self.ctl_pktLeaveTime[i].popleft()
                    continue
                else:
                    self.ctl_pktLeaveTime[i].popleft()
                    pkt = self.ctl_queues[i].dequeue()
                    t = pkt.generateTime
                    if tempsch == pkt.scheduler:
                        reward += 1/(x-t+ self.set.sch2ctlLink[pkt.scheduler][i])
                    resp_time += x - t + self.set.sch2ctlLink[pkt.scheduler][i]
                    pkt_num += 1
                    # reward -= x-t  # update reward

                    self.stat.add_response_time(x, pkt.scheduler, i, x - t)  # for state update
                    logging.info("Remove packets from controller queue %s" % (i))

        # todo: whether it is the end of the episode
        if self.tot_pktNum == 0:
            done = True
            logging.info("A trajectory is sampled")
            self._init()
            vf_ob, ac_ob = self._init_ob()
            # sum all the remaining packets into the reward
            # for ctl in range(self.ctlNum):
            #     remainPktNum = self.ctl_queues[ctl].queue.qsize()
            #     if remainPktNum == 0:
            #         continue
            #     else:
            #         latency = 0.
            #         for sch in range(Setting.schNum):
            #             latency += Setting.sch2ctlLink[sch][ctl]
            #         reward -= remainPktNum * ((latency/Setting.schNum)+(1.0/Setting.ctlRate))
        else:
            # update response time history for both observation space, both vf_resp_his and ac_resp_his are lists
            vf_resp_his, ac_resp_his = self.stat.update_response_time_history(self.sch)

            # update utilization info
            vf_util_his, ac_util_his = self.stat.update_utilization_history()

            vf_ob = [self.tot_ctlRate, self.tot_latency, self.tot_arrRate] + vf_resp_his + vf_util_his
            ac_ob = []
            for ctl_id in range(self.ctlNum):
                ac_ob.append([])
                ac_ob[ctl_id] = [self.set.ctlRate[ctl_id], self.set.sch2ctlLink[self.sch][ctl_id], self.set.pktRate[self.sch]]
                ac_ob[ctl_id] += ac_resp_his[ctl_id] + ac_util_his[ctl_id]

            # vf_ob, ac_ob = self.update_RunningMeanStd(np.array(vf_ob), np.array(ac_ob))
            vf_ob, ac_ob = self.min_max_norm(np.array(vf_ob), np.array(ac_ob))
        return vf_ob, ac_ob, reward, resp_time, pkt_num, done, {}
        # return np.array(vf_ob), np.array(ac_ob), reward, done, {}
















