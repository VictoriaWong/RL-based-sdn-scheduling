from collections import deque


class Stats:
    def __init__(self, setting):
        self.set = setting
        self.sch_no = self.set.schNum
        self.ctl_no = self.set.ctlNum
        self.resp_index = deque([])
        self.packet_response_map = []  # sch -> [#] * len(ctl); # -> {^} * history_len; ^ -> [resp_time within respTime_update_interval]
        # for i in range(self.sch_no):
        #     self.packet_response_map.append([])
        #     for j in range(self.ctl_no):
        #         self.packet_response_map[i].append({})
        self._init_packet_response_map()

    def _init_packet_response_map(self):
        for i in range(self.sch_no):
            self.packet_response_map.append([])
            for j in range(self.ctl_no):
                self.packet_response_map[i].append({})
                for his in range(self.set.history_len):
                    self.packet_response_map[i][j][his-self.set.history_len] = [0.]
        for his in range(self.set.history_len):
            self.resp_index.append(his-self.set.history_len)

    def check_resp_index(self, ind_new):
        if ind_new not in self.resp_index:
            ind_old = self.resp_index.popleft()
            self.resp_index.append(ind_new)
            for sch in range(self.sch_no):
                for ctl in range(self.ctl_no):
                    del self.packet_response_map[sch][ctl][ind_old]
                    self.packet_response_map[sch][ctl][ind_new] = []

    def add_response_time(self, timestamp, sch_id, ctl_id, response_time):
        ind_new = int(timestamp/self.set.respTime_update_interval)
        self.check_resp_index(ind_new)
        self.packet_response_map[sch_id][ctl_id][ind_new].append(response_time)

    def update_response_time_history(self, sch_id):
        resp_ac_ob = []  # [history_len] * ctl_no
        for i in range(self.ctl_no):
            resp_ac_ob.append([])
            for key in self.resp_index:
                num = len(self.packet_response_map[sch_id][i][key])
                tot = sum(self.packet_response_map[sch_id][i][key])
                if num == 0:
                    resp_ac_ob[i].append(0)
                else:
                    resp_ac_ob[i].append(tot/num)
        resp_vf_ob = []  # [history_len]
        for key in self.resp_index:
            num = 0
            tot = 0
            for sch in range(self.sch_no):
                for ctl in range(self.ctl_no):
                    num += len(self.packet_response_map[sch][ctl][key])
                    tot += sum(self.packet_response_map[sch][ctl][key])
            if num == 0:
                resp_vf_ob.append(0)
            else:
                resp_vf_ob.append(tot/num)
        return resp_vf_ob, resp_ac_ob

    def update_utilization_history(self):
        util_ac_ob = []  # [history_len] * ctl_no
        for i in range(self.ctl_no):
            util_ac_ob.append([])
            num = 0
            for key in self.resp_index:
                for sch_id in range(self.sch_no):
                    num += len(self.packet_response_map[sch_id][i][key])
                ratio = num / (self.set.ctlRate[i]*self.set.util_update_interval)
                if ratio > 1:
                    util_ac_ob[i].append(1.0)
                else:
                    util_ac_ob[i].append(ratio)
        util_vf_ob = []  # [history_len]
        for key in self.resp_index:
            num = 0
            for sch in range(self.sch_no):
                for ctl in range(self.ctl_no):
                    num += len(self.packet_response_map[sch][ctl][key])
            ratio = num / (sum(self.set.ctlRate)*self.set.util_update_interval)
            if ratio > 1:
                util_vf_ob.append(1.0)
            else:
                util_vf_ob.append(ratio)
        return util_vf_ob, util_ac_ob


