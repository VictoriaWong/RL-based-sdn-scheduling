import logging
import sys
from queue import PriorityQueue

class SimQueue:
    def __init__(self):
        self.queue = PriorityQueue()

    #pkt has a type of packet
    def enqueue(self, pkt, t):
        pkt.enqueueTime = t
        self.queue.put(pkt)

        # sort the queue to make sure the first pkt always enqueues first
        #self.queue.sort(key=operator.attrgetter('enqueueTime'))

    def dequeue(self):
        if self.queue.qsize() > 0:
            return self.queue.get(0)
        else:
            logging.error("queue is empty")
            sys.exit(1)

    def getFirstPktTime(self):
        if self.queue.qsize() > 0:
            #Todo This is not thread safe
            firstPkt = self.queue.queue[0]
            return firstPkt.enqueueTime
        else:
            #logging.error("queue is empty when getting first pkt time")
            return None

