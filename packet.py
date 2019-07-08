class Packet:
    def __init__(self, generateTime1, scheduler1=None):
        self.generateTime = generateTime1
        self.scheduler = scheduler1
        self.enqueueTime = 0

    def __cmp__(self, other):
        # call global(builtin) function cmp for int
        return (self.enqueueTime > other.enqueueTime) - (self.enqueueTime < other.enqueueTime)

    def __lt__(self, other):  # operator <
        return self.enqueueTime < other.enqueueTime

    def __ge__(self, other):  # oprator >=
        return self.enqueueTime >= other.enqueueTime

    def __le__(self, other):  # oprator <=
        return self.enqueueTime <= other.enqueueTime