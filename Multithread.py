import queue


class Multithreading:
    def __init__(self):
        self.qu = queue.Queue(maxsize=2)

    def put(self, item):
        self.qu.put(item)

    def get(self):
        return self.qu.get()

    def empty(self):
        return self.qu.empty()

