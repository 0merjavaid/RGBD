from queue import Queue
from threading import Thread
import copy

# Object that signals shutdown
_sentinel = object()

# increment function


def increment(i):
    i += 1
    return i

# A thread that produces data


def producer(out_q):
    i = 0
    while True:
        # Produce some data
        i = increment(i)
        print(i)
        out_q.put(i)
        if i > 1000000:
            out_q.put(_sentinel)
            break

# A thread that consumes data


def consumer(in_q):
    for data in iter(in_q.get, _sentinel):
        # Process the data
        pass


# Create the shared queue and launch both threads
q = Queue()
t1 = Thread(target=consumer, args=(q,))
t2 = Thread(target=producer, args=(q,))
t1.start()
t2.start()
