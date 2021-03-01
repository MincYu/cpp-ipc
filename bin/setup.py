import libPyCpp
import numpy as np
def client(i, read, info, dStruc):
    s = libPyCpp.kvs_client(i, read, info, dStruc)
    print(s)
    a = libPyCpp.kvs_free(s)

    return
client(1, 0, "30", 0)
client(1, 1, "30", 0)
client(1, 0, "50", 1)
client(1, 1, "50", 1)
print("client finished")
