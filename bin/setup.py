import libPyCpp
import time
import numpy as np

def put_npArray(id, keyInfo):
    shmArray = libPyCpp.kvs_ShmPtr_npArray(keyInfo)
    shmArray[0:int(keyInfo)] = 25
    libPyCpp.kvs_Put(id, keyInfo)
    return

def get_npArray(id, keyInfo):
    s = libPyCpp.kvs_GetNp(id, keyInfo)
    # if s is an int, it means there is an error
    if isinstance(s,int):
        if status == 1:
            print("Ack error")
        if status == 2:
            print("Not my ack")
        if status == 3:
            print("request id doesn't match")
        return s, bytearray()
    print("get success")
    return 0, s

def put_BtArray(id, keyInfo):
    shmArray = libPyCpp.kvs_ShmPtr_btArray(keyInfo)
    for i in range(int(keyInfo)):
        shmArray[i] = 0x03
    status = libPyCpp.kvs_Put(id, keyInfo)
    if status == 0:
        print("put success")
    if status == 1:
        print("Ack error")
    if status == 2:
        print("Not my ack")
    if status == 3:
        print("request id doesn't match")
    libPyCpp.kvs_free(shmArray)
    return status

"""
get function:
    return (err, results)
err: 0 -- succ
     1 -- ack error
     2 -- not my ack
     3 -- request id doesn't match
"""
def get_BtArray(id, keyInfo):
    s = libPyCpp.kvs_GetBt(id, keyInfo)
    if isinstance(s,int):
        if status == 1:
            print("Ack error")
        if status == 2:
            print("Not my ack")
        if status == 3:
            print("request id doesn't match")
        return s, bytearray()
    print("get success")
    return 0, s

"""
put function:
    id: client identity
    keyInfo: key information
    dStruc: whay is the value type -- 1: bytearray 0: numpy array
"""
def put(id, keyInfo, dataStruc):
    if (dataStruc == 0):
        return put_npArray(id, keyInfo)
    if (dataStruc == 1):
        return put_BtArray(id, keyInfo)

"""
get function:
    id: client identity
    keyInfo: key information
    dStruc: whay is the value type -- 1: bytearray 0: numpy array
"""
def get(id, keyInfo, dataStruc):
    if (dataStruc == 0):
        return get_npArray(id, keyInfo)
    if (dataStruc == 1):
        return get_BtArray(id, keyInfo)



# put(1, "256", 0)
# rs = get(1, "256", 0)
status = put(1, "45", 1)
status, rs2 = get(1, "45", 1)
print(rs2)
status = put(2, "45", 1)
status, rs3 = get(2, "45", 1)
libPyCpp.kvs_quit()

"""
require explicitly free bytearray object. 
"""
a = libPyCpp.kvs_free(rs2)
a = libPyCpp.kvs_free(rs3)