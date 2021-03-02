import libPyCpp
import numpy as np

def put_npArray(id, keyInfo):
    shmArray = libPyCpp.kvs_ShmPtr_npArray(keyInfo)
    shmArray[0:int(keyInfo)] = 25
    libPyCpp.kvs_Put(id, keyInfo)
    return

def get_npArray(id, keyInfo):
    s = libPyCpp.kvs_GetNp(id, keyInfo)
    return s

def put_BtArray(id, keyInfo):
    shmArray = libPyCpp.kvs_ShmPtr_btArray(keyInfo)
    for i in range(int(keyInfo)):
        shmArray[i] = 0x03
    libPyCpp.kvs_Put(id, keyInfo)
    a = libPyCpp.kvs_free(shmArray)
    return

def get_BtArray(id, keyInfo):
    s = libPyCpp.kvs_GetBt(id, keyInfo)
    return s

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



put(1, "256", 0)
rs = get(1, "256", 0)
put(1, "1024", 1)
rs2 = get(1, "1024", 1)


"""
require explicitly free bytearray object. 
"""
a = libPyCpp.kvs_free(rs2)


