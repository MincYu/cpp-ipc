import libPyCpp
def client(i, read, info):
    s = libPyCpp.kvs_client(i, read, info)
    print(s)
    a = libPyCpp.kvs_free(s)
    return
client(1, 0, "30")
client(1, 1, "30")
print("client finished")
