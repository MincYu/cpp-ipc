#include <signal.h>
#include <iostream>
#include <string>
#include <cstring>
#include <thread>
#include <chrono>
#include <cstddef>
#include "libipc/ipc.h"
#include <atomic>
#include "libipc/shm.h"
#include "capo/random.hpp"
#include "util.h"

using string = std::string;


using namespace ipc::shm;

namespace ipc {

constexpr char const name__  [] = "ipc-kvs";

capo::random<> rand__{
    static_cast<int>(1),
    static_cast<int>(127)
};

ipc::channel shared_chan { name__, ipc::sender | ipc::receiver };

std::atomic<bool> is_quit__{ false };

char * kvs_client(char id, bool is_read, string info) {

    auto exit = [](int) {
        is_quit__.store(true, std::memory_order_release);
        shared_chan.disconnect();
    };
    ::signal(SIGINT  , exit);
    ::signal(SIGABRT , exit);
    ::signal(SIGSEGV , exit);
    ::signal(SIGTERM , exit);
    ::signal(SIGHUP  , exit);
    
    auto client_id = 2 + id;
    std::cout << "Launching client " << client_id << " ...\n";

    auto start_stamp = std::chrono::system_clock::now();
    string req;
    auto req_id = rand__();
    string key_name = "a" + info;
    
    // request addres (1 byte) | resp address (1 byte) | get/put (1 byte) | request id (1 byte) | metadata len (1 byte)| metadata | optional value
    req.push_back(1);
    req.push_back(client_id);
    if (is_read){
        req.push_back(1);
        req.push_back(req_id);
        req.push_back((char) key_name.size());
        req += key_name;
    }
    else {
        req.push_back(2);
        req.push_back(req_id);
        req.push_back((char) key_name.size());
        req += key_name;
        auto shm_size = stoi(info) + 1;
        req += std::to_string(shm_size);
        int data_len = stoi(info);
        auto shm_id = acquire(key_name.c_str(), shm_size);
        auto shm_ptr_ = (char *) get_mem(shm_id, nullptr);
        memset(shm_ptr_, '1', data_len);
        shm_ptr_[data_len] = '\0';
    }

    auto ready_stamp = std::chrono::system_clock::now();

    while (!shared_chan.send(req)) {
        // waiting for connection
        shared_chan.wait_for_recv(2);
    }

    // recv ack
    auto dd = shared_chan.recv();
    auto str = static_cast<char*>(dd.data());

    // response address (1 byte) | request id (1 byte) | is_success (1 byte) | optional value
    if (str == nullptr) {
        char *err = "Ack error";
        return err;
    }
    if (client_id != (int) str[0]){
        char * err = "Not my ack";
        return err;
    }  
    if (str[1] != req_id) {
        char * err = "request id doesn't match";
        return err;
    }
    auto ack_stamp = std::chrono::system_clock::now();

        
    if (is_read){
        auto size_len = stoi(string(str + 3));
        auto shm_id = acquire(key_name.c_str(), size_len);
        auto shm_ptr = (char *) get_mem(shm_id, nullptr);

        auto ptr_stamp = std::chrono::system_clock::now();

        auto val_size = strlen(shm_ptr);
        auto val_stamp = std::chrono::system_clock::now();

        auto ready_time = std::chrono::duration_cast<std::chrono::microseconds>(ready_stamp - start_stamp).count();
        auto ack_time = std::chrono::duration_cast<std::chrono::microseconds>(ack_stamp - ready_stamp).count();
        auto ptr_time = std::chrono::duration_cast<std::chrono::microseconds>(ptr_stamp - ack_stamp).count();
        auto val_time = std::chrono::duration_cast<std::chrono::microseconds>(val_stamp - ptr_stamp).count();

        std::cout << "Receive Get " << key_name << ", val_size: " << val_size
                                                << ", shm_size: " << size_len
                                                << ", ready_time: " << ready_time
                                                << ", ack_time: " << ack_time
                                                << ", ptr_time: " << ptr_time
                                                << ", val_time: " << val_time
                                                <<"\n";

        return shm_ptr;
    } else {
        auto ready_time = std::chrono::duration_cast<std::chrono::microseconds>(ready_stamp - start_stamp).count();
        auto ack_time = std::chrono::duration_cast<std::chrono::microseconds>(ack_stamp - ready_stamp).count();

        std::cout << "Receive Put " << key_name << ", ready_time "<< ready_time 
                                                << ", ack_time: " << ack_time 
                                                <<"\n";
        char * put_Msg = "Receive Put";
        return put_Msg;
    }
}

PyObject* WrappClient(PyObject* self, PyObject *args)
{
    int id;
    int is_read;
    const char * info;
    if(!PyArg_ParseTuple(args, "iiz", &id, &is_read, &info)){
        return NULL;
    }
    char * resp = kvs_client(id, is_read, info);
    return PyByteArray_FromString_WithoutCopy(resp, strlen(resp));

}

PyObject* WrappFree(PyObject* self, PyObject *args)
{
    PyByteArrayObject * toFree  = (PyByteArrayObject *) PyTuple_GET_ITEM(args, 0);
    toFree->ob_bytes = NULL;

    return Py_None;

}

static PyMethodDef client_methods[] = {
 {"kvs_client", WrappClient, METH_VARARGS, "something"},
 {"kvs_free", WrappFree, METH_VARARGS},
 {NULL, NULL}
};

static struct PyModuleDef client_module = {
        PyModuleDef_HEAD_INIT,
        "test",
        NULL,
        -1,
        client_methods
};

PyMODINIT_FUNC PyInit_libPyCpp()
{
     return PyModule_Create(&client_module);
}

}


