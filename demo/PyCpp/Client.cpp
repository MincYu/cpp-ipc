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
#include <numpy/arrayobject.h>


using string = std::string;
using namespace ipc::shm;

namespace ipc {

    capo::random<> rand__{
        static_cast<int>(1),
        static_cast<int>(127)
    };
    std::atomic<bool> is_quit__{ false };
    constexpr char const name__  [] = "ipc-kvs";
    ipc::channel shared_chan { name__, ipc::sender | ipc::receiver };

    PyObject *
    PyArray_FromIntArray(int *rind, Py_ssize_t size) {
        npy_intp dim[1] = {size};
        PyArrayObject *mat = (PyArrayObject*) PyArray_SimpleNewFromData(1, dim, NPY_INT, (char*) rind);
        return PyArray_Return(mat);
    }

    PyObject * WrappGetNp(PyObject* self, PyObject *args) {
        int id;
        const char * info_;
        int finalSize = 0; // dataStruc: 1 represent bytearray; 0 represent numpy array
        
        if(!PyArg_ParseTuple(args, "iz", &id, &info_)){
            return NULL;
        }
        string info(info_);
        auto client_id = 2 + id;
        std::cout << "Launching client " << client_id << " ...\n";
        string req;
        auto req_id = rand__();
        string key_name = "a" + info;
        req.push_back(1);
        req.push_back(client_id);
        req.push_back(1);
        req.push_back(req_id);
        req.push_back((char) key_name.size());
        req += key_name;
        while (!shared_chan.send(req)) {
            // waiting for connection
            shared_chan.wait_for_recv(2);
        }
        auto dd = shared_chan.recv();
        auto str = static_cast<char*>(dd.data());
        
        // response address (1 byte) | request id (1 byte) | is_success (1 byte) | optional value
        if (str == nullptr) {
            std::cout << "Ack error" << std::endl;
            return Py_None;
        }
        if (client_id != (int) str[0]){
            std::cout << "Not my ack" << std::endl;
            return Py_None;
        }  
        if (str[1] != req_id) {
            std::cout << "request id doesn't match" << std::endl;
            return Py_None;
        }
        auto size_len = stoi(string(str + 3));
        auto shm_id = acquire(key_name.c_str(), size_len, open);
        auto shm_ptr = (int *) get_mem(shm_id, nullptr);
        auto val_size = strlen((char *) shm_ptr);
        std::cout << "----Receive Get" << std::endl;
        auto start_stamp = std::chrono::system_clock::now();
        auto mat = PyArray_FromIntArray(shm_ptr, size_len - 1);
        auto trans_stamp = std::chrono::system_clock::now();
        auto trans_time = std::chrono::duration_cast<std::chrono::microseconds>(trans_stamp - start_stamp).count();
        std::cout << "transfer from int * to numpy array using " << trans_time << " ms\n";
        return mat;
    }

    PyObject * WrappGetBt(PyObject* self, PyObject *args) {
        int id;
        const char * info_;
        
        if(!PyArg_ParseTuple(args, "iz", &id, &info_)){
            return NULL;
        }
        string info(info_);
        auto client_id = 2 + id;
        std::cout << "Launching client " << client_id << " ...\n";
        string req;
        auto req_id = rand__();
        string key_name = "a" + info;
        req.push_back(1);
        req.push_back(client_id);
        req.push_back(1);
        req.push_back(req_id);
        req.push_back((char) key_name.size());
        req += key_name;
        while (!shared_chan.send(req)) {
            // waiting for connection
            shared_chan.wait_for_recv(2);
        }
        auto dd = shared_chan.recv();
        auto str = static_cast<char*>(dd.data());
        
        // response address (1 byte) | request id (1 byte) | is_success (1 byte) | optional value
        if (str == nullptr) {
            std::cout << "Ack error" << std::endl;
            return Py_None;
        }
        if (client_id != (int) str[0]){
            std::cout << "Not my ack" << std::endl;
            return Py_None;
        }  
        if (str[1] != req_id) {
            std::cout << "request id doesn't match" << std::endl;
            return Py_None;
        }
        auto size_len = stoi(string(str + 3));
        auto shm_id = acquire(key_name.c_str(), size_len, open);
        auto shm_ptr = (char *) get_mem(shm_id, nullptr);
        auto val_size = strlen((char *) shm_ptr);

        std::cout << "----Receive Get" << std::endl;

        auto start_stamp = std::chrono::system_clock::now();
        auto bArray = PyByteArray_FromString_WithoutCopy(shm_ptr, size_len - 1);
        auto trans_stamp = std::chrono::system_clock::now();
        auto trans_time = std::chrono::duration_cast<std::chrono::microseconds>(trans_stamp - start_stamp).count();
        std::cout << "transfer from char * to python bytearray using " << trans_time << " ms\n";
        
        return bArray;
    }

    PyObject* WrappFree(PyObject* self, PyObject *args)
    {
        PyByteArrayObject * toFree  = (PyByteArrayObject *) PyTuple_GET_ITEM(args, 0);
        toFree->ob_bytes = NULL;
        return Py_None;
    }

    PyObject* WrappShmNp(PyObject* self, PyObject *args)
    {
        const char * info_;
        
        if(!PyArg_ParseTuple(args, "z", &info_)){
            return NULL;
        }
        string info(info_);
        string key_name = "a" + info;
        std::size_t shm_size = stoi(info) + 1;
        auto shm_id = acquire(key_name.c_str(), shm_size);
        int * shm_ptr = (int *) get_mem(shm_id, nullptr);
        auto mat = PyArray_FromIntArray(shm_ptr, shm_size - 1);
        return mat;
    }

    PyObject* WrappShmBt(PyObject* self, PyObject *args)
    {
        const char * info_;
        
        if(!PyArg_ParseTuple(args, "z", &info_)){
            return NULL;
        }
        string info(info_);
        string key_name = "a" + info;
        std::size_t shm_size = stoi(info) + 1;
        auto shm_id = acquire(key_name.c_str(), shm_size);
        auto shm_ptr = (char *) get_mem(shm_id, nullptr);
        auto shm_id_ = acquire(key_name.c_str(), shm_size, open);
        auto shm_ptr_ = (char *) get_mem(shm_id_, nullptr);
        auto bArray = PyByteArray_FromString_WithoutCopy(shm_ptr_, shm_size - 1);
        
        return bArray;
    }

    PyObject* WrappPut(PyObject* self, PyObject *args)
    {
        int id;
        const char *info_;
        if(!PyArg_ParseTuple(args, "iz", &id, &info_)){
            return NULL;
        }
        int client_id = 2 + id;
        std::cout << "Launching client " << client_id << " ...\n";
        string info(info_);
        string req, key_name = "a" + info;
        auto req_id = rand__();
        req.push_back(1);
        req.push_back(client_id);
        req.push_back(2);
        req.push_back(req_id);
        req.push_back((char) key_name.size());
        req += key_name;
        std::size_t shm_size = stoi(info) + 1;
        req += std::to_string(shm_size);
        int data_len = stoi(info);
        while (!shared_chan.send(req)) {
            // waiting for connection
            shared_chan.wait_for_recv(2);
        }
        auto dd = shared_chan.recv();
        auto str = static_cast<char*>(dd.data());

        // response address (1 byte) | request id (1 byte) | is_success (1 byte) | optional value
        if (str == nullptr) {
            std::cout << "Ack error" << std::endl;
            return Py_None;
        }
        if (client_id != (int) str[0]){
            std::cout << "Not my ack" << std::endl;
            return Py_None;
        }  
        if (str[1] != req_id) {
            std::cout << "request id doesn't match" << std::endl;
            return Py_None;
        }
        std::cout << "----Receive Put" << std::endl;
        return Py_None;
            
    }


    static PyMethodDef client_methods[] = {
    {"kvs_free", WrappFree, METH_VARARGS},
    {"kvs_ShmPtr_npArray", WrappShmNp, METH_VARARGS},
    {"kvs_Put", WrappPut, METH_VARARGS},
    {"kvs_GetNp", WrappGetNp, METH_VARARGS},
    {"kvs_ShmPtr_btArray", WrappShmBt, METH_VARARGS},
    {"kvs_GetBt", WrappGetBt, METH_VARARGS},
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
        import_array(); // init numpy module when loading lib
        return PyModule_Create(&client_module);
    }

}



