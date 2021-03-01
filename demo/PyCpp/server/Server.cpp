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
#include <unordered_map>


using string = std::string;

template <class K, class V>
using map = std::unordered_map<K, V>;
using namespace ipc::shm;



namespace {
    constexpr char const name__  [] = "ipc-kvs";
    ipc::channel shared_chan { name__, ipc::sender | ipc::receiver };
    std::atomic<bool> is_quit__{ false };
    void kvs_server() {
        map<string, char *> key_val_map;
        map<string, uint32_t> key_len_map;
        std::cout << "Running kvs server...\n";
        while (1){
            auto dd = shared_chan.recv();
            auto str = static_cast<char*>(dd.data());

            auto recv_stamp = std::chrono::system_clock::now();


            if (str == nullptr) {
                std::cout << "Receive null str\n";
                continue;
            }
            
            // request addres (1 byte) | resp address (1 byte) | get/put (1 byte) | request id (1 byte) | metadata len (1 byte)| metadata | optional value
            if (str[0] != 1) {
                std::cout << "Not for server\n";
                continue;
            }

            // std::printf("2 recv: %s\n", str);
            auto resp_address = str[1];
            bool is_read = (str[2] == 1);
            auto req_id = str[3];
            int meta_data_len = (int)str[4];

            string key_name(str + 5, meta_data_len);

            string resp;
            resp.push_back(resp_address);
            resp.push_back(req_id);

            // response address (1 byte) | request id (1 byte) | is_success (1 byte) | optional value
            if (is_read){
                // get request
                std::cout << "Getting " << key_name << " ...\n";
                if (key_len_map.find(key_name) != key_len_map.end()) {
                    auto size_len = key_len_map[key_name];

                    resp.push_back(1);
                    // resp.push_back((char) size_len);
                    // resp.push_back((char) size_len >> 8);
                    // resp.push_back((char) size_len >> 16);
                    // resp.push_back((char) size_len >> 24);
                    resp += std::to_string(size_len);
                }
                else {
                    std::cout << key_name << " not exists\n";
                    resp.push_back(2);
                }
            }
            else{
                // put request
                std::cout << "Putting " << key_name << " ...\n";
                
                // auto size_len = (uint32_t) str[5 + meta_data_len]       |
                //                 (uint32_t) str[7 + meta_data_len] << 8  |
                //                 (uint32_t) str[8 + meta_data_len] << 16 |
                //                 (uint32_t) str[9 + meta_data_len] << 24;
                
                auto size_len = stoi(string(str + 5 + meta_data_len));

                // handle shm_hd(key_name.c_str(), size_len);
                // auto shm_ptr = (char *) shm_hd.get();
                auto shm_id = acquire(key_name.c_str(), size_len, open);

                // auto shm_ptr = (char *) get_mem(shm_id, nullptr);
                // for (int i = 0; i < strlen(shm_ptr); i++){
                //     std::cout << shm_ptr[i] << " ";
                // }
                // std::cout << "\n";
                

                if (shm_id == nullptr){
                    std::cout << "Shm null ptr for " << key_name << "\n";
                    resp.push_back(2);
                }
                else {
                    auto shm_ptr = (char *) get_mem(shm_id, nullptr);
                    // auto val_size = strlen(shm_ptr);
                    std::cout << "shm_size " << size_len << " " << shm_ptr[20] << "\n";

                    key_val_map[key_name] = shm_ptr;
                    key_len_map[key_name] = size_len;
                    resp.push_back(1);
                }
            }

            auto ready_stamp = std::chrono::system_clock::now();
            auto handling_time = std::chrono::duration_cast<std::chrono::microseconds>(ready_stamp - recv_stamp).count();

            auto req_type = is_read ? "Get" : "Put";
            std::cout << "Handled " << req_type << " " << key_name << ", handling_time: " << handling_time << "\n";

            // try sending ack
            while (!shared_chan.send(resp)) {
                // waiting for connection
                shared_chan.wait_for_recv(2);
            }
        }
        std::cout << __func__ << ": quit...\n";
    }
}
int main(int argc, char ** argv) {
    auto exit = [](int) {
        is_quit__.store(true, std::memory_order_release);
        shared_chan.disconnect();
    };
    ::signal(SIGINT  , exit);
    ::signal(SIGABRT , exit);
    ::signal(SIGSEGV , exit);
    ::signal(SIGTERM , exit);
    ::signal(SIGHUP  , exit);

    kvs_server();
    return 0;
}




