#pragma once
#include <inttypes.h>
#include <robin_hood.h>
#include <atomic>
#include <thread_safe_struct.hpp>
#include <logger.hpp>

namespace structures{
struct dataset_registry{
    typedef uint64_t registrator_id_t;

    struct registry_entry{
        registrator_id_t registrator_id;
        bool            dataset_used;  // signals that the dataset has been readily used

        bool operator==(const registry_entry& o) const {return registrator_id == o.registrator_id && dataset_used == o.dataset_used;}
    };

    robin_hood::unordered_map<registrator_id_t, registry_entry> registry{};
    bool                                                        all_registrators_done{};  // set to true after all registrators signal that they are done

    void register_registrator(registrator_id_t id){
        registry[id].dataset_used = false;
        registry[id].registrator_id = id;
    }

    void unregister_histogram(registrator_id_t id){
        if(!this)
            return;
        assert(registry.contains(id));
        registry.erase(id);
    }

    void reset_registrators(){
        all_registrators_done = false;
        for(auto& [key, entry]: registry)
            entry.dataset_used = false;
    }

    struct scoped_registrator_t{
        static std::atomic<registrator_id_t> registrator_id_counter;    // defined in glboals.cpp
        const registrator_id_t  registrator_id_none{static_cast<registrator_id_t>(-1)};
        dataset_registry&       registry;
        std::mutex*             mutex{};            // optional mutex which can be used for synchronization
        registrator_id_t        registrator_id{};

        scoped_registrator_t(dataset_registry& registry, std::mutex* mutex = {}):
            registry(registry),
            registrator_id(registrator_id_counter++)
        {
            registry.register_registrator(registrator_id);
            registry.all_registrators_done = false;
        }
        scoped_registrator_t(const scoped_registrator_t&) = delete;     // no copy construction
        scoped_registrator_t(scoped_registrator_t&& o):
            registry(registry),
            mutex(o.mutex),
            registrator_id(o.registrator_id)
        {
            o.registrator_id = registrator_id_none;
        }
        scoped_registrator_t& operator=(const scoped_registrator_t&) = delete;  // no copy assignment
        scoped_registrator_t& operator=(scoped_registrator_t&& o){
            registry = o.registry;
            registrator_id = o.registrator_id;
            mutex = o.mutex;
            o.registrator_id = registrator_id_none;
            return *this;
        }
        ~scoped_registrator_t(){
            if(registrator_id != registrator_id_none)
                registry.unregister_histogram(registrator_id);
        }
        
        std::scoped_lock<std::mutex> lock_scoped(){
            assert(mutex);
            return std::scoped_lock(*mutex);
        }
    };
    scoped_registrator_t scoped_registrator(std::mutex* mutex = {}){
        return scoped_registrator_t(*this, mutex);
    }
    void signal_registrator_used(const scoped_registrator_t& registrator){
        assert(registry.contains(registrator.registrator_id));
        registry[registrator.registrator_id].dataset_used = true;
        for(const auto& [key, entry]: registry){
            if(!entry.dataset_used)
                return;
        }
        all_registrators_done = true;
    }
};
using thread_safe_dataset_reg = thread_safe<dataset_registry>;
}