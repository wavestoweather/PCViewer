#pragma once
#include <vector>
#include <set>
#include <robin_hood.h>
#include <thread_safe_struct.hpp>
#include <std_util.hpp>
#include <buffer_info.hpp>
#include <memory_view.hpp>
#include <vk_util.hpp>
#include <histogram_registry_util.hpp>
#include <numeric>

namespace structures{
struct histogram_registry_key{
    bool                    is_min_histogram:1;
    bool                    is_max_histogram:1;
    std::vector<uint32_t>   attribute_indices{};
    std::vector<int>        bin_sizes{};            // for - any bin size available will be taken and defaults to the abs() of the val

    bool operator==(const histogram_registry_key& o) const {return attribute_indices == o.attribute_indices && bin_sizes == o.bin_sizes && is_min_histogram == o.is_min_histogram && is_max_histogram == o.is_max_histogram;}
};

typedef uint64_t registrator_id_t;
struct histogram_registry_entry{
    std::string         hist_id{};
    bool                cpu_histogram_needed{};
    robin_hood::unordered_set<registrator_id_t> registered_registrators{};
    robin_hood::unordered_set<registrator_id_t> registrator_signals{};      // contains the signals of all registrators
    
    bool operator==(const histogram_registry_entry& o) const {return hist_id == o.hist_id;}
};
}

template<> struct std::hash<structures::histogram_registry_key>{
    size_t operator()(const structures::histogram_registry_key& k) const {
        return std::hash_combine(std::hash_combine(util::memory_view<const uint32_t>(k.attribute_indices).data_hash(), util::memory_view<const int>(k.bin_sizes).data_hash()), size_t((k.is_max_histogram << 1) | k.is_min_histogram));
    }
};

namespace structures{
struct histogram_registry{
    robin_hood::unordered_map<histogram_registry_key, histogram_registry_entry> registry{};
    robin_hood::unordered_map<std::string_view, histogram_registry_key>         name_to_registry_key{}; 
    robin_hood::unordered_map<std::string_view, buffer_info>                    gpu_buffers{};
    robin_hood::unordered_map<std::string_view, std::vector<uint32_t>>          cpu_histograms{};
    bool                                                                        block_update_done{};    // is set to true after a single gpu block of data has been processed (used to signal next gpu buffer counting)
    bool                                                                        dataset_update_done{};  // is set to true when the whole dataset was counted/when last block update is done (used to signal rendering)
    bool                                                                        registrators_done{true};// is true after all scoped registrators have called signal_registry_done(...) (used to wait with next update round for rendering/processing of all histograms) 
    std::set<std::string_view>                                                  change_request{};

    const histogram_registry_entry* registry_by_name(std::string_view id) const{
        if(!name_to_registry_key.contains(id))
            return {};
        const auto& key = name_to_registry_key.at(id);
        return &registry.at(key);
    }

    // the preferred way of registering and unregistering is by using a scoped_registrator_t object which can be retrieved via scoped_registrator(...)
    std::string_view register_histogram(util::memory_view<const uint32_t> attribute_indices, util::memory_view<const int> bin_sizes, registrator_id_t registrator_id,  bool is_min_hist, bool is_max_hist, bool cpu_hist_needed){
        histogram_registry_key key{is_min_hist, is_max_hist, std::vector<uint32_t>(attribute_indices.size()), std::vector<int>(bin_sizes.size())};
        std::vector<uint32_t> sorted(attribute_indices.size()); std::iota(sorted.begin(), sorted.end(), 0);
        std::sort(sorted.begin(), sorted.end(), [&](uint32_t l, uint32_t r){return attribute_indices[l] < attribute_indices[r];});
        for(int i: util::size_range(sorted)){
            key.attribute_indices[i] = attribute_indices[sorted[i]];
            key.bin_sizes[i] = bin_sizes[sorted[i]];
        }
        auto& entry = registry[key];
        entry.registered_registrators.insert(registrator_id);
        if(entry.hist_id.empty()){
            entry.hist_id = util::histogram_registry::get_id_string(key.attribute_indices, key.bin_sizes, is_min_hist, is_max_hist);
            name_to_registry_key[entry.hist_id] = key;
            entry.cpu_histogram_needed |= cpu_hist_needed;
            change_request.insert(entry.hist_id);
            block_update_done = false;
            dataset_update_done = false;
            registrators_done = false;
        }
        else{
            entry.cpu_histogram_needed = cpu_hist_needed;
        }
        return entry.hist_id;
    }

    // copy registration
    std::string_view register_histogram(std::string_view id, registrator_id_t registrator_id){
        assert(name_to_registry_key.contains(id) && "Registry does not hold id");
        const auto& key = name_to_registry_key[id];
        auto& entry = registry[key];
        entry.registered_registrators.insert(registrator_id);
        return entry.hist_id;
    }

    // the preferred way of registering and unregistering is by using a scoped_registrator_t object which can be retrieved via scoped_registrator(...)
    void unregister_histogram(std::string_view id, registrator_id_t registrator_id){
        assert(name_to_registry_key.contains(id) && "Registry does not hold id");
        const auto key = name_to_registry_key[id];
        auto& entry = registry[key];
        assert(entry.registered_registrators.contains(registrator_id) && "Registry entry does not hold registrator_id");
        entry.registered_registrators.erase(registrator_id);
        if(entry.registrator_signals.contains(registrator_id))
            entry.registrator_signals.erase(registrator_id);
        if(entry.registered_registrators.empty()){
            util::vk::destroy_buffer(gpu_buffers[id]);
            gpu_buffers.erase(id);
            if(cpu_histograms.contains(id))
                cpu_histograms.erase(id);
            name_to_registry_key.erase(id);
            registry.erase(key);
        }
    }

    void request_change_all(){
        for(const auto& [id, reg]: name_to_registry_key)
            change_request.insert(id);
        block_update_done = false;
        dataset_update_done = false;
    }

    bool is_used() const {
        return registry.size();
    }

    bool multi_dim_min_max_used() const{
        for(const auto& [key, val]: registry){
            if(key.bin_sizes.size() > 1 && (key.is_max_histogram || key.is_min_histogram))
                return true;
        }
        return false;
    }

    struct scoped_registrator_t{
        static std::atomic<registrator_id_t> registrator_id_counter; // defined in globals.cpp

        histogram_registry& registry;
        registrator_id_t    registrator_id{};
        std::string_view    registry_id{};
        
        scoped_registrator_t(histogram_registry& registry, util::memory_view<const uint32_t> attribute_indices, util::memory_view<const int> bin_sizes, bool is_min_hist, bool is_max_hist, bool cpu_hist_needed):
            registry(registry),
            registrator_id(registrator_id_counter++),
            registry_id(registry.register_histogram(attribute_indices, bin_sizes, registrator_id, is_min_hist, is_max_hist, cpu_hist_needed))
        {
            // signaling that the registrator was used to run first count
            signal_registry_used();
        }
        scoped_registrator_t(const scoped_registrator_t& o):
            registry(o.registry), 
            registrator_id(registrator_id_counter++),
            registry_id(registry.register_histogram(o.registry_id, registrator_id))
        {
            signal_registry_used();
        }
        scoped_registrator_t(scoped_registrator_t&& o):
            registry(o.registry),
            registrator_id(o.registrator_id),
            registry_id(o.registry_id)
        {
            o.registry_id = {};
        }
        scoped_registrator_t& operator=(const scoped_registrator_t& o) {
            if(registry_id.size())
                registry.unregister_histogram(registry_id, registrator_id); 
            registry = o.registry; 
            registrator_id = registrator_id_counter++;
            registry_id = o.registry_id; 
            registry.register_histogram(registry_id, registrator_id);
            signal_registry_used();
            return *this;
        }
        scoped_registrator_t& operator=(scoped_registrator_t&& o){
            if(registry_id.size())
                registry.unregister_histogram(registry_id, registrator_id);
            registry = o.registry;
            registry_id = o.registry_id;
            registrator_id = o.registrator_id;
            o.registry_id = {};
            return *this;
        }
        ~scoped_registrator_t(){
            if(registry_id.size())
                registry.unregister_histogram(registry_id, registrator_id);
        }

        void signal_registry_used(){
            auto& entry = registry.registry[registry.name_to_registry_key[registry_id]];
            entry.registrator_signals.insert(registrator_id);
            for(const auto& [key, entry]: registry.registry){
                if(entry.registered_registrators != entry.registrator_signals)
                    return;
            }
            registry.registrators_done = true;          // all registrators used the histograms
        }
    };
    scoped_registrator_t scoped_registrator(util::memory_view<const uint32_t> attribute_indices, util::memory_view<const int> bin_sizes, bool is_min_hist, bool is_max_hist, bool cpu_hist_needed){
        return scoped_registrator_t(*this, attribute_indices, bin_sizes, is_min_hist, is_max_hist, cpu_hist_needed);
    }
};
using thread_safe_hist_reg = thread_safe<histogram_registry>;
}