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

namespace structures{
struct histogram_registry_key{
    std::vector<uint32_t>   attribute_indices{};
    std::vector<int>        bin_sizes{};            // for - any bin size available will be taken and defaults to the abs() of the val

    bool operator==(const histogram_registry_key& o) const {return attribute_indices == o.attribute_indices && bin_sizes == o.bin_sizes;}
};

struct histogram_registry_entry{
    std::string             hist_id{};
    uint32_t                registered_count{};
    
    bool operator==(const histogram_registry_entry& o) const {return hist_id == o.hist_id;}
};
}

template<> struct std::hash<structures::histogram_registry_key>{
    size_t operator()(const structures::histogram_registry_key& k) const {
        return std::hash_combine(util::memory_view<const uint32_t>(k.attribute_indices).data_hash(), util::memory_view<const int>(k.bin_sizes).data_hash());
    }
};

namespace structures{
struct histogram_registry{
    robin_hood::unordered_map<histogram_registry_key, histogram_registry_entry> registry{};
    robin_hood::unordered_map<std::string_view, histogram_registry_key>         name_to_registry_key{}; 
    robin_hood::unordered_map<std::string_view, buffer_info>                    gpu_buffers{};
    bool                                                                        gpu_buffers_edited{};   // is set to true when structures::histogram_counter is editing the histograms, no rendering should be done when this happens
    bool                                                                        gpu_buffers_updated{};  // is set to true by structures::histogram_counter after counting is done
    std::set<std::string_view>                                                  change_request{};

    // the preferred way of registering and unregistering is by using a scoped_registrator_t object which can be retrieved via scoped_registrator(...)
    void register_histogram(util::memory_view<const uint32_t> attribute_indices, util::memory_view<const int> bin_sizes){
        histogram_registry_key key{std::vector<uint32_t>(attribute_indices.begin(), attribute_indices.end()), std::vector<int>(bin_sizes.begin(), bin_sizes.end())};
        if(registry.contains(key))
            return;
        auto& entry = registry[key];
        entry.registered_count++;
        if(entry.hist_id.empty()){
            entry.hist_id = util::histogram_registry::get_id_string(attribute_indices, bin_sizes);
            name_to_registry_key[entry.hist_id] = key;
            change_request.insert(entry.hist_id);
        }
    }

    // the preferred way of registering and unregistering is by using a scoped_registrator_t object which can be retrieved via scoped_registrator(...)
    void unregister_histogram(std::string_view id){
        assert(name_to_registry_key.contains(id) && "Registry does not hold id");
        auto key = name_to_registry_key[id];
        auto& entry = registry[key];
        if(--entry.registered_count == 0){
            util::vk::destroy_buffer(gpu_buffers[id]);
            gpu_buffers.erase(id);
            name_to_registry_key.erase(id);
        }
    }

    void request_change_all(){
        for(const auto& [id, reg]: name_to_registry_key)
            change_request.insert(id);
    }

    struct scoped_registrator_t{
        histogram_registry& registry;
        std::string_view    registry_id;

        scoped_registrator_t(histogram_registry& registry, util::memory_view<const uint32_t> attribute_indices, util::memory_view<const int> bin_sizes):
            registry(registry)
        {
            registry.register_histogram(attribute_indices, bin_sizes);
            histogram_registry_key key{std::vector<uint32_t>(attribute_indices.begin(), attribute_indices.end()), std::vector<int>(bin_sizes.begin(), bin_sizes.end())};
            registry_id = std::string_view(registry.registry[key].hist_id);
        }
        scoped_registrator_t(const scoped_registrator_t& o):
            registry(o.registry), 
            registry_id(o.registry_id)
        {
            registry.register_histogram(registry.name_to_registry_key[registry_id].attribute_indices, registry.name_to_registry_key[registry_id].bin_sizes);
        }
        scoped_registrator_t(scoped_registrator_t&& o):
            registry(o.registry),
            registry_id(o.registry_id)
        {
            o.registry_id = {};
        }
        scoped_registrator_t& operator=(const scoped_registrator_t& o) {
            registry.unregister_histogram(registry_id); 
            registry = o.registry; 
            registry_id = o.registry_id; 
            registry.register_histogram(registry.name_to_registry_key[registry_id].attribute_indices, registry.name_to_registry_key[registry_id].bin_sizes);
            return *this;
        }
        scoped_registrator_t& operator=(scoped_registrator_t&& o){
            registry.unregister_histogram(registry_id);
            registry = o.registry;
            registry_id = o.registry_id;
            o.registry_id = {};
            return *this;
        }
        ~scoped_registrator_t(){
            if(registry_id.size())
                registry.unregister_histogram(registry_id);
        }
    };
    scoped_registrator_t scoped_registrator(util::memory_view<const uint32_t> attribute_indices, util::memory_view<const int> bin_sizes){
        return scoped_registrator_t(*this, attribute_indices, bin_sizes);
    }
};
using thread_safe_hist_reg = thread_safe<histogram_registry>;
}