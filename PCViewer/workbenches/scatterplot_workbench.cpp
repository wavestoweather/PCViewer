#include "scatterplot_workbench.hpp"
#include <imgui_util.hpp>
#include <vma_initializers.hpp>
#include <imgui_internal.h>
#include <scatterplot_renderer.hpp>
#include <mutex>

namespace workbenches
{
void scatterplot_workbench::_update_registered_histograms(){
    std::array<int, 2> bucket_sizes{int(settings.read().plot_width), int(settings.read().plot_width)};
    auto active_indices = get_active_ordered_indices();
    for(const auto& dl: drawlist_infos.read()){
        if(dl.templatelist_read().data_size < settings.read().large_vis_threshold){
            _registered_histograms.erase(dl.drawlist_id);
            continue;
        }
        
        // setting up the bin sizes
        std::vector<bool> registrator_needed(_registered_histograms[dl.drawlist_id].size(), false);
        std::array<uint32_t, 2> indices;

        // creating the new registratros and flag the used registrators as true -------------------------
        switch(settings.read().plot_type){
        case plot_type_t::list:

            break;
        case plot_type_t::matrix:
            for(int i: util::i_range(active_indices.size() - 1)){
                for(int j: util::i_range(i + 1, active_indices.size())){
                    indices[0] = active_indices[i];
                    indices[1] = active_indices[j];
                    auto registrator_id = util::histogram_registry::get_id_string(indices, bucket_sizes, false, false);
                    size_t registrator_index = util::memory_view(_registered_histograms[dl.drawlist_id]).index_of([&registrator_id](const registered_histogram& h){return registrator_id == h.registry_id;});
                    if(registrator_index != util::memory_view<>::n_pos)
                        registrator_needed[registrator_index] = true;
                    else{
                        // adding new histogram
                        auto& drawlist = dl.drawlist_write();
                        _registered_histograms[dl.drawlist_id].emplace_back(drawlist.histogram_registry.access()->scoped_registrator(indices, bucket_sizes, false, false, false));
                        registrator_needed.push_back(true);
                    }
                }
            }
            break;
        default:
            throw std::runtime_error{"scatterplot_workbench() Unimplemented plot type"};
        }

        // removing unused registrators -----------------------------------------------------------------
        auto registry_lock = dl.drawlist_read().histogram_registry.const_access();
        for(int i: util::rev_size_range(_registered_histograms[dl.drawlist_id])){
            if(!registrator_needed[i])
                _registered_histograms[dl.drawlist_id].erase(_registered_histograms[dl.drawlist_id].begin() + i);
        }
        // printing out the registrators
        if(logger.logging_level >= logging::level::l_5){
            logger << logging::info_prefix << " scatterplot_workbenche (" << active_indices.size() << " attributes, " << registry_lock->registry.size() << " registrators, " << registry_lock->name_to_registry_key.size() << " name to registry entries), registered histograms: ";
            for(const auto& [key, val]: registry_lock->registry)
                logger << val.hist_id << " ";
            logger << logging::endl;
        }
    }
    // setting update singal flags
    for(const auto& dl: drawlist_infos.read())
        dl.drawlist_write().histogram_registry.access()->request_change_all();
}

void scatterplot_workbench::_update_plot_images(){
    constexpr VkImageUsageFlags image_usage{VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT};
    auto active_indices = get_active_ordered_indices();
    // waiting for the device to finish all command buffers using the image before statring to destroy/create plot images
    {
        for(auto& m: globals::vk_context.mutex_storage)
            m->lock();
        auto res = vkDeviceWaitIdle(globals::vk_context.device); util::check_vk_result(res);
        for(auto& m: globals::vk_context.mutex_storage)
            m->unlock();
    }
    
    robin_hood::unordered_set<attribute_pair> used_attribute_pairs;
    std::vector<VkImageMemoryBarrier> image_barriers;
    // creating new pairs / recreating images if size changed
    auto destroy_image=[&](const attribute_pair& p) {
        auto& plot_data = plot_datas[p];
        util::vk::destroy_image(plot_data.image);
        util::vk::destroy_image_view(plot_data.image_view);
        util::imgui::free_image_descriptor_set(plot_data.image_descriptor);
    };
    auto register_image = [&](const attribute_pair& p) {
        // destruction of old image if necessary
        if(plot_datas.contains(p) && (plot_datas[p].image_width != settings.read().plot_width || plot_datas[p].image_format != settings.read().plot_format))
            destroy_image(p);
        // creating new image if needed
        if(!plot_datas[p].image){
            auto& plot_data = plot_datas[p];
            auto image_info = util::vk::initializers::imageCreateInfo(settings.read().plot_format, {settings.read().plot_width, settings.read().plot_width, 1}, image_usage);
            auto mem_alloc = util::vma::initializers::allocationCreateInfo();
            std::tie(plot_data.image, plot_data.image_view) = util::vk::create_image_with_view(image_info, mem_alloc);
            plot_data.image_descriptor = util::imgui::create_image_descriptor_set(plot_data.image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            plot_data.image_format = settings.read().plot_format;

            // updating the image layout
            image_barriers.emplace_back(util::vk::initializers::imageMemoryBarrier(plot_data.image.image, VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, {}, {}, {}, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
        }
        used_attribute_pairs.insert(p);
    };

    for(const auto& p: plot_list.read())
        register_image(p);
    util::vk::convert_image_layouts_execute(image_barriers);

    // removing all unused images
    robin_hood::unordered_set<attribute_pair> unused_attribute_pairs;
    for(const auto& [pair, image_data]: plot_datas) if(!used_attribute_pairs.contains(pair)) unused_attribute_pairs.insert(pair);
    for(const auto& pair: unused_attribute_pairs){
        destroy_image(pair);
        plot_datas.erase(pair);
    }
}

void scatterplot_workbench::_update_plot_list(){
    auto active_indices = get_active_ordered_indices();
    plot_list().clear();
    if(settings.read().plot_type == plot_type_t::matrix){
        for(uint32_t i: util::i_range(1, active_indices.size())){
            for(uint32_t j: util::i_range(i))
                plot_list().emplace_back(attribute_pair{i, j});
        }
    }
}

void scatterplot_workbench::_render_plot(){
    // check for still active histogram update
    for(const auto& dl: drawlist_infos.read()){
        if(_registered_histograms.contains(dl.drawlist_id) && _registered_histograms[dl.drawlist_id].size()){
            auto access = dl.drawlist_read().histogram_registry.const_access();
            if(!access->dataset_update_done)    
                return;
        }
    }

    _update_plot_images();  // all plot images are recreated before rendering is issued

    if(logger.logging_level >= logging::level::l_5)
        logger << logging::info_prefix << " scatterplot_workbench::_render_plot()" << logging::endl;
    pipelines::scatterplot_renderer::render_info render_info{*this};
    pipelines::scatterplot_renderer::instance().render(render_info);

    for(auto& dl_info: drawlist_infos())
        if(!dl_info.linked_with_drawlist)
            dl_info.clear_changes();
    drawlist_infos.changed = false;
    attribute_order_infos.changed = false;
    settings.changed = false;
    attributes.changed = false;
    for(auto& [dl, registrators]: _registered_histograms){
        auto registry_lock = globals::drawlists.read().at(dl).read().histogram_registry.const_access();
        for(auto& registrator: registrators)
            registrator.signal_registry_used();
    }
}

scatterplot_workbench::scatterplot_workbench(std::string_view id):
    workbench(id)
{
    
}

void scatterplot_workbench::notify_drawlist_dataset_update() 
{
    
}
    
void scatterplot_workbench::show() 
{
    if(!active) 
        return;

    // checking for setting updates and updating the rendering if necessary
    bool local_change{false};
    bool request_render{false};
    local_change |= drawlist_infos.changed;
    request_render |= local_change;
    request_render |= attributes.changed;
    request_render |= attribute_order_infos.changed;
    request_render |= settings.changed;
    if(globals::drawlists.changed){
        for(const auto& dl: drawlist_infos.read())
            request_render |= globals::drawlists.read().at(dl.drawlist_id).changed;
    }

    if(request_render)
        _render_plot();

    ImGui::Begin(id.data(), &active, ImGuiWindowFlags_HorizontalScrollbar);

    const auto active_indices = get_active_ordered_indices();
    // plot views ------------------------------------------------------
    switch(settings.read().plot_type){
    case plot_type_t::matrix:
        // matrix should be displayed as a left lower triangular matrix
        for(uint32_t i: util::i_range(1, active_indices.size())){
            ImVec2 text_pos = ImGui::GetCursorScreenPos(); text_pos.y += settings.read().plot_width / 2;
            util::imgui::AddTextVertical(attributes.read()[active_indices[i]].display_name.c_str(), text_pos, .5f);
            ImGui::SetCursorScreenPos({ImGui::GetCursorScreenPos().x + ImGui::GetTextLineHeightWithSpacing(), ImGui::GetCursorScreenPos().y});
            for(uint32_t j: util::i_range(i)){
                if(!plot_datas.contains({i, j}))
                    continue;
                if(j != 0)  
                    ImGui::SameLine();
                auto c_pos = ImGui::GetCursorScreenPos();
                ImGui::GetWindowDrawList()->AddRectFilled(c_pos, {c_pos.x + settings.read().plot_width, c_pos.y + settings.read().plot_width}, ImColor(settings.read().plot_background_color));
                ImGui::Image(plot_datas[{i, j}].image_descriptor, {float(settings.read().plot_width), float(settings.read().plot_width)});
            }
        }
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetTextLineHeight());
        for(int i: util::i_range(int(active_indices.size()) - 1)){
            if(i != 0)
                ImGui::SameLine(i * settings.read().plot_width + ImGui::GetTextLineHeight());
            ImGui::Text("%s", attributes.read()[active_indices[i]].display_name.c_str());
        }
        break;
    case plot_type_t::list:
        for(const auto& [p, first]: util::first_iter(plot_list.read())){
            if(!plot_datas.contains(p))
                continue;
            if(!first)
                ImGui::SameLine();
            ImGui::Image(plot_datas[p].image_descriptor, {float(settings.read().plot_width), float(settings.read().plot_width)});
        }
        break;
    }

    // settings ---------------------------------------------------------
    ImGui::BeginTable("scatterplot_setting", 2, ImGuiTableFlags_Resizable);
    // column setup
    ImGui::TableSetupScrollFreeze(0, 1);    // make top row always visible
    ImGui::TableSetupColumn("Settings");
    ImGui::TableSetupColumn("Drawlists");
    ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
    ImGui::TableNextColumn();
    ImGui::TableHeader("Settings");
    ImGui::TableNextColumn();
    ImGui::TableHeader("Drawlists");

    // settings
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    if(ImGui::BeginCombo("plot type", plot_type_names[settings.read().plot_type].data())){
        for(auto t: structures::enum_iteration<plot_type_t>()){
            if(ImGui::MenuItem(plot_type_names[t].data())){
                settings().plot_type = t;
                _update_plot_list();
            }
        }
        ImGui::EndCombo();
    }
    if(ImGui::TreeNodeEx("Attribute Settings", ImGuiTreeNodeFlags_Framed)){
        switch(settings.read().plot_type){
        case plot_type_t::matrix:
            ImGui::PushID("s_wb_att_set");
            for(auto& attribute: attribute_order_infos.ref_no_track()){
                if(ImGui::Checkbox(attributes.read()[attribute.attribut_index].id.data(), &attribute.active))
                    attribute_order_infos();
            }
            ImGui::PopID();
            break;
        case plot_type_t::list:
            for(uint32_t i: util::i_range(1, attributes.read().size())){
                for(uint32_t j: util::i_range(0, i)){
                    //bool active = util::memory_view(plot_list.read()).contains([&](const attribute_pair& p) {return p.a == i && p.b == j;});
                    if(!util::memory_view<const attribute_pair>(plot_list.read()).contains(attribute_pair{i, j}) && ImGui::MenuItem((attributes.read()[i].display_name + "|" + attributes.read()[j].display_name).c_str()))
                        plot_list().emplace_back(attribute_pair{i, j});
                }
            }
            break;
        }
        ImGui::TreePop();
    }
    if(ImGui::TreeNodeEx("General Settings", ImGuiTreeNodeFlags_Framed)){
        ImGui::InputDouble("plot padding", &settings.read().plot_padding);
        ImGui::ColorEdit4("##cols", &settings.read().plot_background_color.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar);
        ImGui::TreePop();
    }

    // drawlists
    std::string_view delete_drawlist{};
    ImGui::TableNextColumn();
        ImGui::BeginTable("drawlists", 8, ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg);
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Up");
        ImGui::TableSetupColumn("Down");
        ImGui::TableSetupColumn("Delete");
        ImGui::TableSetupColumn("Active");
        ImGui::TableSetupColumn("Color");
        ImGui::TableSetupColumn("Splat form");
        ImGui::TableSetupColumn("Radius");

        // header labels
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        ImGui::TableNextColumn();
        ImGui::TableHeader("Name");
        ImGui::TableNextColumn();
        ImGui::TableHeader("Up");
        ImGui::TableNextColumn();
        ImGui::TableHeader("Down");
        ImGui::TableNextColumn();
        ImGui::TableHeader("Delete");
        ImGui::TableNextColumn();
        ImGui::TableHeader("Active");
        ImGui::TableNextColumn();
        ImGui::TableHeader("Color");
        ImGui::TableNextColumn();
        ImGui::TableHeader("Splat form");
        ImGui::TableNextColumn();
        ImGui::TableHeader("Radius");

        int up_index{-1}, down_index{-1};
        ImGui::PushID(id.data());  // used to distinguish all ui elements in different workbenches
        for(int dl_index: util::rev_size_range(drawlist_infos.read())){
            auto& dl = drawlist_infos.ref_no_track()[dl_index];
            ImGui::PushID(dl.drawlist_id.data());
            
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            DRAWLIST_SELECTABLE(dl.drawlist_id);
            ImGui::TableNextColumn();
            ImGui::BeginDisabled(dl_index == drawlist_infos.read().size() - 1);
            if(ImGui::ArrowButton("##us", ImGuiDir_Up))
                up_index = dl_index;
            ImGui::EndDisabled();
            ImGui::TableNextColumn();
            ImGui::BeginDisabled(dl_index == 0);
            if(ImGui::ArrowButton("##ds", ImGuiDir_Down))
                down_index = dl_index;
            ImGui::EndDisabled();
            ImGui::TableNextColumn();
            if(ImGui::Button("X##xs"))
                delete_drawlist = dl.drawlist_id;
            ImGui::TableNextColumn();
            if(ImGui::Checkbox("##acts", &dl.appearance->ref_no_track().show))
                dl.appearance->write();
            ImGui::TableNextColumn();
            if(ImGui::ColorEdit4("##cols", &dl.appearance->ref_no_track().color.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar))
                dl.appearance->write();
            ImGui::TableNextColumn();
            if(ImGui::BeginCombo("##form", splat_form_names[dl.scatter_appearance.read().splat].data())){
                for(auto e: structures::enum_iteration<splat_form_t>()){
                    if(ImGui::MenuItem(splat_form_names[e].data()))
                        drawlist_infos()[dl_index].scatter_appearance().splat = e;
                }
                ImGui::EndCombo();
            }
            ImGui::TableNextColumn();
            if(ImGui::DragFloat("##rad", &dl.scatter_appearance.ref_no_track().radius, 1, .5f, 100))
                drawlist_infos()[dl_index].scatter_appearance();
            ImGui::PopID();
        }
        ImGui::PopID();
        if(up_index >= 0 && up_index < drawlist_infos.read().size() - 1)
            std::swap(drawlist_infos()[up_index], drawlist_infos()[up_index + 1]);
        if(down_index > 0)
            std::swap(drawlist_infos()[down_index], drawlist_infos()[down_index - 1]);

        ImGui::EndTable();

    ImGui::EndTable();

    ImGui::End();
}

void scatterplot_workbench::add_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info){
    if(drawlist_infos.read().empty()){
        attributes = globals::drawlists.read().at(drawlist_ids.front()).read().dataset_read().attributes;
        attribute_order_infos().clear();
        for(uint32_t i: util::size_range(attributes.read()))
            attribute_order_infos().emplace_back(attribute_order_info{i, true});
    }

    for(const auto& dl_id: drawlist_ids){
        // check for already added drawlists
        bool exists = false;
        for(const auto& dl: drawlist_infos.read()){
            if(dl.drawlist_id == dl_id){
                exists = true;
                break;
            }
        }
        if(exists)
            continue;

        auto& dl = globals::drawlists.write().at(dl_id).write();
        auto& ds = dl.dataset_read();
        if(drawlist_infos.read().empty()){
            // setting up the internal states
            attributes = ds.attributes;
            for(auto& attribute: attributes()){
                if(attribute.bounds.read().min == attribute.bounds.read().max){
                    float diff = (std::abs(attribute.bounds.read().max) + .1f) * .01f;
                    attribute.bounds().min -= diff;
                    attribute.bounds().max += diff;
                }
            }
            attribute_order_infos().resize(attributes.read().size());
            for(int i: util::size_range(attribute_order_infos.read()))
                attribute_order_infos.write()[i].attribut_index = i;
        }
        // check attribute consistency
        for(int var: util::size_range(attributes.read()))
            if(attributes.read()[var].id != ds.attributes[var].id)
                throw std::runtime_error{"parallel_coordinates_workbench::addDrawlist() Inconsistent attributes for the new drawlist"};

        // combining min max with new attributes
        for(int var: util::size_range(attributes.read())){
            if(attributes.read()[var].bounds.read().min > ds.attributes[var].bounds.read().min)
                attributes()[var].bounds().min = ds.attributes[var].bounds.read().min;
            if(attributes.read()[var].bounds.read().max < ds.attributes[var].bounds.read().max)
                attributes()[var].bounds().max = ds.attributes[var].bounds.read().max;
        }

        drawlist_infos.write().push_back(drawlist_info{dl_id, true, dl.appearance_drawlist});

        _update_plot_list();

        // checking histogram (large vis/axis histograms) rendering or standard rendering
        _update_registered_histograms();
    }
}

void scatterplot_workbench::remove_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info){
    for(int i: util::rev_size_range(drawlist_infos.read())){
        if(drawlist_ids.contains(drawlist_infos.read()[i].drawlist_id)){
            std::string_view dl = drawlist_infos.read()[i].drawlist_id;
            // locking registry
            auto registry_lock = globals::drawlists.read().at(dl).read().histogram_registry.const_access();
            _registered_histograms.erase(dl);
            drawlist_infos().erase(drawlist_infos().begin() + i);
        }
    }
}

std::vector<uint32_t> scatterplot_workbench::get_active_ordered_indices(){
    std::vector<uint32_t> indices;
    for(const auto& i: attribute_order_infos.read()){
        if(i.active)
            indices.push_back(i.attribut_index);
    }
    return indices;
}
}