#pragma once
#include <imgui.h>
#include <workbench_base.hpp>
#include <logger.hpp>
#include <settings_manager.hpp>
#include <tinyfiledialogs.h>

namespace util
{
    namespace main_window
    {
        inline void menu_bar()
        {
            if (ImGui::BeginMenuBar())
            {
                static constexpr const char *file_patterns[2]{"*.csv", "*.nc"};
                if (ImGui::BeginMenu("File"))
                {
                    if (ImGui::MenuItem("Open File"))
                    {
                        if (globals::file_dialog_thread.joinable())
                            globals::file_dialog_thread.join();
                        globals::file_dialog_thread = std::thread([&]
                                                                  {
                                                                      auto path = tinyfd_openFileDialog(NULL, NULL, 2, file_patterns, NULL, 1);
                                                                      if (path)
                                                                      {
                                                                          std::string_view p(path);
                                                                          for (std::string_view c; getline(p, c, '|');)
                                                                              globals::paths_to_open.emplace_back(c);
                                                                      }
                                                                  });
                    }
                    if (ImGui::BeginMenu("Recent Files"))
                    {
                        const auto &prev = globals::settings_manager.get_setting("recent_data_files");
                        if (prev.is_object())
                        {
                            for (const auto &path : prev["files"].get<crude_json::array>())
                            {
                                const auto &p = path.get<std::string>();
                                if (ImGui::MenuItem(p.c_str()))
                                    globals::paths_to_open.emplace_back(p);
                            }
                        }
                        ImGui::EndMenu();
                    }
                    ImGui::EndMenu();
                }
                if (ImGui::BeginMenu("Edit"))
                {
                    ImGui::ShowFontSelector("Select font");
                    if (ImGui::BeginCombo("Logging level", logging::level_names[logger.logging_level].data()))
                    {
                        for (auto level : structures::enum_iteration<logging::level>{})
                        {
                            if (ImGui::MenuItem(logging::level_names[level].data()))
                                logger.logging_level = level;
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::EndMenu();
                }
                if (ImGui::BeginMenu("View"))
                {
                    // making all workbenches selectable
                    for (const auto &wb : globals::workbenches)
                    {
                        if (wb.get() == globals::primary_workbench)
                            continue;
                        ImGui::MenuItem(wb->id.c_str(), "", &wb->active);
                    }
                    ImGui::EndMenu();
                }
                ImGui::EndMenuBar();
            }
        }
    }
}