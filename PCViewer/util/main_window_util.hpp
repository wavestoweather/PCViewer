#pragma once
#include <imgui.h>
#include <workbench_base.hpp>
#include <logger.hpp>
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
                    if (ImGui::MenuItem("Load File"))
                    {
                        if (globals::file_dialog_thread.joinable())
                            logger << logging::error_prefix << " File dialog still running. Close before opening another one" << logging::endl;
                        else
                        {
                            globals::file_dialog_thread = std::thread([&]
                                                                      {
                                                                          auto path = tinyfd_openFileDialog(NULL, NULL, 2, file_patterns, NULL, 1);
                                                                          if (path)
                                                                          {
                                                                              std::string_view p(path);
                                                                              for (std::string_view c; getline(p, c, '|');)
                                                                                  globals::paths_to_open.push_back(std::string(c));
                                                                          }
                                                                      });
                        }
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