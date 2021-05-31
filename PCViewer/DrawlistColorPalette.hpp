#pragma once

#include <vector>
#include <string>
#include <cmath>
#include "SettingsManager.h"
#include "Color.h"
#include "imgui/imgui.h"

class DrawlistColorPalette{
public:
    struct Color{
        Color(const rgb& r): x(r.r), y(r.g), z(r.b), w(1){}
        float x,y,z,w;
        operator rgb() const {return {x,y,z};}
    };

    DrawlistColorPalette(SettingsManager* settingsManager): settingType("DrawListColorPalette"), settingId("DrawListColorPalette"), popupName("DrawListColorPalette"), settingsManager(settingsManager), curColor(0){
        SettingsManager::Setting& s = settingsManager->getSetting(settingId);
        if(s == settingsManager->notFound){
            //default colors are being filled
            for(int h = 0; h < 36; h += colorDistance){
                float hue = h * 10;
                colors.push_back(Color(hsv2rgb({hue, saturation, value})));
            }
            SettingsManager::Setting set{};
            set.id = settingId;
            set.type = settingType;
            set.byteLength = colors.size() * sizeof(Color);
            set.data = colors.data();
            settingsManager->addSetting(set);
        }
        else{
            Color* d = static_cast<Color*>(s.data);
            colors = std::vector<Color>(d, d + s.byteLength / sizeof(Color));
        }
    }

    const std::string settingType;
    const std::string settingId;
    const std::string popupName;
    SettingsManager* settingsManager;
    //adding new colors should be done by directly accessing the colors member
    std::vector<Color> colors;
    uint curColor;
    float saturation = .8f;
    float value = .8f;
    int colorDistance = 4;

    Color& getNextColor(){
        if(++curColor >= colors.size()) curColor -= colors.size();
        return colors[curColor];
    }

    void openColorPaletteEditor(){
        ImGui::OpenPopup(popupName.c_str());
    }

    //
    void drawColorPaletteEditor(){
        if(ImGui::BeginPopup(popupName.c_str(), ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDecoration)){
            float size = 400;
            int labelAmt = colors.size() * 2;
            ImVec2 baseCursorPos = ImGui::GetCursorPos();
            int del = -1;
            for(int i = 0; i < labelAmt; ++i){
                ImGui::SetCursorPos({baseCursorPos.x + size / 2 + std::sin(float(i) / labelAmt * 2 * M_PI) * size / 2, baseCursorPos.y + size / 2 + std::cos(float(i)/ labelAmt * 2 * M_PI) * size / 2});
                if(i & 1){ // + symbol to add a new color
                    if(ImGui::Button(("+##newColor" + std::to_string(i)).c_str())){
                        colors.insert(colors.begin() + (i >> 1) + 1, Color({1,1,1}));
                        savePalette();
                    }
                }
                else{       //color widget
                    if(ImGui::ColorEdit4(("##Editor" + std::to_string(i)).c_str(), &colors[i >> 1].x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel) && !ImGui::IsMouseDown(1)){
                        savePalette();
                    }
                    if(ImGui::IsItemClicked() && ImGui::IsMouseClicked(0)){
                        del = i >> 1;
                    }
                }
            }
            if(del >= 0){
                colors.erase(colors.begin() + del);
                savePalette();
            }
            //ImGui::PushItemWidth(100);
            //ImGui::SetCursorPos({baseCursorPos.x + size / 2, baseCursorPos.y + size / 2});
            //ImGui::DragInt("Color distance", &colorDistance, 1, 1, 36);
            //ImGui::SetCursorPos({baseCursorPos.x + size / 2, baseCursorPos.y + size / 2});
            //ImGui::DragFloat("Saturation", &saturation, .01, 0, 1);
            //ImGui::SetCursorPos({baseCursorPos.x + size / 2, baseCursorPos.y + size / 2});
            //ImGui::DragFloat("Value", &value, .01, 0, 1);
            //if(ImGui::Button("Update Color")){
            //    colors.clear();
            //    for(int h = 0; h < 36; h += colorDistance){
            //        float hue = h * 10;
            //        colors.push_back(Color(hsv2rgb({hue, saturation, value})));
            //    }
            //    savePalette();
            //}
            //ImGui::PopItemWidth();
            ImGui::EndPopup();
        }
    }

    //function to manually write back to settings file
    void savePalette(){
        SettingsManager::Setting set{};
        set.id = settingId;
        set.type = settingType;
        set.byteLength = colors.size() * sizeof(Color);
        set.data = colors.data();
        settingsManager->addSetting(set);
    }
};