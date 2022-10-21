#pragma once

#include <vector>
#include <string>
#include <cmath>
#include "SettingsManager.h"
#include "Color.h"
#include "imgui/imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui/imgui_internal.h"

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
            resetPalette();
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
    uint32_t curColor;
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
            ImVec2 screenCorrection  = ImGui::GetCursorScreenPos() - ImGui::GetCursorPos();
            ImVec2 baseCursorPos = ImGui::GetCursorPos();
            ImVec2 center{baseCursorPos.x + size / 2, baseCursorPos.y + size / 2} ;
            int del = -1;
            for(int i = 0; i < labelAmt; ++i){
                ImVec2 curPos{center.x + std::sin(float(i) / labelAmt * 2 * float(M_PI)) * size / 2, center.y + std::cos(float(i)/ labelAmt * 2 * float(M_PI)) * size / 2};
                ImGui::SetCursorPos(curPos);
                if(i & 1){ // + symbol to add a new color
                    if(ImGui::Button(("+##newColor" + std::to_string(i)).c_str())){
                        colors.insert(colors.begin() + (i >> 1) + 1, Color({1,1,1}));
                        savePalette();
                    }
                }
                else{       //color widget
                    if(i == curColor){
                        curPos = curPos + screenCorrection;
                        ImGui::GetWindowDrawList()->AddRectFilled({curPos.x - 10, curPos.y - 10}, {curPos.x + 500, curPos.y + 500}, ImGui::GetColorU32({0, .7f, .7f, 1}), 2);
                    }
                    if(ImGui::ColorEdit4(("##Editor" + std::to_string(i)).c_str(), &colors[i >> 1].x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel) && !ImGui::IsMouseDown(1)){
                       savePalette();
                    }
                    if(ImGui::IsItemClicked(1)){    //delete on rightclick
                        del = i >> 1;
                    }
                }
            }
            const float buttonWidth = 200;
            ImGui::SetCursorPos({center.x - buttonWidth / 2, center.y - ImGui::GetTextLineHeight() / 2});
            if(ImGui::Button("reset palette", {buttonWidth, 0})){
                resetPalette();
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

    void resetPalette(){
        colors.clear();
        for(int h = 0; h < 36; h += colorDistance){
            float hue = h * 10;
            colors.push_back(Color(hsv2rgb({hue, saturation, value})));
        }
        curColor = colors.size() - 1;
        colors.push_back(Color({1,1,1}));
    }
};