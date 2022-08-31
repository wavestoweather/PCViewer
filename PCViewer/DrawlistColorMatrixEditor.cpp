#define NOSTATICS
#include "DrawlistColorMatrixEditor.hpp"
#include "range.hpp"
#include "Structures.hpp"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui/imgui_internal.h"
#include <string>

auto sqDist = [](auto a, auto b){return (a * a) + (b * b);};

void DrawlistColorMatrixEditor::open(){
    ImGui::OpenPopup(_popupName.data());
}

void DrawlistColorMatrixEditor::draw(){
    if(ImGui::BeginPopup(_popupName.data(), ImGuiWindowFlags_AlwaysAutoResize)){
        const size_t colorPickerWidth = 20;    // TODO adjust, is size_t to avoid overflow
        const float widthHalf = colorPickerWidth / 2;

        // section for setting the grid dimensions
        if(ImGui::InputInt2("Matrix width/height", &_matrixWidth)){
            if(_matrixWidth < 1)
                _matrixWidth = 1;
            if(_matrixHeight < 1)
                _matrixHeight = 1;
            if(_matrixWidth * _matrixHeight > _matrixColors.size())
                _matrixColors.resize(_matrixWidth * _matrixHeight, _matrixColors.back());
        }

        static bool boxSelect = false;
        if(ImGui::Checkbox("Activate Box Selection", &boxSelect));
        ImGui::SameLine();
        if(ImGui::Button("Clear Box Select"))
            _selRowStart = _selRowEnd = _selColStart = _selColEnd = -1;

        // row to set a whole column to a certain color value
        // skip one color mat place
        auto preSpacing = ImGui::GetStyle().ItemSpacing;
        ImGui::GetStyle().ItemSpacing = {0,0};
        _matrixStart = ImGui::GetCursorPos();
        for(int i: irange(_matrixWidth * _matrixHeight)){
            int col = i % _matrixWidth;
            int row = i / _matrixWidth;
            if(col != 0)
                ImGui::SameLine(colorPickerWidth * col + ImGui::GetStyle().WindowPadding.x);
            if(ImGui::ColorEdit4(("##CM" + std::to_string(i)).c_str(), &_matrixColors[i].x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel)){
                auto dl = _drawlists->begin(); 
                if(i < _drawlists->size()){
                    std::advance(dl, i);
                    std::memcpy(&dl->color.x, &_matrixColors[i].x, sizeof(ImVec4));
                }
                colorsUpdated = true;
                if(_selColStart != 1){
                    for(int c: irange(_selColStart, _selColEnd)){
                        for(int r: irange(_selRowStart, _selRowEnd)){
                            int ind = r * _matrixWidth + c;
                            _matrixColors[ind] = _matrixColors[i];
                            if(i < _drawlists->size()){
                                dl = _drawlists->begin(); std::advance(dl, ind);
                                std::memcpy(&dl->color.x, &_matrixColors[i].x, sizeof(ImVec4));
                            }
                        }
                    }
                }
            }
            if(ImGui::IsItemClicked()){
                if(col < _selColStart || col >= _selColEnd || row < _selRowStart || row >= _selRowEnd)
                    _selRowStart = _selRowEnd = _selColStart = _selColEnd = -1;
            }
        }
        ImGui::GetStyle().ItemSpacing = preSpacing;
        // drag handling
        if(ImGui::IsMouseClicked(0) && boxSelect){
            _dragging = true;
            _dragStart = ImGui::GetMousePos() - ImGui::GetWindowPos();
        }

        if(ImGui::IsMouseReleased(0) && _dragging){
            _dragging = false;
            ImVec2 dragEnd = ImGui::GetMousePos() - ImGui::GetWindowPos();
            ImVec2 diff = _dragStart - dragEnd;
            const float minDist = 5;
            if(sqDist(diff.x, diff.y) > minDist){
                ImVec2 a, b;    // a is left upper corner, b is right lower corner
                if(_dragStart.x < dragEnd.x){    
                    a.x = _dragStart.x;
                    b.x = dragEnd.x;
                }
                else{
                    a.x = dragEnd.x;
                    b.x = _dragStart.x;
                }
                if(_dragStart.y < dragEnd.y){
                    a.y = _dragStart.y;
                    b.y = dragEnd.y;
                }
                else{
                    a.y = dragEnd.y;
                    b.y = _dragStart.y;
                }

                _selRowStart = (a.y - _matrixStart.y + widthHalf) / colorPickerWidth;
                _selRowEnd = (b.y - _matrixStart.y + widthHalf) / colorPickerWidth;
                _selColStart = (a.x - _matrixStart.x + widthHalf) / colorPickerWidth;
                _selColEnd = (b.x - _matrixStart.x + widthHalf) / colorPickerWidth;

                _selRowStart = std::clamp(_selRowStart, 0, _matrixHeight - 1);
                _selRowEnd = std::clamp(_selRowEnd, 1, _matrixHeight);
                _selColStart = std::clamp(_selColStart, 0, _matrixWidth - 1);
                _selColEnd = std::clamp(_selColEnd, 1, _matrixWidth);
            }
            else{
                _selRowStart = _selRowEnd = _selColStart = _selColEnd = -1;
            }
            boxSelect = false;
        }

        const auto rectCol = IM_COL32(30, 30, 255, 255);
        if(_selColStart >= 0){
            float w = colorPickerWidth;
            float h = colorPickerWidth + 1;
            ImVec2 a = _matrixStart + ImVec2{_selColStart * w, _selRowStart * h} + ImGui::GetWindowPos();
            ImVec2 b = _matrixStart + ImVec2{_selColEnd * w, _selRowEnd * h} + ImGui::GetWindowPos();
            ImGui::GetWindowDrawList()->AddRect(a, b, rectCol, 0, 15, 3);
        }
        if(_dragging){
            ImGui::GetWindowDrawList()->AddRect(_dragStart + ImGui::GetWindowPos(), ImGui::GetMousePos(), rectCol, 0, 15, 3);
        }

        ImGui::EndPopup();
    }
}