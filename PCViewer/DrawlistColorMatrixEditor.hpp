#pragma once
#include <vector>
#include <list>
#include <string_view>
#include "imgui/imgui.h"

struct DrawList;
// editor to open a matrix view of color fields which can be edited and are applied automatically
// to the drawlists in order
class DrawlistColorMatrixEditor{
public:
    DrawlistColorMatrixEditor() = delete;
    DrawlistColorMatrixEditor(std::list<DrawList>* drawlists): _drawlists(drawlists){};

    void open();
    void draw();

    bool colorsUpdated{false};                      // is set to true if colors were edited, requires plot rerendering
private:
    std::vector<ImVec4> _matrixColors{{1,1,1,1}};   // default color is white
    int _matrixWidth{1}, _matrixHeight{1};
    std::list<DrawList>* _drawlists;
    int _selColStart{-1}, _selColEnd{-1}, _selRowStart{-1}, _selRowEnd{-1};
    const std::string_view  _popupName{"DrawlistColorMatrixEditor"};
    ImVec2 _dragStart, _matrixStart;
    bool _dragging{false};
};