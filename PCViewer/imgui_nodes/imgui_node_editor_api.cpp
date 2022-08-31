//------------------------------------------------------------------------------
// LICENSE
//   This software is dual-licensed to the public domain and under the following
//   license: you are granted a perpetual, irrevocable license to copy, modify,
//   publish, and distribute this file as you see fit.
//
// CREDITS
//   Written by Michal Cichon
//------------------------------------------------------------------------------
# include "imgui_node_editor_internal.h"
# include <algorithm>


//------------------------------------------------------------------------------
static ax::NodeEditor::Detail::EditorContext* s_Editor = nullptr;


//------------------------------------------------------------------------------
template <typename C, typename I, typename F>
static int BuildIdList(C& container, I* list, int listSize, F&& accept)
{
    if (list != nullptr)
    {
        int count = 0;
        for (auto object : container)
        {
            if (listSize <= 0)
                break;

            if (accept(object))
            {
                list[count] = I(object->ID().AsPointer());
                ++count;
                --listSize;
            }
        }

        return count;
    }
    else
        return static_cast<int>(std::count_if(container.begin(), container.end(), accept));
}


//------------------------------------------------------------------------------
ax::NodeEditor::EditorContext* ax::NodeEditor::CreateEditor(const Config* config)
{
    return reinterpret_cast<ax::NodeEditor::EditorContext*>(new ax::NodeEditor::Detail::EditorContext(config));
}

void ax::NodeEditor::DestroyEditor(EditorContext* ctx)
{
    if (GetCurrentEditor() == ctx)
        SetCurrentEditor(nullptr);

    auto editor = reinterpret_cast<ax::NodeEditor::Detail::EditorContext*>(ctx);

    delete editor;
}

void ax::NodeEditor::SetCurrentEditor(EditorContext* ctx)
{
    s_Editor = reinterpret_cast<ax::NodeEditor::Detail::EditorContext*>(ctx);
}

ax::NodeEditor::EditorContext* ax::NodeEditor::GetCurrentEditor()
{
    return reinterpret_cast<ax::NodeEditor::EditorContext*>(s_Editor);
}

ax::NodeEditor::Style& ax::NodeEditor::GetStyle()
{
    return s_Editor->GetStyle();
}

const char* ax::NodeEditor::GetStyleColorName(StyleColor colorIndex)
{
    return s_Editor->GetStyle().GetColorName(colorIndex);
}

void ax::NodeEditor::PushStyleColor(StyleColor colorIndex, const ImVec4& color)
{
    s_Editor->GetStyle().PushColor(colorIndex, color);
}

void ax::NodeEditor::PopStyleColor(int count)
{
    s_Editor->GetStyle().PopColor(count);
}

void ax::NodeEditor::PushStyleVar(StyleVar varIndex, float value)
{
    s_Editor->GetStyle().PushVar(varIndex, value);
}

void ax::NodeEditor::PushStyleVar(StyleVar varIndex, const ImVec2& value)
{
    s_Editor->GetStyle().PushVar(varIndex, value);
}

void ax::NodeEditor::PushStyleVar(StyleVar varIndex, const ImVec4& value)
{
    s_Editor->GetStyle().PushVar(varIndex, value);
}

void ax::NodeEditor::PopStyleVar(int count)
{
    s_Editor->GetStyle().PopVar(count);
}

void ax::NodeEditor::Begin(const char* id, const ImVec2& size)
{
    s_Editor->Begin(id, size);
}

void ax::NodeEditor::End()
{
    s_Editor->End();
}

void ax::NodeEditor::BeginNode(NodeId id)
{
    s_Editor->GetNodeBuilder().Begin(id);
}

void ax::NodeEditor::BeginPin(PinId id, PinKind kind)
{
    s_Editor->GetNodeBuilder().BeginPin(id, kind);
}

void ax::NodeEditor::PinRect(const ImVec2& a, const ImVec2& b)
{
    s_Editor->GetNodeBuilder().PinRect(a, b);
}

void ax::NodeEditor::PinPivotRect(const ImVec2& a, const ImVec2& b)
{
    s_Editor->GetNodeBuilder().PinPivotRect(a, b);
}

void ax::NodeEditor::PinPivotSize(const ImVec2& size)
{
    s_Editor->GetNodeBuilder().PinPivotSize(size);
}

void ax::NodeEditor::PinPivotScale(const ImVec2& scale)
{
    s_Editor->GetNodeBuilder().PinPivotScale(scale);
}

void ax::NodeEditor::PinPivotAlignment(const ImVec2& alignment)
{
    s_Editor->GetNodeBuilder().PinPivotAlignment(alignment);
}

void ax::NodeEditor::EndPin()
{
    s_Editor->GetNodeBuilder().EndPin();
}

void ax::NodeEditor::Group(const ImVec2& size)
{
    s_Editor->GetNodeBuilder().Group(size);
}

void ax::NodeEditor::EndNode()
{
    s_Editor->GetNodeBuilder().End();
}

bool ax::NodeEditor::BeginGroupHint(NodeId nodeId)
{
    return s_Editor->GetHintBuilder().Begin(nodeId);
}

ImVec2 ax::NodeEditor::GetGroupMin()
{
    return s_Editor->GetHintBuilder().GetGroupMin();
}

ImVec2 ax::NodeEditor::GetGroupMax()
{
    return s_Editor->GetHintBuilder().GetGroupMax();
}

ImDrawList* ax::NodeEditor::GetHintForegroundDrawList()
{
    return s_Editor->GetHintBuilder().GetForegroundDrawList();
}

ImDrawList* ax::NodeEditor::GetHintBackgroundDrawList()
{
    return s_Editor->GetHintBuilder().GetBackgroundDrawList();
}

void ax::NodeEditor::EndGroupHint()
{
    s_Editor->GetHintBuilder().End();
}

ImDrawList* ax::NodeEditor::GetNodeBackgroundDrawList(NodeId nodeId)
{
    if (auto node = s_Editor->FindNode(nodeId))
        return s_Editor->GetNodeBuilder().GetUserBackgroundDrawList(node);
    else
        return nullptr;
}

bool ax::NodeEditor::Link(LinkId id, PinId startPinId, PinId endPinId, const ImVec4& color/* = ImVec4(1, 1, 1, 1)*/, float thickness/* = 1.0f*/)
{
    return s_Editor->DoLink(id, startPinId, endPinId, ImColor(color), thickness);
}

void ax::NodeEditor::Flow(LinkId linkId)
{
    if (auto link = s_Editor->FindLink(linkId))
        s_Editor->Flow(link);
}

bool ax::NodeEditor::BeginCreate(const ImVec4& color, float thickness)
{
    auto& context = s_Editor->GetItemCreator();

    if (context.Begin())
    {
        context.SetStyle(ImColor(color), thickness);
        return true;
    }
    else
        return false;
}

bool ax::NodeEditor::QueryNewLink(PinId* startId, PinId* endId)
{
    using Result = ax::NodeEditor::Detail::CreateItemAction::Result;

    auto& context = s_Editor->GetItemCreator();

    return context.QueryLink(startId, endId) == Result::True;
}

bool ax::NodeEditor::QueryNewLink(PinId* startId, PinId* endId, const ImVec4& color, float thickness)
{
    using Result = ax::NodeEditor::Detail::CreateItemAction::Result;

    auto& context = s_Editor->GetItemCreator();

    auto result = context.QueryLink(startId, endId);
    if (result != Result::Indeterminate)
        context.SetStyle(ImColor(color), thickness);

    return result == Result::True;
}

bool ax::NodeEditor::QueryNewNode(PinId* pinId)
{
    using Result = ax::NodeEditor::Detail::CreateItemAction::Result;

    auto& context = s_Editor->GetItemCreator();

    return context.QueryNode(pinId) == Result::True;
}

bool ax::NodeEditor::QueryNewNode(PinId* pinId, const ImVec4& color, float thickness)
{
    using Result = ax::NodeEditor::Detail::CreateItemAction::Result;

    auto& context = s_Editor->GetItemCreator();

    auto result = context.QueryNode(pinId);
    if (result != Result::Indeterminate)
        context.SetStyle(ImColor(color), thickness);

    return result == Result::True;
}

bool ax::NodeEditor::AcceptNewItem()
{
    using Result = ax::NodeEditor::Detail::CreateItemAction::Result;

    auto& context = s_Editor->GetItemCreator();

    return context.AcceptItem() == Result::True;
}

bool ax::NodeEditor::AcceptNewItem(const ImVec4& color, float thickness)
{
    using Result = ax::NodeEditor::Detail::CreateItemAction::Result;

    auto& context = s_Editor->GetItemCreator();

    auto result = context.AcceptItem();
    if (result != Result::Indeterminate)
        context.SetStyle(ImColor(color), thickness);

    return result == Result::True;
}

void ax::NodeEditor::RejectNewItem()
{
    auto& context = s_Editor->GetItemCreator();

    context.RejectItem();
}

void ax::NodeEditor::RejectNewItem(const ImVec4& color, float thickness)
{
    using Result = ax::NodeEditor::Detail::CreateItemAction::Result;

    auto& context = s_Editor->GetItemCreator();

    if (context.RejectItem() != Result::Indeterminate)
        context.SetStyle(ImColor(color), thickness);
}

void ax::NodeEditor::EndCreate()
{
    auto& context = s_Editor->GetItemCreator();

    context.End();
}

bool ax::NodeEditor::BeginDelete()
{
    auto& context = s_Editor->GetItemDeleter();

    return context.Begin();
}

bool ax::NodeEditor::QueryDeletedLink(LinkId* linkId, PinId* startId, PinId* endId)
{
    auto& context = s_Editor->GetItemDeleter();

    return context.QueryLink(linkId, startId, endId);
}

bool ax::NodeEditor::QueryDeletedNode(NodeId* nodeId)
{
    auto& context = s_Editor->GetItemDeleter();

    return context.QueryNode(nodeId);
}

bool ax::NodeEditor::AcceptDeletedItem()
{
    auto& context = s_Editor->GetItemDeleter();

    return context.AcceptItem();
}

void ax::NodeEditor::RejectDeletedItem()
{
    auto& context = s_Editor->GetItemDeleter();

    context.RejectItem();
}

void ax::NodeEditor::EndDelete()
{
    auto& context = s_Editor->GetItemDeleter();

    context.End();
}

void ax::NodeEditor::SetNodePosition(NodeId nodeId, const ImVec2& position)
{
    s_Editor->SetNodePosition(nodeId, position);
}

ImVec2 ax::NodeEditor::GetNodePosition(NodeId nodeId)
{
    return s_Editor->GetNodePosition(nodeId);
}

ImVec2 ax::NodeEditor::GetNodeSize(NodeId nodeId)
{
    return s_Editor->GetNodeSize(nodeId);
}

void ax::NodeEditor::CenterNodeOnScreen(NodeId nodeId)
{
    if (auto node = s_Editor->FindNode(nodeId))
        node->CenterOnScreenInNextFrame();
}

void ax::NodeEditor::RestoreNodeState(NodeId nodeId)
{
    if (auto node = s_Editor->FindNode(nodeId))
        s_Editor->MarkNodeToRestoreState(node);
}

void ax::NodeEditor::Suspend()
{
    s_Editor->Suspend();
}

void ax::NodeEditor::Resume()
{
    s_Editor->Resume();
}

bool ax::NodeEditor::IsSuspended()
{
    return s_Editor->IsSuspended();
}

bool ax::NodeEditor::IsActive()
{
    return s_Editor->IsActive();
}

bool ax::NodeEditor::HasSelectionChanged()
{
    return s_Editor->HasSelectionChanged();
}

int ax::NodeEditor::GetSelectedObjectCount()
{
    return (int)s_Editor->GetSelectedObjects().size();
}

int ax::NodeEditor::GetSelectedNodes(NodeId* nodes, int size)
{
    return BuildIdList(s_Editor->GetSelectedObjects(), nodes, size, [](auto object)
    {
        return object->AsNode() != nullptr;
    });
}

int ax::NodeEditor::GetSelectedLinks(LinkId* links, int size)
{
    return BuildIdList(s_Editor->GetSelectedObjects(), links, size, [](auto object)
    {
        return object->AsLink() != nullptr;
    });
}

void ax::NodeEditor::ClearSelection()
{
    s_Editor->ClearSelection();
}

void ax::NodeEditor::SelectNode(NodeId nodeId, bool append)
{
    if (auto node = s_Editor->FindNode(nodeId))
    {
        if (append)
            s_Editor->SelectObject(node);
        else
            s_Editor->SetSelectedObject(node);
    }
}

void ax::NodeEditor::SelectLink(LinkId linkId, bool append)
{
    if (auto link = s_Editor->FindLink(linkId))
    {
        if (append)
            s_Editor->SelectObject(link);
        else
            s_Editor->SetSelectedObject(link);
    }
}

void ax::NodeEditor::DeselectNode(NodeId nodeId)
{
    if (auto node = s_Editor->FindNode(nodeId))
        s_Editor->DeselectObject(node);
}

void ax::NodeEditor::DeselectLink(LinkId linkId)
{
    if (auto link = s_Editor->FindLink(linkId))
        s_Editor->DeselectObject(link);
}

bool ax::NodeEditor::DeleteNode(NodeId nodeId)
{
    if (auto node = s_Editor->FindNode(nodeId))
        return s_Editor->GetItemDeleter().Add(node);
    else
        return false;
}

bool ax::NodeEditor::DeleteLink(LinkId linkId)
{
    if (auto link = s_Editor->FindLink(linkId))
        return s_Editor->GetItemDeleter().Add(link);
    else
        return false;
}

void ax::NodeEditor::NavigateToContent(float duration)
{
    s_Editor->NavigateTo(s_Editor->GetContentBounds(), true, duration);
}

void ax::NodeEditor::NavigateToSelection(bool zoomIn, float duration)
{
    s_Editor->NavigateTo(s_Editor->GetSelectionBounds(), zoomIn, duration);
}

bool ax::NodeEditor::ShowNodeContextMenu(NodeId* nodeId)
{
    return s_Editor->GetContextMenu().ShowNodeContextMenu(nodeId);
}

bool ax::NodeEditor::ShowPinContextMenu(PinId* pinId)
{
    return s_Editor->GetContextMenu().ShowPinContextMenu(pinId);
}

bool ax::NodeEditor::ShowLinkContextMenu(LinkId* linkId)
{
    return s_Editor->GetContextMenu().ShowLinkContextMenu(linkId);
}

bool ax::NodeEditor::ShowBackgroundContextMenu()
{
    return s_Editor->GetContextMenu().ShowBackgroundContextMenu();
}

void ax::NodeEditor::EnableShortcuts(bool enable)
{
    s_Editor->EnableShortcuts(enable);
}

bool ax::NodeEditor::AreShortcutsEnabled()
{
    return s_Editor->AreShortcutsEnabled();
}

bool ax::NodeEditor::BeginShortcut()
{
    return s_Editor->GetShortcut().Begin();
}

bool ax::NodeEditor::AcceptCut()
{
    return s_Editor->GetShortcut().AcceptCut();
}

bool ax::NodeEditor::AcceptCopy()
{
    return s_Editor->GetShortcut().AcceptCopy();
}

bool ax::NodeEditor::AcceptPaste()
{
    return s_Editor->GetShortcut().AcceptPaste();
}

bool ax::NodeEditor::AcceptDuplicate()
{
    return s_Editor->GetShortcut().AcceptDuplicate();
}

bool ax::NodeEditor::AcceptCreateNode()
{
    return s_Editor->GetShortcut().AcceptCreateNode();
}

int ax::NodeEditor::GetActionContextSize()
{
    return static_cast<int>(s_Editor->GetShortcut().m_Context.size());
}

int ax::NodeEditor::GetActionContextNodes(NodeId* nodes, int size)
{
    return BuildIdList(s_Editor->GetSelectedObjects(), nodes, size, [](auto object)
    {
        return object->AsNode() != nullptr;
    });
}

int ax::NodeEditor::GetActionContextLinks(LinkId* links, int size)
{
    return BuildIdList(s_Editor->GetSelectedObjects(), links, size, [](auto object)
    {
        return object->AsLink() != nullptr;
    });
}

void ax::NodeEditor::EndShortcut()
{
    return s_Editor->GetShortcut().End();
}

float ax::NodeEditor::GetCurrentZoom()
{
    return s_Editor->GetView().InvScale;
}

ax::NodeEditor::NodeId ax::NodeEditor::GetDoubleClickedNode()
{
    return s_Editor->GetDoubleClickedNode();
}

ax::NodeEditor::PinId ax::NodeEditor::GetDoubleClickedPin()
{
    return s_Editor->GetDoubleClickedPin();
}

ax::NodeEditor::LinkId ax::NodeEditor::GetDoubleClickedLink()
{
    return s_Editor->GetDoubleClickedLink();
}

bool ax::NodeEditor::IsBackgroundClicked()
{
    return s_Editor->IsBackgroundClicked();
}

bool ax::NodeEditor::IsBackgroundDoubleClicked()
{
    return s_Editor->IsBackgroundDoubleClicked();
}

bool ax::NodeEditor::PinHadAnyLinks(PinId pinId)
{
    return s_Editor->PinHadAnyLinks(pinId);
}

ImVec2 ax::NodeEditor::GetScreenSize()
{
    return s_Editor->GetRect().GetSize();
}

ImVec2 ax::NodeEditor::ScreenToCanvas(const ImVec2& pos)
{
    return s_Editor->ToCanvas(pos);
}

ImVec2 ax::NodeEditor::CanvasToScreen(const ImVec2& pos)
{
    return s_Editor->ToScreen(pos);
}

static float CalcMaxPopupHeightFromItemCount(int items_count)
{
    ImGuiContext& g = *GImGui;
    if (items_count <= 0)
        return FLT_MAX;
    return (g.FontSize + g.Style.ItemSpacing.y) * items_count - g.Style.ItemSpacing.y + (g.Style.WindowPadding.y * 2);
}

bool ax::NodeEditor::BeginNodeCombo(const char* label, const char* preview_value, ImGuiComboFlags flags, float scale)
{
    using namespace ImGui;

    // Always consume the SetNextWindowSizeConstraint() call in our early return paths
    ImGuiContext& g = *GImGui;
    bool has_window_size_constraint = (g.NextWindowData.Flags & ImGuiNextWindowDataFlags_HasSizeConstraint) != 0;
    g.NextWindowData.Flags &= ~ImGuiNextWindowDataFlags_HasSizeConstraint;

    ImGuiWindow* window = GetCurrentWindow();
    if (window->SkipItems)
        return false;

    IM_ASSERT((flags & (ImGuiComboFlags_NoArrowButton | ImGuiComboFlags_NoPreview)) != (ImGuiComboFlags_NoArrowButton | ImGuiComboFlags_NoPreview)); // Can't use both flags together

    const ImGuiStyle& style = g.Style;
    const ImGuiID id = window->GetID(label);

    const float arrow_size = (flags & ImGuiComboFlags_NoArrowButton) ? 0.0f : GetFrameHeight();
    const ImVec2 label_size = CalcTextSize(label, NULL, true);
    const float expected_w = CalcItemWidth();
    const float w = (flags & ImGuiComboFlags_NoPreview) ? arrow_size : expected_w;
    const ImRect frame_bb(window->DC.CursorPos, window->DC.CursorPos + ImVec2(w, label_size.y + style.FramePadding.y * 2.0f));
    const ImRect total_bb(frame_bb.Min, frame_bb.Max + ImVec2(label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f, 0.0f));
    ItemSize(total_bb, style.FramePadding.y);
    if (!ItemAdd(total_bb, id, &frame_bb))
        return false;

    bool hovered, held;
    bool pressed = ButtonBehavior(frame_bb, id, &hovered, &held);
    bool popup_open = IsPopupOpen(id, ImGuiPopupFlags_None);

    const ImU32 frame_col = GetColorU32(hovered ? ImGuiCol_FrameBgHovered : ImGuiCol_FrameBg);
    const float value_x2 = ImMax(frame_bb.Min.x, frame_bb.Max.x - arrow_size);
    RenderNavHighlight(frame_bb, id);
    if (!(flags & ImGuiComboFlags_NoPreview))
        window->DrawList->AddRectFilled(frame_bb.Min, ImVec2(value_x2, frame_bb.Max.y), frame_col, style.FrameRounding, (flags & ImGuiComboFlags_NoArrowButton) ? ImDrawCornerFlags_All : ImDrawCornerFlags_Left);
    if (!(flags & ImGuiComboFlags_NoArrowButton))
    {
        ImU32 bg_col = GetColorU32((popup_open || hovered) ? ImGuiCol_ButtonHovered : ImGuiCol_Button);
        ImU32 text_col = GetColorU32(ImGuiCol_Text);
        window->DrawList->AddRectFilled(ImVec2(value_x2, frame_bb.Min.y), frame_bb.Max, bg_col, style.FrameRounding, (w <= arrow_size) ? ImDrawCornerFlags_All : ImDrawCornerFlags_Right);
        if (value_x2 + arrow_size - style.FramePadding.x <= frame_bb.Max.x)
            RenderArrow(window->DrawList, ImVec2(value_x2 + style.FramePadding.y, frame_bb.Min.y + style.FramePadding.y), text_col, ImGuiDir_Down, 1.0f);
    }
    RenderFrameBorder(frame_bb.Min, frame_bb.Max, style.FrameRounding);
    if (preview_value != NULL && !(flags & ImGuiComboFlags_NoPreview))
        RenderTextClipped(frame_bb.Min + style.FramePadding, ImVec2(value_x2, frame_bb.Max.y), preview_value, NULL, NULL, ImVec2(0.0f, 0.0f));
    if (label_size.x > 0)
        RenderText(ImVec2(frame_bb.Max.x + style.ItemInnerSpacing.x, frame_bb.Min.y + style.FramePadding.y), label);

    if ((pressed || g.NavActivateId == id) && !popup_open)
    {
        if (window->DC.NavLayerCurrent == 0)
            window->NavLastIds[0] = id;
        OpenPopupEx(id, ImGuiPopupFlags_None);
        popup_open = true;
    }

    if (!popup_open)
        return false;

    if (has_window_size_constraint)
    {
        g.NextWindowData.Flags |= ImGuiNextWindowDataFlags_HasSizeConstraint;
        g.NextWindowData.SizeConstraintRect.Min.x = ImMax(g.NextWindowData.SizeConstraintRect.Min.x, w);
    }
    else
    {
        if ((flags & ImGuiComboFlags_HeightMask_) == 0)
            flags |= ImGuiComboFlags_HeightRegular;
        IM_ASSERT(ImIsPowerOfTwo(flags & ImGuiComboFlags_HeightMask_));    // Only one
        int popup_max_height_in_items = -1;
        if (flags & ImGuiComboFlags_HeightRegular)     popup_max_height_in_items = 8;
        else if (flags & ImGuiComboFlags_HeightSmall)  popup_max_height_in_items = 4;
        else if (flags & ImGuiComboFlags_HeightLarge)  popup_max_height_in_items = 20;
        SetNextWindowSizeConstraints(ImVec2(w, 0.0f), ImVec2(FLT_MAX, CalcMaxPopupHeightFromItemCount(popup_max_height_in_items)));
    }

    char name[16];
    ImFormatString(name, IM_ARRAYSIZE(name), "##Combo_%02d", g.BeginPopupStack.Size); // Recycle windows based on depth

    // Position the window given a custom constraint (peak into expected window size so we can position it)
    // This might be easier to express with an hypothetical SetNextWindowPosConstraints() function.
    if (ImGuiWindow* popup_window = FindWindowByName(name))
        if (popup_window->WasActive)
        {
            // Always override 'AutoPosLastDirection' to not leave a chance for a past value to affect us.
            ImVec2 size_expected = CalcWindowNextAutoFitSize(popup_window);
            if (flags & ImGuiComboFlags_PopupAlignLeft)
                popup_window->AutoPosLastDirection = ImGuiDir_Left; // "Below, Toward Left"
            else
                popup_window->AutoPosLastDirection = ImGuiDir_Down; // "Below, Toward Right (default)"
            auto prefVal = popup_window->ViewportAllowPlatformMonitorExtend;
            popup_window->ViewportAllowPlatformMonitorExtend = -1;
            ImRect r_outer = GetPopupAllowedExtentRect(popup_window);
            popup_window->ViewportAllowPlatformMonitorExtend = prefVal;
            ImVec2 pos = FindBestWindowPosForPopupEx(frame_bb.GetBL(), size_expected, &popup_window->AutoPosLastDirection, r_outer, frame_bb, ImGuiPopupPositionPolicy_ComboBox);
            SetNextWindowPos(CanvasToScreen(pos));
        }

    // We don't use BeginPopupEx() solely because we have a custom name string, which we could make an argument to BeginPopupEx()
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_Popup | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove;

    Suspend();

    // Horizontally align ourselves with the framed text
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2(style.FramePadding.x, style.WindowPadding.y));
    bool ret = ImGui::Begin(name, NULL, window_flags);
    ImGui::PopStyleVar();
    if (!ret)
    {
        EndPopup();
        IM_ASSERT(0);   // This should never happen as we tested for IsPopupOpen() above
        return false;
    }
    //ImGui::SetWindowFontScale(scale);
    return true;
}

void ax::NodeEditor::EndNodeCombo()
{
    ImGui::EndPopup();

    Resume();
}
