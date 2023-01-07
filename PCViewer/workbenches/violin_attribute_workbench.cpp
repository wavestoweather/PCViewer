#include "violin_attribute_workbench.hpp"

namespace workbenches{
violin_attribute_workbench::violin_attribute_workbench(std::string_view id): workbench(id){
}

void violin_attribute_workbench::show(){
    if(!active)
        return;

    ImGui::Begin(id.data(), &active);

    // violin plots ---------------------------------------------------------------------------------------

    

    // drawlists and settings -----------------------------------------------------------------------------

    ImGui::End();
}
}