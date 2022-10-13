#pragma once

namespace CamNav {
    struct NavigationInput {
        bool w,a,s,d,q,e,shift,lm,rm,mm; //W,A,S,D,Q,E keys, shift Shift, LeftMouse, RightMouse, MiddleMouse
        float mouseDeltaX,mouseDeltaY,mouseScrollDelta;
    };
}