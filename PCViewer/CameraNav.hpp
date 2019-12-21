#pragma once

namespace CamNav {
	struct NavigationInput {
		bool w,a,s,d,shift,lm,rm,mm; //W,A,S,D keys, shift Shift, LeftMouse, RightMouse, MiddleMouse
		float mouseDeltaX,mouseDeltaY,mouseScrollDelta;
	};
}