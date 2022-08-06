#include "DrawlistUpdateCoupler.hpp"
#include "../range.hpp"

void DrawlistUpdateCoupler::drawlistWaitForUpdate(){
    _drawlistsSempahore.acquire();
}  

void DrawlistUpdateCoupler::drawlistSignalDone(){
    _updateThreadSemaphtore.release();
}  

void DrawlistUpdateCoupler::updateThreadWaitDrawlists(){
    for(int i: irange(drawlistCount))
        _updateThreadSemaphtore.acquire();  // waiting for drawlistCount threads
}

void DrawlistUpdateCoupler::updateThreadSignalDone(){
    _drawlistsSempahore.releaseN(drawlistCount);
}