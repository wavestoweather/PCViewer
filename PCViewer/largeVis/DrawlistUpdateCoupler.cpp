#include "DrawlistUpdateCoupler.hpp"
#include "../range.hpp"

void DrawlistUpdateCoupler::drawlistWaitForUpdate(){
    _drawlistsSempahore.acquire();
}  

void DrawlistUpdateCoupler::drawlistSignalDone(){
    _updateThreadSemaphtore.release();
}  

unsigned long DrawlistUpdateCoupler::updateThreadCheckDrawlists(){
    return _updateThreadSemaphtore.peekCount();
}

void DrawlistUpdateCoupler::updateThreadWaitDrawlists(){
    for(int i: irange(drawlistCount))
        _updateThreadSemaphtore.acquire();  // waiting for drawlistCount threads
}

void DrawlistUpdateCoupler::updateThreadSignalDone(){
    _drawlistsSempahore.releaseN(drawlistCount);
}

void DrawlistUpdateCoupler::updateThreadWaitDrawlistCount(){
    _drawlistCountSemaphore.acquire();
}

void DrawlistUpdateCoupler::mainThreadSignalDrawlistCountDone(){
    _drawlistCountSemaphore.release();
}