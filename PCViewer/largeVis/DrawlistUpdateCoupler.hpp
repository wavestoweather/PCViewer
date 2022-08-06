#pragma once
#include "../PCUtil.h"
#include <atomic>

// class to provide functions for coupling all drawlists from a single dataset when updating the 2d histogram counts
class DrawlistUpdateCoupler{
public:
    // variables -----------------------------
    std::atomic_uint32_t drawlistCount{};           // has to be set by the main thread when update loop is started
    // functions -----------------------------
    DrawlistUpdateCoupler() = default;
    // no copy
    DrawlistUpdateCoupler(const DrawlistUpdateCoupler&) = delete;
    DrawlistUpdateCoupler& operator=(const DrawlistUpdateCoupler&) = delete;
    // default move
    DrawlistUpdateCoupler(DrawlistUpdateCoupler&&) = default;
    DrawlistUpdateCoupler& operator=(DrawlistUpdateCoupler&&) = default;

    void drawlistWaitForUpdate();                   // waiting for the update thread
    void drawlistSignalDone();                      // signaling the update thread

    void updateThreadWaitDrawlists();               // waiting for the drawlist scatter updates
    void updateThreadSignalDone();                  // signaling the drawwlist scater threads

private:
    // variables -----------------------------
    PCUtil::Semaphore _drawlistsSempahore{};        // semaphore the drawlists are waiting for to get notified that an update is ready
    PCUtil::Semaphore _updateThreadSemaphtore{};    // semaphore the update thread is waiting for (waits for drawlistCount counts)

    // functions -----------------------------
};