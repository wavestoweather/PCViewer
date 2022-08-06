#pragma once
#include "../PCUtil.h"
#include <atomic>

// class to provide functions for coupling all drawlists from a single dataset when updating the 2d histogram counts
class DrawlistUpdateCoupler{
public:
    // variables -----------------------------
    std::atomic_uint32_t drawlistCount{0};          // has to be set by the main thread when update loop is started
    std::atomic_bool drawlistCountUpdateMode{false};// set to signal that the worker thread has to call updateThreadWaitDrawlistCount
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

    unsigned long updateThreadCheckDrawlists();               // peeking if a thread waits for the worker to update
    void updateThreadWaitDrawlists();               // waiting for the drawlist scatter updates
    void updateThreadSignalDone();                  // signaling the drawwlist scater threads

    void updateThreadWaitDrawlistCount();           // update thread waiting for the drawlist count update
    void mainThreadSignalDrawlistCountDone();       // main thread signaling the update thread that the drawlist count is updated

private:
    // variables -----------------------------
    PCUtil::Semaphore _drawlistsSempahore{};        // semaphore the drawlists are waiting for to get notified that an update is ready
    PCUtil::Semaphore _updateThreadSemaphtore{};    // semaphore the update thread is waiting for (waits for drawlistCount counts)
    PCUtil::Semaphore _drawlistCountSemaphore{};    // used by the main thread to signal the update thread when the drawlist count was updated

    // functions -----------------------------
};