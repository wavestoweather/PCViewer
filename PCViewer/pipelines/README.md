This folder contains all gpu pipelines needed for rendering and computing

The basic structure fo the renderer classes is the following:

class xxx
    private members

    xxx() private constructor
public:
    public members
    
    copy/move constructors = delete;

    static xxx* instance()      // here the private class constructor is called if not yet initialized

    execute()                   // executes the functionality, sbmits the work instantly and returns a semaphore for snychronization

the destructors are not needed anymore including the signaling release of reference, as everything is cleared by the global vk context