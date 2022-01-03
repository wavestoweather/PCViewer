
#define NUMKEYS 256        //2^8 -> resutlts in 4 passes for 32 bit keys (If higher than 1024 global scan has to be updated)
#define KPT 18             //keys per thread
#define KPLST 16           //keys per local sort thread
#define TPB 384            //threads per block
#define TPLSB 512          //threads per local sort block
#define KPB (TPB * KPT)
#define KPLSB (TPLSB * KPLST)//keys per local sort block
//#define HISTOGRAMSUBGROUPREDUCTION
#define HISTOGRAMTHREADREDUCTION    //no optimization .08 ms thread reduction .09 ms subgroup thread reduction .27 ms
#define MAXVAL 0xffffffff
#define LOADVERSION1

struct GroupInfo{
    uint globalHistIndex;
    uint startOffset;  //total start offset for this work group
    uint keyCount[NUMKEYS];
};

struct GlobalHistogram{
    uint startOffset;
    uint endOffset;
    uint keyCount[NUMKEYS];
};

// keys are always sorted from the front to the back buffer
layout(binding = 0) buffer K{uint k[];}keys[2]; // front and back buffer for keys
layout(binding = 1) buffer GI
{
    GroupInfo i[];
}
groupInfos;   //contains group histograms
layout(binding = 2) buffer DI
{
    uint xSize, ySize, zSize, xSizeScan, ySizeScan, zSizScan, xCtrlSize, yCtrlSize, zCtrlSize, xLocalSortSize, yLocalSortSize, zLocalSortSize;
}
dispatchInfo; //contains dispatch info
layout(binding = 3) buffer UI
{
    uint pass;                          //pass is always managed in the front uniform info, the back pass info is available for control shader as copy
    uint amtOfGlobalHistograms;
    uint amtOfBlocks;                   //amt of blocks/workgroups that have to be executed
    GlobalHistogram globalHistograms[]; //exlusive scan add of the global Histogram for each bucket. For each bucket an entry here exists
}
uniformInfo[2]; //there exists front and back buffer to be able to append global histograms.
                //first histogram is in uniformInfo[0]

struct LocalSort{
    uint begin, end, front;     //begin and end are pointers to the array bounds, front is the data array where the data is stored
};
layout(binding = 4) buffer LI
{
    LocalSort sorts[];
}localSortInfo;

layout (constant_id = 0) const int SUBGROUP_SIZE = 32;

uint getMaskedKey(uint val, uint pass){
    pass = 3 - pass; //inverting the pass to start from the front
    pass *= 8; // 8 bit per pass
    return (val >> pass) & 0xff;
}