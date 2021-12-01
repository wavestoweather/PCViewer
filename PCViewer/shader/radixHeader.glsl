
#define NUMKEYS 256        //2^8 -> resutlts in 4 passes for 32 bit keys (If higher than 1024 global scan has to be updated)
#define KPT 18             //keys per thread
#define KPLST 16           //keys per local sort thread
#define TPB 384            //threads per block
#define TPLSB 512          //threads per local sort block
#define KPB (TPB * KPT)
#define KPLSB (TPLSB * KPLST)//keys per local sort block
#define HISTOGRAMSUBGROUPREDUCTION
#define MAXVAL 0xffffffff

struct GroupInfo{
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
    uint xSize, ySize, zSize;
}
dispatchInfo; //contains dispatch info
layout(binding = 3) buffer UI
{
    uint pass;
    GroupInfo globalHistograms[]; //exlusive scan add of the global Histogram for each bucket. For each bucket an entry here exists
}
uniformInfo;

layout (constant_id = 0) const int SUBGROUP_SIZE = 32;

uint getMaskedKey(uint val, uint pass){
    pass *= 8; // 8 bit per pass
    return (val >> pass) & 0xff;
}