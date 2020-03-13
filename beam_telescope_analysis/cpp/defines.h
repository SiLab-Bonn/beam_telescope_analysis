#ifndef DEFINES_H
#define DEFINES_H

//#pragma pack(1) //data struct in memory alignement
#pragma pack(push, 1)

// for int64_t event_number
#ifdef _MSC_VER
#if _MSC_VER >= 1600  // MSVC++ 10 (2010)
#include <stdint.h>
#else
#include "external/stdint.h"
#endif
#else
#include <stdint.h>
#endif

//structure to store the hits with cluster info
typedef struct ClusterHitInfo{
  int64_t eventNumber;  // event number value
  uint16_t column;  // column value
  uint16_t row; // row value
  float charge; // pixel charge
  uint16_t frame;  // relative BCID value
  int16_t clusterID;  // the cluster id of the hit
  uint8_t isSeed;  // flag to mark seed pixel
  uint32_t clusterSize;  // the cluster size of the cluster belonging to the hit
  uint32_t nCluster;  // the number of cluster in the event
  uint16_t tdcValue;  // the tdc value of cluster in the event
  uint16_t tdcTimestamp;  // the tdc timestamp of cluster in the event
  uint8_t tdcStatus;  // the tdc status of cluster in the event
} ClusterHitInfo;

//structure to store the cluster
typedef struct ClusterInfo{
  int64_t eventNumber;  // event number value
  uint16_t ID;  // the cluster id of the cluster
  uint32_t n_hits;  // number of all hits in all clusters
  float charge;  // sum charge of all cluster hits
  uint16_t frame;  // relative BCID value
  uint16_t seed_column;  // seed pixel column value
  uint16_t seed_row;  // seed pixel row value
  double mean_column;  // column mean value
  double mean_row;  // row mean value
  uint16_t tdcValue;  // the tdc value of cluster in the event
  uint16_t tdcTimestamp;  // the tdc timestamp of cluster in the event
  uint8_t tdcStatus;  // the tdc status of cluster in the event
  float err_column;  // column position error
  float err_row;  // row position error
  uint32_t n_cluster;  // number of all clusters in the event
  int64_t cluster_shape;  // cluster shape
} ClusterInfo;

#pragma pack(pop) // pop needed to suppress VS C4103 compiler warning
#endif // DEFINES_H
