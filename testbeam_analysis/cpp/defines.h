#ifndef DEFINES_H
#define DEFINES_H

//#pragma pack(1) //data struct in memory alignement
#pragma pack(push, 1)

// for int64_t event_number
#ifdef _MSC_VER
#include "external/stdint.h"
#else
#include <stdint.h>
#endif

//structure of the hits
typedef struct HitInfo{
  int64_t eventNumber; //event number value (long int: -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807)
  unsigned char frame; //relative BCID value (unsigned char: 0 to 255)
  unsigned short int column; //column value (unsigned int: 0 to 65,535)
  unsigned short int row; //row value (unsigned short int: 0 to 65,535)
  unsigned short int charge; //tot value (unsigned int: 0 to 255)
} HitInfo;

//structure to store the hits with cluster info
typedef struct ClusterHitInfo{
  int64_t eventNumber; //event number value (long int: -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807)
  unsigned char frame; //relative BCID value (unsigned char: 0 to 255)
  unsigned short int column; //column value (unsigned char: 0 to 65,535)
  unsigned short int row; //row value (unsigned short int: 0 to 65,535)
  unsigned short int charge; //tot value (unsigned char: 0 to 65,535)
  int clusterID; //the cluster id of the hit
  unsigned char isSeed; //flag to mark seed pixel
  unsigned int clusterSize; //the cluster size of the cluster belonging to the hit
  unsigned int nCluster; //the number of cluster in the event
} ClusterHitInfo;

//structure to store the cluster
typedef struct ClusterInfo{
  int64_t eventNumber; //event number value (long int: -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807)
  unsigned int ID; //the cluster id of the cluster
  unsigned int n_hits; //number of all hits in all clusters
  float charge; //sum charge of all cluster hits
  unsigned short int seed_column; //seed pixel column value (unsigned short int: 0 to 65,535)
  unsigned short int seed_row; //seed pixel row value (unsigned short int: 0 to 65,535)
  double mean_column; //column mean value
  double mean_row; //row mean value
  float err_column; //column position error
  float err_row; //row position error
  unsigned int n_cluster; //number of all clusters in the event
} ClusterInfo;

#pragma pack(pop) // pop needed to suppress VS C4103 compiler warning
#endif // DEFINES_H
