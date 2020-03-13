cimport numpy as cnp
cnp.import_array()  # if array is used it has to be imported, otherwise possible runtime error


cdef packed struct numpy_cluster_hit_info:
    cnp.int64_t eventNumber  # event number value
    cnp.uint16_t column  # column value
    cnp.uint16_t row  # row value
    cnp.float32_t charge  # pixel charge
    cnp.uint16_t frame  # relative BCID value
    cnp.int16_t clusterID  # the cluster id of the hit
    cnp.uint8_t isSeed  # flag to mark seed pixel
    cnp.uint32_t clusterSize  # the cluster id of the hit
    cnp.uint32_t nCluster  # the cluster id of the hit
    cnp.uint16_t tdcValue  # the tdc value of the hit
    cnp.uint16_t tdcTimestamp  # the tdc timestamp of the hit
    cnp.uint8_t tdcStatus  # the tdc status of the hit

cdef packed struct numpy_cluster_info:
    cnp.int64_t eventNumber  # event number value
    cnp.uint16_t clusterID  # the cluster id of the cluster
    cnp.uint32_t n_hits  # number of all hits in all clusters
    cnp.float32_t charge  # sum charge of all cluster hits
    cnp.uint16_t frame  # relative BCID value
    cnp.uint16_t seed_column  # column value
    cnp.uint16_t seed_row  # row value
    cnp.float64_t mean_column  # sum charge of all cluster hits
    cnp.float64_t mean_row  # sum charge of all cluster hits
    cnp.uint16_t tdcValue  # the tdc value of the hit
    cnp.uint16_t tdcTimestamp  # the tdc timestamp of the hit
    cnp.uint8_t tdcStatus  # the tdc status of the hit
    cnp.float32_t err_column  # sum charge of all cluster hits
    cnp.float32_t err_row  # sum charge of all cluster hits
    cnp.uint32_t n_cluster  # number of all clusters in the event
    cnp.int64_t cluster_shape  # cluster shape
