import tables as tb


class ClusterHitInfoTable(tb.IsDescription):
    event_number = tb.Int64Col(pos=0)
    column = tb.UInt16Col(pos=1)
    row = tb.UInt16Col(pos=2)
    charge = tb.Float32Col(pos=3)
    frame = tb.UInt16Col(pos=4)
    cluster_id = tb.Int16Col(pos=5)
    is_seed = tb.UInt8Col(pos=6)
    cluster_size = tb.UInt32Col(pos=7)
    n_cluster = tb.UInt32Col(pos=8)


class ClusterInfoTable(tb.IsDescription):
    event_number = tb.Int64Col(pos=0)
    cluster_id = tb.UInt16Col(pos=1)
    n_hits = tb.UInt32Col(pos=2)
    charge = tb.Float32Col(pos=3)
    frame = tb.UInt16Col(pos=4)
    seed_column = tb.UInt16Col(pos=5)
    seed_row = tb.UInt16Col(pos=6)
    mean_column = tb.Float64Col(pos=7)
    mean_row = tb.Float64Col(pos=8)
    err_column = tb.Float32Col(pos=9)
    err_row = tb.Float32Col(pos=10)
    n_cluster = tb.UInt32Col(pos=11)
    cluster_shape = tb.Int64Col(pos=12)
