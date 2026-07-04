import numpy as np

# MeerKAT L-band static RFI mask — oxkat/MeerKAT Cookbook short-baseline emitter list.
# These are quasi-static across time and absorbed by tricolour's background estimator.
LBAND_PERSISTENT_MHZ = [
    (900,  915),
    (925,  960),
    (1080, 1095),
    (1166, 1186),
    (1191, 1217),
    (1217, 1237),
    (1242, 1249),
    (1260, 1300),
    (1375, 1387),
    (1453, 1490),
    (1526, 1554),
    (1565, 1585),
    (1592, 1610),
    (1616, 1626),
    (1599, 1601),
]


def persist_chan_mask(freqs_mhz):
    freqs = np.asarray(freqs_mhz)
    m = np.zeros(freqs.shape, bool)
    for a, b in LBAND_PERSISTENT_MHZ:
        m |= (freqs >= a) & (freqs <= b)
    return m
