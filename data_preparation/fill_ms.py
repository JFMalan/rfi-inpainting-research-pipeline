import argparse
import numpy as np
from casacore.tables import table


def predict_point_sources(uvw, freqs, sources):
    """
    Compute visibility amplitudes for a list of point sources via the
    discrete visibility equation: V(u,v,w) = sum_k S_k * exp(-2pi*i*(ul+vm+w(n-1)))
    where l,m,n are direction cosines of source k relative to phase centre.

    sources: list of (l, m, flux_ref, spectral_index, ref_freq)
    uvw: (n_row, 3) in metres
    freqs: (n_chan,) in Hz
    returns: (n_row, n_chan) complex
    """
    c = 2.99792458e8
    vis = np.zeros((uvw.shape[0], len(freqs)), dtype=np.complex128)

    for l, m, flux_ref, alpha, ref_freq in sources:
        n = np.sqrt(max(1.0 - l**2 - m**2, 0.0))
        for ci, freq in enumerate(freqs):
            wavelength = c / freq
            flux = flux_ref * (freq / ref_freq) ** alpha
            phase = -2j * np.pi * (
                uvw[:, 0] * l + uvw[:, 1] * m + uvw[:, 2] * (n - 1.0)
            ) / wavelength
            vis[:, ci] += flux * np.exp(phase)

    return vis.astype(np.complex64)


def radec_to_lm(ra_deg, dec_deg, ra0_deg, dec0_deg):
    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    ra0, dec0 = np.deg2rad(ra0_deg), np.deg2rad(dec0_deg)
    dra = ra - ra0
    l = np.cos(dec) * np.sin(dra)
    m = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(dra)
    return float(l), float(m)


def load_sky_model(path):
    sources = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # name ra_d dec_d i q u v spectral_index ref_freq_hz
            ra_d, dec_d = float(parts[1]), float(parts[2])
            flux = float(parts[3])
            alpha = float(parts[8])
            ref_freq = float(parts[9])
            sources.append((ra_d, dec_d, flux, alpha, ref_freq))
    return sources


def main(args):
    ms = table(args.ms, readonly=False)
    uvw = ms.getcol('UVW')
    freqs_table = table(args.ms + '/SPECTRAL_WINDOW')
    freqs = freqs_table.getcol('CHAN_FREQ')[0]
    freqs_table.close()

    phase_table = table(args.ms + '/FIELD')
    phase_dir = phase_table.getcol('PHASE_DIR')[0, 0]  # radians
    ra0_deg = np.rad2deg(phase_dir[0])
    dec0_deg = np.rad2deg(phase_dir[1])
    phase_table.close()

    raw_sources = load_sky_model(args.sky_model)
    sources = [
        (radec_to_lm(ra, dec, ra0_deg, dec0_deg), flux, alpha, ref_freq)
        for ra, dec, flux, alpha, ref_freq in raw_sources
    ]
    sources = [(l, m, flux, alpha, ref_freq) for (l, m), flux, alpha, ref_freq in sources]

    print(f"predicting {len(sources)} sources into {len(freqs)} channels x {uvw.shape[0]} rows")

    vis = predict_point_sources(uvw, freqs, sources)

    # MS DATA column: (n_row, n_chan, n_pol) — write same vis to both pols
    n_pol = ms.getcol('DATA').shape[2]
    data = np.zeros((uvw.shape[0], len(freqs), n_pol), dtype=np.complex64)
    for p in range(n_pol):
        data[:, :, p] = vis

    ms.putcol('DATA', data)
    ms.close()
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms', required=True)
    parser.add_argument('--sky-model', required=True)
    main(parser.parse_args())
