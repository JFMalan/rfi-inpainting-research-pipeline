import argparse
import numpy as np


HEADER = ("Format = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, "
          "ReferenceFrequency='1280000000.0', MajorAxis, MinorAxis, Orientation\n")


def hms(h):
    hh = int(h); mm = int((h - hh) * 60); ss = (((h - hh) * 60) - mm) * 60
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"


def dms(d):
    sign = '-' if d < 0 else ''
    d = abs(d)
    dd = int(d); mm = int((d - dd) * 60); ss = (((d - dd) * 60) - mm) * 60
    return f"{sign}{dd:02d}.{mm:02d}.{ss:06.3f}"


def main(args):
    rng = np.random.default_rng(args.seed)
    ra_c, dec_c = 4.0, -30.0          # field centre (hours, deg) — matches the sim pointing
    n = rng.integers(args.n_min, args.n_max + 1)
    lines = [HEADER]
    for i in range(n):
        ra = ra_c + rng.uniform(-0.15, 0.15)            # ~+-2 deg spread in RA hours
        dec = dec_c + rng.uniform(-1.0, 1.0)
        flux = float(np.round(10 ** rng.uniform(np.log10(args.flux_min), np.log10(args.flux_max)), 4))
        si = round(rng.uniform(-0.9, -0.5), 2)
        lines.append(f"src{i+1:03d}, POINT, {hms(ra)}, {dms(dec)}, {flux}, [{si}], "
                     f"false, 1280000000.0, , , \n")
    with open(args.output, 'w') as f:
        f.writelines(lines)
    print(f"wrote {n} sources (flux {args.flux_min}-{args.flux_max} Jy, seed {args.seed}) -> {args.output}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-min', type=int, default=20)
    ap.add_argument('--n-max', type=int, default=30)
    ap.add_argument('--flux-min', type=float, default=0.1)
    ap.add_argument('--flux-max', type=float, default=5.0)
    main(ap.parse_args())
