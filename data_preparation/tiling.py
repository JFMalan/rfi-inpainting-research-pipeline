import numpy as np


def freq_tile_starts(n_chan, size):
    if n_chan <= size:
        return [0]
    n = int(np.ceil(n_chan / size))
    return [int(round(t * (n_chan - size) / (n - 1))) for t in range(n)]


def freq_tile_width(n_chan, size):
    return min(n_chan, size)


def feather_weight(flo, nc, starts, size):
    w = np.ones(nc, dtype=np.float32)
    g = np.arange(flo, flo + nc)
    for s in starts:
        if s == flo:
            continue
        ov_lo = max(flo, s)
        ov_hi = min(flo + nc, s + size)
        if ov_hi <= ov_lo:
            continue
        ramp = np.clip((g - ov_lo) / float(ov_hi - ov_lo), 0.0, 1.0)
        w *= (1.0 - ramp) if s > flo else ramp
    return w


def time_extent(n_time, size):
    if n_time >= size:
        return (n_time - size) // 2, size
    return 0, n_time


if __name__ == '__main__':
    for N in (898, 512, 400, 1024, 1300):
        S = 512
        starts = freq_tile_starts(N, S)
        nc = freq_tile_width(N, S)
        cover = np.zeros(N)
        wsum = np.zeros(N)
        for s in starts:
            w = feather_weight(s, nc, starts, S)
            cover[s:s + nc] += 1
            wsum[s:s + nc] += w
        overlaps = [(starts[i] + nc - starts[i + 1]) for i in range(len(starts) - 1)]
        pou_ok = np.allclose(wsum[cover > 0], 1.0, atol=1e-6)
        print(f"N={N:5d} S={S} -> tiles={len(starts)} starts={starts} width={nc} "
              f"overlaps={overlaps} covered={int((cover > 0).sum())}/{N} PoU_ok={pou_ok} "
              f"max_cover={int(cover.max())}")
    print()
    starts = freq_tile_starts(898, 512)
    w0 = feather_weight(starts[0], 512, starts, 512)
    w1 = feather_weight(starts[1], 512, starts, 512)
    print(f"tile0 start={starts[0]} weight at ch[384,386,449,510,512region]: "
          f"{w0[384]:.3f} {w0[386]:.3f} {w0[449]:.3f} {w0[511]:.3f}")
    print(f"tile1 start={starts[1]} weight at native ch 386,449,512,513: "
          f"{w1[386 - starts[1]]:.3f} {w1[449 - starts[1]]:.3f} {w1[512 - starts[1]]:.3f} {w1[513 - starts[1]]:.3f}")
    print(f"crossover (w=0.5) native ch: {starts[0] + int(np.argmin(np.abs(w0 - 0.5)))}")
    for T in (540, 512, 371, 297):
        tlo, th = time_extent(T, 512)
        print(f"T={T} -> time_lo={tlo} height={th} ({'crop' if th == 512 else 'resize'})")
