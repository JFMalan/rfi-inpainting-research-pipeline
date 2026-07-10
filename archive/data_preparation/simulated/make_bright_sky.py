import argparse


def main(args):
    with open(args.input) as f:
        lines = f.readlines()
    out = [lines[0]]
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        parts[4] = f"{float(parts[4]) * args.scale:.4f}"
        out.append(', '.join(parts) + '\n')
    with open(args.output, 'w') as f:
        f.writelines(out)
    print(f"scaled {len(out)-1} sources by {args.scale}x -> {args.output}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data_preparation/simulated/sky_model.txt')
    ap.add_argument('--output', default='archive/data_preparation/simulated/sky_model_bright.txt')
    ap.add_argument('--scale', type=float, default=40.0)
    main(ap.parse_args())
