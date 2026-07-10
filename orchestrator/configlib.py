from pathlib import Path

try:
    import yaml as _pyyaml
except ImportError:
    _pyyaml = None      # ilifu login node has no PyYAML; fall back to the subset parser

REPO = Path(__file__).resolve().parent.parent


# Subset parser for the config files in configs/: 2-space block maps, "- " block lists,
# flow lists/maps, scalars, comments. No anchors, no multiline strings, no multi-doc.
# tests/config_check.py cross-checks it against PyYAML inside a container.

def _strip_comment(line):
    out, q = [], None
    for ch in line:
        if q:
            out.append(ch)
            if ch == q:
                q = None
        elif ch in '\'"':
            q = ch
            out.append(ch)
        elif ch == '#':
            break
        else:
            out.append(ch)
    return ''.join(out).rstrip()


def _scalar(tok):
    tok = tok.strip()
    if tok == '' or tok in ('null', '~'):
        return None
    if tok in ('true', 'True'):
        return True
    if tok in ('false', 'False'):
        return False
    if len(tok) > 1 and tok[0] in '\'"' and tok[-1] == tok[0]:
        return tok[1:-1]
    for cast in (int, float):
        try:
            return cast(tok)
        except ValueError:
            pass
    return tok


def _split_top(s, sep):
    parts, depth, q, cur = [], 0, None, []
    for ch in s:
        if q:
            cur.append(ch)
            if ch == q:
                q = None
        elif ch in '\'"':
            q = ch
            cur.append(ch)
        elif ch in '[{':
            depth += 1
            cur.append(ch)
        elif ch in ']}':
            depth -= 1
            cur.append(ch)
        elif ch == sep and depth == 0:
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append(''.join(cur))
    return parts


def _value(tok):
    tok = tok.strip()
    if tok.startswith('['):
        inner = tok[1:-1].strip()
        return [] if not inner else [_value(p) for p in _split_top(inner, ',')]
    if tok.startswith('{'):
        inner = tok[1:-1].strip()
        out = {}
        for p in _split_top(inner, ','):
            k, _, v = p.partition(':')
            out[_scalar(k)] = _value(v)
        return out
    return _scalar(tok)


def _parse_lines(lines, i, indent):
    if lines[i][1].startswith('- '):
        out = []
        while i < len(lines) and lines[i][0] == indent and lines[i][1].startswith('- '):
            out.append(_value(lines[i][1][2:]))
            i += 1
        return out, i
    out = {}
    while i < len(lines) and lines[i][0] == indent:
        text = lines[i][1]
        key, _, rest = text.partition(':')
        key = _scalar(key)
        rest = rest.strip()
        if rest:
            out[key] = _value(rest)
            i += 1
        else:
            i += 1
            if i < len(lines) and lines[i][0] > indent:
                out[key], i = _parse_lines(lines, i, lines[i][0])
            else:
                out[key] = None
    return out, i


def parse_yaml(text):
    if _pyyaml is not None:
        return _pyyaml.safe_load(text)
    lines = []
    for raw in text.splitlines():
        s = _strip_comment(raw)
        if not s.strip():
            continue
        lines.append((len(s) - len(s.lstrip()), s.strip()))
    result, i = _parse_lines(lines, 0, lines[0][0]) if lines else ({}, 0)
    if i != len(lines):
        raise ValueError(f'config parse stopped at line {i}: {lines[i]}')
    return result


def load_experiment(path):
    path = Path(path)
    exp = parse_yaml(path.read_text())
    tel_path = REPO / 'configs' / 'telescope' / f"{exp['telescope']}.yaml"
    tel = parse_yaml(tel_path.read_text())
    _validate(exp, tel)
    return exp, tel


def _validate(exp, tel):
    for key in ('name', 'telescope', 'seed', 'paths', 'containers', 'sim', 'train'):
        if key not in exp:
            raise ValueError(f'experiment config missing "{key}"')
    for key in ('name', 'dump_time_s', 'band', 'sefd', 'persistent_rfi_mhz', 'sky'):
        if key not in tel:
            raise ValueError(f'telescope config missing "{key}"')
    if len(tel['sefd']['nodes_mhz']) != len(tel['sefd']['nodes_jy']):
        raise ValueError('sefd nodes_mhz and nodes_jy length mismatch')
    lo, hi = exp['sim']['noise_scale']
    if not (0 <= lo <= hi):
        raise ValueError('sim.noise_scale must be an ascending [lo, hi] pair')
