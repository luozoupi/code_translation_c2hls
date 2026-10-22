#!/usr/bin/env python3
"""Sizes match LAAFD (arXiv 2602.06085) since 2026-10-19: 2D 1024^2, 3D 256^3 (was contrast 512^2, 3D 128^3;
old bench kept in ../bench_soda_v1_20261016sizes).
Generator for /scratch/asa582/workspaces/c2hls_claude_code/study/bench_soda/<kernel>/ (plain, baseline, TB, header, metadata, notes)."""
import hashlib, json, os, re, sys

ROOT = "/scratch/asa582/workspaces/c2hls_claude_code/study/bench_soda"
SODA_SRC = "/scratch/asa582/workspaces/soda_src/soda/tests/src"

CT = {"float": "float", "uint16": "uint16_t", "int16": "int16_t"}


def off(v, d):
    if d == 0:
        return v
    return "%s%+d" % (v, d)


def at(n, dx, dy, dz=None, dim=2):
    if dim == 2:
        return "%s[%s][%s]" % (n, off("y", dy), off("x", dx))
    return "%s[%s][%s][%s]" % (n, off("z", dz), off("y", dy), off("x", dx))


def at3(n, dx, dy, dz):
    return at(n, dx, dy, dz, 3)


def arr(n, T, dim, const=False):
    s = "const " if const else ""
    s += "%s %s" % (T, n)
    if dim == 2:
        return s + "[HEIGHT][WIDTH]"
    return s + "[DEPTH][HEIGHT][WIDTH]"


def dims_decl(dim):
    return "[HEIGHT][WIDTH]" if dim == 2 else "[DEPTH][HEIGHT][WIDTH]"


def loops(dim, lo, hi, body, ind="    "):
    """loops over full range with an interior condition; body(then) / else copy given by caller as (compute, copy)"""
    raise NotImplementedError


def interior_cond(dim, lo, hi):
    """condition string for x,y(,z) in [lo, size-hi)."""
    parts = []
    for v, S in (("x", "WIDTH"), ("y", "HEIGHT"), ("z", "DEPTH")):
        if v == "z" and dim == 2:
            continue
        l = lo
        h = hi
        if l:
            parts.append("%s >= %d" % (v, l))
        parts.append("%s < %s - %d" % (v, S, h) if h else "%s < %s" % (v, S))
    return " && ".join(parts)


def nest(dim, inner, ind="    ", rng=None):
    """full-range loop nest around inner (list of lines)."""
    vs = ["y", "x"] if dim == 2 else ["z", "y", "x"]
    S = {"x": "WIDTH", "y": "HEIGHT", "z": "DEPTH"}
    out = []
    for i, v in enumerate(vs):
        lo, hi = (0, 0) if rng is None else rng
        out.append("%sfor (int %s = %d; %s < %s%s; %s++) {" % (ind + "    " * i, v, lo, v, S[v], (" - %d" % hi) if hi else "", v))
    d = len(vs)
    for l in inner:
        out.append(ind + "    " * d + l)
    for i in reversed(range(d)):
        out.append(ind + "    " * i + "}")
    return out


# ------------------------------------------------------------------ kernel definitions
# Each kernel: dict with dim, sizes, q, in list [(name,type)], out (name,type), valid lo/hi, funcs(TOP,STAGE)-> lines, etc.

K = {}


def single_stage(name, dim, in_list, out, lo, hi, expr_lines, pre=None):
    """body for non-iterated single-stage kernel. expr_lines: list of lines (using x,y,z) that end by assigning to
    OUT_LV (placeholder replaced). border = copy of in_list[copy_idx] cast."""
    pass


def blur():
    T = "uint16_t"
    def code(top, stage):
        L = []
        L.append("    static uint16_t bx[HEIGHT][WIDTH];  /* blur_x, SODA local (uint16) */")
        L += nest(2, ["bx[y][x] = (uint16_t)((input[y][x] + input[y+1][x] + input[y+2][x]) / 3);"], rng=None)[0:0]
        # bx needed for y in [0, H-2), all x
        L += ["    for (int y = 0; y < HEIGHT - 2; y++) {",
              "        for (int x = 0; x < WIDTH; x++) {",
              "            bx[y][x] = (uint16_t)((input[y][x] + input[y+1][x] + input[y+2][x]) / 3);",
              "        }", "    }"]
        L += ["    for (int y = 0; y < HEIGHT; y++) {",
              "        for (int x = 0; x < WIDTH; x++) {",
              "            if (x < WIDTH - 2 && y < HEIGHT - 2)",
              "                output[y][x] = (uint16_t)((bx[y][x] + bx[y][x+1] + bx[y][x+2]) / 3);",
              "            else",
              "                output[y][x] = input[y][x];  /* border: copy of input (undefined in SODA) */",
              "        }", "    }"]
        return L
    return dict(dim=2, W=1024, H=1024, D=None, q=1, ins=[("input", "uint16")], out=("output", "uint16"),
                lo=0, hi=2, code=code, bug=("x < WIDTH - 2 && y < HEIGHT - 2", "x < WIDTH - 3 && y < HEIGHT - 2"),
                desc="blur_x(x,y)=(in[y][x]+in[y+1][x]+in[y+2][x])/3 ; out=(bx[y][x]+bx[y][x+1]+bx[y][x+2])/3, both stored uint16 (min-corner anchored)")


def contrast():
    t = open(SODA_SRC + "/contrast.soda").read()
    taps = [(int(a), int(b), int(c)) for a, b, c in re.findall(r"input\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*\*\s*(-?\d+)", t)]
    assert len(taps) == 197
    wt = [[0] * 17 for _ in range(17)]
    for dx, dy, w in taps:
        assert wt[dy][dx] == 0
        wt[dy][dx] = w
    def code(top, stage):
        L = ["    /* 17x17 weights, wt[dy][dx]; 197 non-zero taps copied from tests/src/contrast.soda; 0 = no tap */",
             "    static const int wt[17][17] = {"]
        for r in wt:
            L.append("        {" + ",".join("%d" % v for v in r) + "},")
        L += ["    };",
              "    for (int y = 0; y < HEIGHT; y++) {",
              "        for (int x = 0; x < WIDTH; x++) {",
              "            if (x < WIDTH - 16 && y < HEIGHT - 16) {",
              "                float acc = 0.0f;",
              "                for (int dy = 0; dy < 17; dy++)",
              "                    for (int dx = 0; dx < 17; dx++)",
              "                        if (wt[dy][dx] != 0) acc += input[y+dy][x+dx] * (float)wt[dy][dx];",
              "                output[y][x] = acc;",
              "            } else",
              "                output[y][x] = input[y][x];  /* border: copy of input (undefined in SODA) */",
              "        }", "    }"]
        return L
    return dict(dim=2, W=1024, H=1024, D=None, q=1, ins=[("input", "float")], out=("output", "float"), lo=0, hi=16, code=code,
                bug=("x < WIDTH - 16 && y < HEIGHT - 16", "x < WIDTH - 17 && y < HEIGHT - 16"),
                desc="out[y][x]=sum_{dy,dx in 0..16} wt[dy][dx]*in[y+dy][x+dx] (197 taps, min-corner anchored)",
                data="float_k16")


def erosion_like(kind):
    isero = kind == "erosion"
    def code(top, stage):
        L = ["    static int16_t tmp[HEIGHT][WIDTH];  /* SODA local int16 */"]
        if isero:
            L += ["    for (int y = 9; y < HEIGHT - 9; y++) {",
                  "        for (int x = 0; x < WIDTH; x++) {",
                  "            int16_t m = input[y-9][x];",
                  "            for (int k = -9; k <= 9; k++) if (input[y+k][x] < m) m = input[y+k][x];",
                  "            tmp[y][x] = m;",
                  "        }", "    }"]
            fin = ["                int16_t m = tmp[y][x-9];",
                   "                for (int k = -9; k <= 9; k++) if (tmp[y][x+k] < m) m = tmp[y][x+k];",
                   "                output[y][x] = m;"]
        else:
            L += ["    for (int y = 9; y < HEIGHT - 9; y++) {",
                  "        for (int x = 0; x < WIDTH; x++) {",
                  "            int s = 0;",
                  "            for (int k = -9; k <= 9; k++) s += input[y+k][x];",
                  "            tmp[y][x] = (int16_t)s;  /* stored to int16: wraps (SODA local int16) */",
                  "        }", "    }"]
            fin = ["                int s = 0;",
                   "                for (int k = -9; k <= 9; k++) s += tmp[y][x+k];",
                   "                int16_t t2 = (int16_t)s;  /* tmp2 local int16: wraps */",
                   "                output[y][x] = (int16_t)(((int32_t)t2 - input[y][x]) * input[y][x] / 256);"]
        L += ["    for (int y = 0; y < HEIGHT; y++) {",
              "        for (int x = 0; x < WIDTH; x++) {",
              "            if (x >= 9 && x < WIDTH - 9 && y >= 9 && y < HEIGHT - 9) {"]
        L += fin
        L += ["            } else",
              "                output[y][x] = input[y][x];  /* border: copy of input (undefined in SODA) */",
              "        }", "    }"]
        return L
    d = "tmp[y][x]=min_{k=-9..9} in[y+k][x]; out[y][x]=min_{k=-9..9} tmp[y][x+k] (centred 19x19 min)" if isero else \
        "t1[y][x]=(int16)sum_{k=-9..9} in[y+k][x]; t2[y][x]=(int16)sum_{k=-9..9} t1[y][x+k]; out=(int16)(((int32)t2-in)*in/256) (int promotion, C truncating division)"
    return dict(dim=2, W=1024, H=1024, D=None, q=1, ins=[("input", "int16")], out=("output", "int16"), lo=9, hi=9, code=code,
                bug=("x >= 9 && x < WIDTH - 9 && y >= 9 && y < HEIGHT - 9", "x >= 9 && x < WIDTH - 9 && y >= 9 && y < HEIGHT - 10"), desc=d)


def sobel():
    def code(top, stage):
        I = lambda dx, dy: "img[%s][%s]" % (off("y", dy), off("x", dx))
        L = ["    for (int y = 0; y < HEIGHT; y++) {",
             "        for (int x = 0; x < WIDTH; x++) {",
             "            if (x >= 1 && x < WIDTH - 1 && y >= 1 && y < HEIGHT - 1) {",
             "                /* mag_x: int16 local; mag_y: uint16 local (truncate at the store, as SODA does) */",
             "                int16_t mx = (int16_t)((%s - %s) + (%s - %s) * 3 + (%s - %s));" % (I(1,-1), I(-1,-1), I(1,0), I(-1,0), I(1,1), I(-1,1)),
             "                uint16_t my = (uint16_t)((%s - %s) + (%s - %s) * 3 + (%s - %s));" % (I(-1,1), I(-1,-1), I(0,1), I(0,-1), I(1,1), I(1,-1)),
             "                /* SODA: 65535 - (mx*mx + my*my) in int, stored uint16. Done in uint32 here (no signed overflow UB);",
             "                   the low 16 bits are identical. */",
             "                uint32_t s = (uint32_t)((int32_t)mx * mx) + (uint32_t)my * my;",
             "                mag[y][x] = (uint16_t)(65535u - s);",
             "            } else",
             "                mag[y][x] = (uint16_t)img[y][x];  /* border: copy of input (undefined in SODA) */",
             "        }", "    }"]
        return L
    return dict(dim=2, W=1024, H=1024, D=None, q=1, ins=[("img", "int16")], out=("mag", "uint16"), lo=1, hi=1, code=code,
                bug=("x >= 1 && x < WIDTH - 1 && y >= 1 && y < HEIGHT - 1", "x >= 1 && x < WIDTH - 2 && y >= 1 && y < HEIGHT - 1"),
                desc="mx=(int16)(sobel x), my=(uint16)(sobel y), mag=(uint16)(65535-(mx*mx+my*my))")


def denoise2d():
    def code(top, stage):
        L = ["    static float g[HEIGHT][WIDTH];  /* SODA local g, valid on [1,size-1) */",
             "    for (int y = 1; y < HEIGHT - 1; y++) {",
             "        for (int x = 1; x < WIDTH - 1; x++) {",
             "            float du = u[y][x] - u[y-1][x];",
             "            float dd = u[y][x] - u[y+1][x];",
             "            float dl = u[y][x] - u[y][x-1];",
             "            float dr = u[y][x] - u[y][x+1];",
             "            g[y][x] = 1.0f / sqrtf(1.0f + du * du + dd * dd + dl * dl + dr * dr);",
             "        }", "    }",
             "    for (int y = 0; y < HEIGHT; y++) {",
             "        for (int x = 0; x < WIDTH; x++) {",
             "            if (x >= 2 && x < WIDTH - 2 && y >= 2 && y < HEIGHT - 2) {",
             "                float r0 = u[y][x] * f[y][x] * 4.9f;",
             "                float r1 = (r0 * (2.5f + r0 * (10.2f + r0))) * (4.3f + r0 * (5.4f + r0 * (6.3f + r0)));",
             "                output[y][x] = (u[y][x] + 7.7f *",
             "                    (u[y+1][x] * g[y+1][x] + u[y-1][x] * g[y-1][x] + u[y][x-1] * g[y][x-1] + u[y][x+1] * g[y][x+1] +",
             "                     5.7f * f[y][x] * r1)) * (11.1f + 7.7f *",
             "                    (g[y+1][x] + g[y-1][x] + g[y][x-1] + g[y][x+1] + 5.7f));",
             "            } else",
             "                output[y][x] = u[y][x];  /* border: copy of u (undefined in SODA) */",
             "        }", "    }"]
        return L
    return dict(dim=2, W=1024, H=1024, D=None, q=1, ins=[("f", "float"), ("u", "float")], out=("output", "float"), lo=2, hi=2, code=code,
                bug=("x >= 2 && x < WIDTH - 2 && y >= 2 && y < HEIGHT - 2", "x >= 2 && x < WIDTH - 3 && y >= 2 && y < HEIGHT - 2"),
                desc="TV denoise: g=1/sqrt(1+sum diff^2) on [1,n-1), output uses g at 4 neighbours -> valid [2,n-2)")


def denoise3d():
    def code(top, stage):
        L = ["    static float g[DEPTH][HEIGHT][WIDTH];  /* SODA local g, valid on [1,size-1)^3 */",
             "    for (int z = 1; z < DEPTH - 1; z++) {",
             "        for (int y = 1; y < HEIGHT - 1; y++) {",
             "            for (int x = 1; x < WIDTH - 1; x++) {",
             "                float du = u[z][y][x] - u[z][y-1][x];",
             "                float dd = u[z][y][x] - u[z][y+1][x];",
             "                float dl = u[z][y][x] - u[z][y][x-1];",
             "                float dr = u[z][y][x] - u[z][y][x+1];",
             "                float di = u[z][y][x] - u[z-1][y][x];",
             "                float dO = u[z][y][x] - u[z+1][y][x];",
             "                g[z][y][x] = 1.0f / sqrtf(0.00005f + du * du + dd * dd + dl * dl + dr * dr + di * di + dO * dO);",
             "            }", "        }", "    }",
             "    for (int z = 0; z < DEPTH; z++) {",
             "        for (int y = 0; y < HEIGHT; y++) {",
             "            for (int x = 0; x < WIDTH; x++) {",
             "                if (x >= 2 && x < WIDTH - 2 && y >= 2 && y < HEIGHT - 2 && z >= 2 && z < DEPTH - 2) {",
             "                    float r0 = u[z][y][x] * f[z][y][x] * (1.0f / 0.03f);",
             "                    float r1 = (r0 * (2.38944f + r0 * (0.950037f + r0))) / (4.65314f + r0 * (2.57541f + r0 * (1.48937f + r0)));",
             "                    output[z][y][x] = (u[z][y][x] + 5.0f * (u[z][y][x+1] * g[z][y][x+1] + u[z][y][x-1] * g[z][y][x-1] +",
             "                        u[z][y+1][x] * g[z][y+1][x] + u[z][y-1][x] * g[z][y-1][x] +",
             "                        u[z+1][y][x] * g[z+1][y][x] + u[z-1][y][x] * g[z-1][y][x] + (1.0f / 0.03f) * f[z][y][x] * r1)) /",
             "                        (1.0f + 5.0f * (g[z][y][x+1] + g[z][y][x-1] + g[z][y+1][x] + g[z][y-1][x] + g[z+1][y][x] + g[z-1][y][x] + (1.0f / 0.03f)));",
             "                } else",
             "                    output[z][y][x] = u[z][y][x];  /* border: copy of u (undefined in SODA) */",
             "            }", "        }", "    }"]
        return L
    return dict(dim=3, W=256, H=256, D=256, q=1, ins=[("f", "float"), ("u", "float")], out=("output", "float"), lo=2, hi=2, code=code,
                bug=("z >= 2 && z < DEPTH - 2", "z >= 2 && z < DEPTH - 3"),
                desc="3D TV denoise (6 face neighbours), valid [2,n-2)^3")


def iterative(name, dim, q, expr, ty="float"):
    """expr(src name) -> list of C expression lines (final assignment 'dst[..] = expr;')"""
    def code(top, stage):
        S = "src"
        # stage function is emitted by the caller via 'helpers'; here top body
        L = []
        if q == 1:
            L.append("    %s(%s, %s);" % (stage, "IN", "OUT"))
        return L
    return None


def make_iter(name, dim, q, exprfn, W, H, D):
    inname, outname = {"jacobi2d": ("t1", "t0"), "jacobi3d": ("t1", "t0"), "seidel2d": ("input", "output"),
                       "heat3d": ("in", "out")}[name]
    cond = interior_cond(dim, 1, 1)
    Z = "z" if dim == 3 else None
    def helpers(top, stage):
        sig = "static void %s(const float src%s, float dst%s)" % (stage, dims_decl(dim), dims_decl(dim))
        L = [sig, "{"]
        vs = ["z", "y", "x"] if dim == 3 else ["y", "x"]
        S = {"x": "WIDTH", "y": "HEIGHT", "z": "DEPTH"}
        for i, v in enumerate(vs):
            L.append("    " * (i + 1) + "for (int %s = 0; %s < %s; %s++) {" % (v, v, S[v], v))
        ind = "    " * (len(vs) + 1)
        idx = "".join("[%s]" % v for v in vs)
        L.append(ind + "if (%s)" % cond)
        L.append(ind + "    dst%s = %s;" % (idx, exprfn("src")))
        L.append(ind + "else")
        L.append(ind + "    dst%s = src%s;  /* border: copy of src (undefined in SODA) */" % (idx, idx))
        for i in reversed(range(len(vs))):
            L.append("    " * (i + 1) + "}")
        L.append("}")
        return L
    def code(top, stage):
        dd = dims_decl(dim)
        L = []
        if q > 1:
            L.append("    static float tmp%s;  /* ping-pong buffer for chained iterations */" % dd)
        L.append("    %s(%s, %s);  /* iteration 1 */" % (stage, inname, outname))
        if q > 1:
            L.append("    for (int it = 1; it < ITERATE; it++) {")
            L.append("        %s(%s, tmp);" % (stage, outname))
            vs = ["z", "y", "x"] if dim == 3 else ["y", "x"]
            S = {"x": "WIDTH", "y": "HEIGHT", "z": "DEPTH"}
            for i, v in enumerate(vs):
                L.append("        " + "    " * i + "for (int %s = 0; %s < %s; %s++)" % (v, v, S[v], v))
            idx = "".join("[%s]" % v for v in vs)
            L.append("        " + "    " * len(vs) + "%s%s = tmp%s;" % (outname, idx, idx))
            L.append("    }")
        return L
    return dict(dim=dim, W=W, H=H, D=D, q=q, ins=[(inname, "float")], out=(outname, "float"), lo=q, hi=q, code=code, helpers=helpers,
                bug=("it < ITERATE", "it < ITERATE - 1"), itbug=True)


def jacobi2d():
    ex = lambda s: "(%s + %s + %s + %s + %s) * 0.2f" % (at(s, 0, 1), at(s, 1, 0), at(s, 0, 0), at(s, 0, -1), at(s, -1, 0))
    k = make_iter("jacobi2d", 2, 10, ex, 1024, 1024, None)
    k["desc"] = "t0=(t1(0,1)+t1(1,0)+t1(0,0)+t1(0,-1)+t1(-1,0))*0.2f, 10 chained applications"
    return k


def jacobi3d():
    ex = lambda s: "(%s + %s + %s + %s + %s + %s + %s) * 0.142857142f" % (at3(s,0,0,0), at3(s,1,0,0), at3(s,-1,0,0), at3(s,0,1,0), at3(s,0,-1,0), at3(s,0,0,1), at3(s,0,0,-1))
    k = make_iter("jacobi3d", 3, 6, ex, 256, 256, 256)
    k["desc"] = "t0=(c+x+ +x- +y+ +y- +z+ +z-)*0.142857142f, 6 chained applications"
    return k


def seidel2d():
    terms = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            pass
    order = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
    ex = lambda s: "(" + " + ".join(at(s, dx, dy) for dx, dy in order) + ") * .1111111f"
    k = make_iter("seidel2d", 2, 9, ex, 1024, 1024, None)
    k["desc"] = "3x3 box average * .1111111f (Jacobi-style, double-buffered), 9 chained applications"
    return k


def heat3d():
    ex = lambda s: ".125f * %s + .125f * %s + .125f * %s + .125f * %s + .125f * %s + .125f * %s + .25f * %s" % (
        at3(s,1,0,0), at3(s,-1,0,0), at3(s,0,1,0), at3(s,0,-1,0), at3(s,0,0,1), at3(s,0,0,-1), at3(s,0,0,0))
    k = make_iter("heat3d", 3, 5, ex, 256, 256, 256)
    k["desc"] = "7-point heat stencil (.125f faces, .25f centre), 5 chained applications"
    return k


def renamed(fn, newout):
    def f():
        k = fn()
        old = k["code"]
        k["code"] = lambda top, stage: [re.sub(r"\boutput\b", newout, l) for l in old(top, stage)]
        k["out"] = (newout, k["out"][1])
        k["bug"] = (k["bug"][0], k["bug"][1])
        return k
    return f


KERNELS = {"blur": renamed(blur, "blur_y"), "contrast": contrast, "denoise2d": denoise2d, "denoise3d": denoise3d,
           "erosion": lambda: erosion_like("erosion"), "heat3d": heat3d, "jacobi2d": jacobi2d, "jacobi3d": jacobi3d,
           "seidel2d": seidel2d, "sobel2d": sobel, "xcorr": renamed(lambda: erosion_like("xcorr"), "tmp3")}


# ------------------------------------------------------------------ emitters
def top_sig(k, name):
    dim = k["dim"]
    args = []
    for n, t in k["ins"]:
        args.append("const %s %s%s" % (CT[t], n, dims_decl(dim)))
    n, t = k["out"]
    args.append("%s %s%s" % (CT[t], n, dims_decl(dim)))
    return "void %s(%s)" % (name, ",\n              ".join(args))


def emit_header(name, k):
    g = name.upper() + "_H"
    L = ["#ifndef %s" % g, "#define %s" % g, "", "#include <stdint.h>", "",
         "/* SODA benchmark '%s' (tests/src/%s.soda). Arrays are plain C arrays, row-major; SODA's FIRST index is the" % (name, name),
         "   fastest-varying one, so SODA name(dx,dy[,dz]) is name[y+dy][x+dx] ([z+dz][y+dy][x+dx] in 3D). */",
         "#define WIDTH %d" % k["W"], "#define HEIGHT %d" % k["H"]]
    if k["dim"] == 3:
        L.append("#define DEPTH %d" % k["D"])
    L.append("#define ITERATE %d" % k["q"])
    L.append("")
    for n, t in k["ins"]:
        L.append("typedef %s %s_t;" % (CT[t], n))
    L.append("typedef %s %s_t;" % (CT[k["out"][1]], k["out"][0]))
    L += ["", "#ifdef __cplusplus", 'extern "C" {', "#endif", top_sig(k, "workload") + ";", "#ifdef __cplusplus", "}", "#endif", "", "#endif", ""]
    return "\n".join(L)


def body(k, top, stage, pragmas=False, static_top=False, extern_c=False):
    L = []
    if "helpers" in k:
        L += k["helpers"](top, stage)
        L.append("")
    sig = top_sig(k, top)
    if static_top:
        sig = "static " + sig
    if extern_c:
        L.append('extern "C" {')
    L.append(sig)
    L.append("{")
    if pragmas:
        for n, t in k["ins"]:
            pass
    bodylines = k["code"](top, stage)
    if pragmas:
        names = [n for n, t in k["ins"]] + [k["out"][0]]
        for i, n in enumerate(names):
            L.append("#pragma HLS INTERFACE m_axi port=%s offset=slave bundle=gmem%d" % (n, i))
        for n in names:
            L.append("#pragma HLS INTERFACE s_axilite port=%s bundle=control" % n)
        L.append("#pragma HLS INTERFACE s_axilite port=return bundle=control")
    L += bodylines
    L.append("}")
    if extern_c:
        L.append("}")
    return "\n".join(L) + "\n"


def emit_plain(name, k):
    h = ["/* plain C reference for SODA kernel '%s' (tests/src/%s.soda), %s." % (name, name, k["desc"]),
         " * Layout: SODA name(dx,dy[,dz]) == name[y+dy][x+dx] ([z+dz][y+dy][x+dx]); first SODA index fastest.",
         " * Valid region (SODA defined): x,y%s in [%d, size-%d). Outside it SODA output is garbage; here the output border" % (",z" if k["dim"] == 3 else "", k["lo"], k["hi"]),
         " * is defined as a copy of the (first/main) input and the testbench never checks it." ]
    if k["q"] > 1:
        h.append(" * ITERATE=%d chained applications (SODA unrolls them into one dataflow); every stage writes its border as a copy of its" % k["q"])
        h.append(" * source, so the valid region shrinks by 1 per iteration: [%d, size-%d)." % (k["q"], k["q"]))
    h.append(" * Large static local arrays are used for intermediate/ping-pong buffers. */")
    s = "\n".join(h) + '\n#include <math.h>\n#include <stdint.h>\n#include "%s.h"\n\n' % name
    s += body(k, "workload", "stage", extern_c=True)
    return s


def emit_baseline(name, k):
    h = ["/* Naive Vitis HLS baseline of plain.cpp for SODA kernel '%s': same top/signature, m_axi per array + s_axilite," % name,
         " * no optimisation pragmas. NOTE: large static local buffer(s) (intermediate / ping-pong, up to 64 MB for 256^3 float)",
         " * are fine for csim but will map to BRAM/URAM (or fail to fit) in csynth; a real design must tile or stream. */"]
    s = "\n".join(h) + '\n#include <math.h>\n#include <stdint.h>\n#include "%s.h"\n\n' % name
    s += body(k, "workload", "stage", pragmas=True, extern_c=True)
    return s


def gen_decl(t, mode):
    if t == "float":
        if mode == "float_k16":
            return "(float)(rand() % 256) / 16.0f"
        return "(float)rand() / (float)RAND_MAX"
    return "(%s)(rand() & 0xFFFF)" % CT[t]


def poison(t):
    return "12345.0f" if t == "float" else "(%s)0x5A5A" % CT[t]


def emit_tb(name, k):
    dim = k["dim"]
    ins = k["ins"]
    on, ot = k["out"]
    mode = k.get("data")
    dd = dims_decl(dim)
    vs = ["z", "y", "x"] if dim == 3 else ["y", "x"]
    S = {"x": "WIDTH", "y": "HEIGHT", "z": "DEPTH"}
    idx = "".join("[%s]" % v for v in vs)
    L = []
    L.append("/*")
    L.append(" * Strict Vitis HLS C-simulation testbench for SODA kernel '%s'." % name)
    L.append(" * Independent golden: plain.cpp's workload copied below as workload_ref (identical inputs, srand(42)).")
    L.append(" * Inputs (%s): read-only. Output %s: pure output, poisoned with %s before the call." % (", ".join(n for n, t in ins), on, poison(ot)))
    L.append(" * ONLY the valid interior x,y%s in [%d, size-%d) is compared (the whole of it); the border is undefined in SODA." % (",z" if dim == 3 else "", k["lo"], k["hi"]))
    tol = "float tol 1e-3*fmax(1,|ref|), NaN is an error" if ot == "float" else "integers exact"
    L.append(" * Tolerance: %s." % tol)
    L.append(" */")
    L += ["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>", "#include <math.h>", "#include <stdint.h>",
          '#include "%s.h"' % name, ""]
    L.append("/* ---- reference: copy of plain.cpp (static, renamed) ---- */")
    L.append(body(k, "workload_ref", "stage_ref", static_top=True))
    # globals
    for n, t in ins:
        L.append("static %s %s%s;" % (CT[t], n, dd))
    L.append("static %s %s_dut%s;" % (CT[ot], on, dd))
    L.append("static %s %s_ref%s;" % (CT[ot], on, dd))
    L.append("")
    L.append("int main(void)")
    L.append("{")
    L.append("    srand(42);")
    for n, t in ins:
        L += nest(dim, ["%s%s = %s;" % (n, idx, gen_decl(t, mode))], ind="    ")
    L += nest(dim, ["%s_dut%s = %s;" % (on, idx, poison(ot))], ind="    ")
    L.append("    workload_ref(%s%s_ref);" % ("".join(n + ", " for n, t in ins), on))
    L.append("    workload(%s%s_dut);" % ("".join(n + ", " for n, t in ins), on))
    L.append("    int errors = 0;")
    cond = interior_cond(dim, k["lo"], k["hi"])
    # loops over valid region
    rng_lo, rng_hi = k["lo"], k["hi"]
    for i, v in enumerate(vs):
        L.append("    " + "    " * i + "for (int %s = %d; %s < %s - %d; %s++) {" % (v, rng_lo, v, S[v], rng_hi, v))
    ind = "    " * (len(vs) + 1)
    L.append(ind + "%s d = %s_dut%s;" % (CT[ot], on, idx))
    L.append(ind + "%s r = %s_ref%s;" % (CT[ot], on, idx))
    if ot == "float":
        L.append(ind + "int bad = (d != d) || !(fabs((double)d - (double)r) <= 1e-3 * fmax(1.0, fabs((double)r)));")
    else:
        L.append(ind + "int bad = (d != r);")
    L.append(ind + "if (bad) {")
    pc = "x=%d y=%d" + (" z=%d" if dim == 3 else "")
    pa = "x, y" + (", z" if dim == 3 else "")
    fmtv = "%g" if ot == "float" else "%ld"
    cast = "(double)" if ot == "float" else "(long)"
    L.append(ind + '    if (errors < 10) printf("mismatch at %s: got %s expected %s\\n", %s, %sd, %sr);' % (pc, fmtv, fmtv, pa, cast, cast))
    L.append(ind + "    errors++;")
    L.append(ind + "}")
    for i in reversed(range(len(vs))):
        L.append("    " + "    " * i + "}")
    L.append('    if (errors) { printf("Failed with %d errors!\\n", errors); return 1; }')
    L.append('    printf("Passed!\\n");')
    L.append("    return 0;")
    L.append("}")
    return "\n".join(L) + "\n"


def emit_meta(name, k, plain, base):
    sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
    ty = k["ins"][-1][1]
    valid = {"x": [k["lo"], "WIDTH-%d" % k["hi"]], "y": [k["lo"], "HEIGHT-%d" % k["hi"]]}
    if k["dim"] == 3:
        valid["z"] = [k["lo"], "DEPTH-%d" % k["hi"]]
    sizes = [k["W"], k["H"]] + ([k["D"]] if k["dim"] == 3 else [])
    m = {"benchmark": name, "source_repo": "soda", "algorithm_source_path": "%s/%s.soda" % (SODA_SRC, name),
         "gold_hls_source_path": "", "gold_hls_source_file": "gold_hls_source.cpp", "gold_hls_baseline_file": "hls_baseline.cpp",
         "kernel_file": "plain.cpp", "header_file": name + ".h", "baseline_variant": name + "_0_baseline",
         "variants": [{"name": name + "_0_baseline", "file": "hls_baseline.cpp", "source_path": ""}],
         "variant_source_paths": {}, "plain_c_file": "plain.cpp", "testbench_file": "testbench.cpp",
         "kernel_top": "workload", "hls_top": "workload", "translated_hls_top": "workload", "support_files": [], "include_dirs": [],
         "supports_csim": True, "supports_cosim": False, "cosim_depths": {},
         "strip_report": {"removed_hls_pragmas": 0, "removed_accel_pragmas": 0, "removed_support_includes": 0, "removed_ap_int_includes": 0,
                          "removed_extern_c_blocks": 0, "plain_contains_hls_pragmas": False, "plain_contains_accel_pragmas": False,
                          "plain_contains_ap_uint": False},
         "provenance": {"gold_hls_baseline_sha256": sha(base), "plain_c_sha256": sha(plain), "plain_derived_from_gold_hls": False,
                        "generator": "study/bench_soda/gen_bench.py"},
         "soda": {"iterate": k["q"], "sizes": sizes, "dtype": ",".join("%s:%s" % (n, t) for n, t in k["ins"]) + " -> %s:%s" % k["out"],
                  "valid_region": valid, "layout": "arr[z][y][x], SODA first index = x (fastest)"}}
    return json.dumps(m, indent=2) + "\n"


def emit_notes(name, k):
    dim = k["dim"]
    sz = "x".join(str(v) for v in [k["W"], k["H"]] + ([k["D"]] if dim == 3 else []))
    return """# TB_NOTES: %s (SODA)
- Semantics: %s. Sizes %s, iterate %d. Layout: SODA name(dx,dy[,dz]) == name[y+dy][x+dx][...] (first SODA index fastest).
- Arguments: inputs %s (read-only), output %s (pure output, never read before written; poisoned with %s).
- Golden: plain.cpp's workload copied into testbench.cpp as workload_ref (static), run on identical srand(42) data. Plain.cpp was also cross-checked
  against the SODA-generated kernel (sodac --xocl-kernel run as C++, its own golden passing) on the same random inputs, see report.
- Compared region: valid interior x,y%s in [%d, size-%d) only, every element. Border of the output is undefined in SODA (plain.cpp and the DUT
  may write anything there; plain copies the input); not checked.
- Tolerance: %s. Data: %s.
- Report: first 10 mismatches printed, then exactly `Passed!` (return 0) or `Failed with N errors!` (return 1).
- Controls (g++ -O2): see CONTROLS line below, filled by the validation script.
""" % (name, k["desc"], sz, k["q"], ", ".join(n for n, t in k["ins"]), k["out"][0], poison(k["out"][1]),
       ",z" if dim == 3 else "", k["lo"], k["hi"],
       "float 1e-3*fmax(1,|ref|), NaN is an error" if k["out"][1] == "float" else "integers exact",
       {"float_k16": "floats on a 1/16 grid in [0,16) so every sum/reassociation is exact (contrast sums of 197 weighted taps would otherwise exceed 1e-3 abs by rounding)",
        None: "uniform [0,1) floats" if k["out"][1] == "float" else "full range of the integer type (rand()&0xFFFF)"}[k.get("data")])


def main():
    names = sys.argv[1:] or list(KERNELS)
    for name in names:
        k = KERNELS[name]()
        d = os.path.join(ROOT, name)
        os.makedirs(d, exist_ok=True)
        plain = emit_plain(name, k)
        base = emit_baseline(name, k)
        files = {name + ".h": emit_header(name, k), "plain.cpp": plain, "hls_baseline.cpp": base,
                 "testbench.cpp": emit_tb(name, k), "metadata.json": emit_meta(name, k, plain, base), "TB_NOTES.md": emit_notes(name, k)}
        for fn, c in files.items():
            open(os.path.join(d, fn), "w").write(c)
        os.makedirs("/home/asa582/.claude/jobs/72130906/tmp/bench_soda_check/" + name, exist_ok=True)
        json.dump({"bug": k.get("bug"), "q": k["q"], "dim": k["dim"], "lo": k["lo"], "hi": k["hi"], "ins": k["ins"], "out": k["out"],
                   "sizes": [k["W"], k["H"], k["D"]], "data": k.get("data")},
                  open("/home/asa582/.claude/jobs/72130906/tmp/bench_soda_check/%s/info.json" % name, "w"))
        print("generated", name)


if __name__ == "__main__":
    main()
