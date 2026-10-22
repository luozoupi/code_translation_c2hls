#!/usr/bin/env python3
"""AutoSA post-generation pass: remove dangling streams from the top-level DATAFLOW region.

Bug (AutoSA rev b61a1b4, LU space_time 2): the last PE is an instance of the generic PE template, so its call in
kernel0 passes output streams (fifo_L_tmp_0_PE_<N>, fifo_L_drain_PE_<N-1>) that no other module reads. For that PE
the writes are dead code, but Vitis checks statically that every stream in a DATAFLOW region has exactly one producer
and one consumer, and stops in pre-synthesis with HLS 200-970.

Fix (generic, nothing AutoSA-specific beyond the "/* fifo */ name" call style AutoSA emits):
  1. In the DATAFLOW top, find hls::stream variables that appear in exactly one module call, bound to an output port
     (`*_out`). Read-only dangling streams are left alone: Vitis accepts them (the dead read folds away).
  2. For each such call, emit a wrapper `<Wrapper>_dfix<k>` that takes the other streams, declares the dangling ones
     as locals (outside any DATAFLOW region) and calls the original wrapper.
  3. In csim only (#ifndef __SYNTHESIS__), the wrapper checks that each local stream is empty afterwards and aborts
     with "AUTOSA_DFIX: dangling stream <name> received data" if not, so a fix can never hide real lost data.
  4. Drop the dangling declarations (and their pragmas) from the top.
A design without dangling streams is left byte-identical. AutoSA itself is not modified.

  fix_dangling_streams.py IN_kernel_kernel.cpp OUT_kernel_kernel.cpp [--only a,b | --only-from vitis.log]
  -> prints a JSON report. The study flow ("AutoSA + dangling-stream repair") runs csynth on the unmodified AutoSA
     output first and applies this pass only when pre-synthesis fails with HLS 200-970, to exactly the named streams.
"""
import json, re, sys
from collections import Counter

DECL = re.compile(r"^\s*/\* PE fifo \*/ hls::stream<([^>]+)> (\w+);|^\s*(?:/\*[^*]*\*/\s*)?hls::stream<([^>]+)> (\w+);")
CALL = re.compile(r"^(\s*)(\w+)(<[^>]*>)?\(\s*$")


def top_region(lines):
    """[start, end) of the kernel0 body that holds #pragma HLS DATAFLOW."""
    start = next(i for i, l in enumerate(lines) if re.match(r"\s*void kernel0\s*\(", l))
    depth, end = 0, None
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0 and "{" in "".join(lines[start:i + 1]):
            end = i + 1
            break
    assert any("HLS DATAFLOW" in l for l in lines[start:end]), "kernel0 has no DATAFLOW"
    return start, end


def parse_calls(lines, s, e):
    calls, i = [], s
    while i < e:
        m = CALL.match(lines[i])
        if m and m.group(2) not in ("for", "if", "while"):
            args, j = [], i + 1
            while not lines[j].strip().startswith(");"):
                a = re.search(r"/\* fifo \*/\s*(\w+)", lines[j])
                args.append(a.group(1) if a else lines[j].strip().rstrip(","))
                j += 1
            calls.append({"first": i, "last": j, "indent": m.group(1), "fn": m.group(2), "tmpl": m.group(3) or "",
                          "args": args})
            i = j + 1
        else:
            i += 1
    return calls


def wrapper_sig(src, fn):
    """(template line or '', parameter list) of the wrapper definition `fn`."""
    m = re.search(r"(template<[^>]*>\s*)?void " + re.escape(fn) + r"\(([^)]*)\)\s*\{", src)
    assert m, f"definition of {fn} not found"
    params = [p.strip() for p in m.group(2).split(",")]
    return (m.group(1) or "").strip(), params


def main(inp, out, only=None):
    src = open(inp).read()
    lines = src.split("\n")
    s, e = top_region(lines)
    decls = {}
    for i in range(s, e):
        m = DECL.match(lines[i])
        if m:
            decls[m.group(2) or m.group(4)] = (i, m.group(1) or m.group(3))
    calls = parse_calls(lines, s, e)
    uses = Counter(a for c in calls for a in c["args"] if a in decls)
    single = {n for n in decls if uses[n] == 1}
    # Vitis accepts a stream that is only read (a dead read is optimized away with the template constant); it rejects
    # one that is written with no reader (HLS 200-970). Localize only dangling streams bound to an output port.
    dangling = []
    for c in calls:
        _, params = wrapper_sig(src, c["fn"])
        for prm, a in zip(params, c["args"]):
            if a in single and re.search(r"&\s*\w+_out\s*$", prm):
                dangling.append(a)
    dangling = sorted(set(dangling))
    if only is not None:   # repair-on-error flow: only the streams Vitis named in HLS 200-970
        missing = sorted(set(only) - set(dangling))
        assert not missing, f"named streams are not dangling outputs here: {missing}"
        dangling = sorted(only)
    report = {"input": inp, "streams": len(decls), "calls": len(calls), "dangling": dangling, "wrappers": []}
    if not dangling:
        open(out, "w").write(src)
        print(json.dumps(report))
        return
    new_defs, drop = [], set()
    for k, c in enumerate([c for c in calls if any(a in dangling for a in c["args"])]):
        tmpl, params = wrapper_sig(src, c["fn"])
        assert len(params) == len(c["args"]), (c["fn"], params, c["args"])
        keep = [(p, a) for p, a in zip(params, c["args"]) if a not in dangling]
        local = [(p, a) for p, a in zip(params, c["args"]) if a in dangling]
        name = f"{c['fn']}_dfix{k}"
        tparam = re.findall(r"int (\w+)", tmpl)
        targs = "<" + ", ".join(tparam) + ">" if tparam else ""
        body = [f"/* AutoSA dangling-stream fix: {', '.join(a for _, a in local)} stay local to this call. */",
                (tmpl + "\n" if tmpl else "") + f"void {name}({', '.join(p for p, _ in keep)}) {{"]
        for p, a in local:
            ty = re.match(r"(hls::stream<[^>]+>)", p).group(1)
            pname = re.search(r"&\s*(\w+)$", p).group(1)
            body.append(f"  {ty} {pname};")
        call_args = [re.search(r"&\s*(\w+)$", p).group(1) for p in params]
        body.append(f"  {c['fn']}{targs}({', '.join(call_args)});")
        body.append("#ifndef __SYNTHESIS__")
        for p, a in local:
            pname = re.search(r"&\s*(\w+)$", p).group(1)
            body.append(f'  if (!{pname}.empty()) {{ printf("AUTOSA_DFIX: dangling stream {a} received data\\n"); exit(1); }}')
        body.append("#endif")
        body.append("}")
        new_defs.append("\n".join(body))
        ind = c["indent"]
        lines[c["first"]] = f"{ind}{name}{c['tmpl']}("
        kept = [l for l in lines[c["first"] + 1:c["last"]] if not any(re.search(rf"\b{a}\b", l) for _, a in local)]
        kept = [l.rstrip(",") + ("," if n < len(kept) - 1 else "") for n, l in enumerate(kept)]
        lines[c["first"] + 1:c["last"]] = kept + [None] * ((c["last"] - c["first"] - 1) - len(kept))
        report["wrappers"].append({"call": c["fn"] + c["tmpl"], "wrapper": name, "local": [a for _, a in local]})
    for n in dangling:
        i = decls[n][0]
        drop.add(i)
        for j in (i + 1, i + 2):   # the STREAM / RESOURCE pragmas AutoSA emits right after the declaration
            if re.search(rf"#pragma HLS \w+ variable={n}\b", lines[j] or ""):
                drop.add(j)
    lines = [l for i, l in enumerate(lines) if i not in drop and l is not None]
    text = "\n".join(lines)
    k0 = re.search(r"\n(extern \"C\" \{\s*\n)?void kernel0\s*\(", text).start()
    text = text[:k0] + "\n" + "\n\n".join(new_defs) + "\n" + text[k0:]
    if "#include <cstdio>" not in text:
        text = "#include <cstdio>\n#include <cstdlib>\n" + text
    open(out, "w").write(text)
    print(json.dumps(report))


if __name__ == "__main__":
    # fix_dangling_streams.py IN OUT [--only s1,s2]   (or --only-from VITIS_LOG: every stream named in HLS 200-970)
    only = None
    if len(sys.argv) > 4 and sys.argv[3] == "--only":
        only = sys.argv[4].split(",")
    elif len(sys.argv) > 4 and sys.argv[3] == "--only-from":
        only = sorted(set(re.findall(r"Internal stream .(\w+). \([^)]*\) failed dataflow checking", open(sys.argv[4], errors="replace").read())))
    main(sys.argv[1], sys.argv[2], only)
