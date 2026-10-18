"""C++ interface and public-workload contracts, extracted without semantic changes."""
from __future__ import annotations
import re
import json
import hashlib
from typing import Any, List, Optional

def extract_cpp_code(text: str) -> Optional[str]:
    """Extract C/C++ code from the last fenced block in an LLM response."""
    if not text:
        return None
    fence_pattern = re.compile(r"```(?:cpp|c\+\+|c|hls)?\s*(.*?)```", re.DOTALL)
    matches = fence_pattern.findall(text)
    if matches:
        return matches[-1].strip()
    return None

def _normalize_signature_text(text: str) -> str:
    # Strip C/C++ comments that appear inline in parameter lists
    # (e.g. `float *feature /*[N][F]*/`). The comparison should be on types
    # and names only, not on documentation.
    stripped = re.sub(r"/\*.*?\*/", " ", text or "", flags=re.DOTALL)
    stripped = re.sub(r"//[^\n]*", " ", stripped)
    normalized = " ".join(stripped.strip().split())
    normalized = re.sub(r"\s*([(),\[\]])\s*", r"\1", normalized)
    normalized = normalized.replace(",", ", ")
    normalized = re.sub(r"\s*\*\s*", "*", normalized)
    normalized = re.sub(r"\s*&\s*", "&", normalized)
    return normalized.strip()

def _extract_function_signature(code: str, function_name: str, definitions_only: bool = False) -> Optional[dict]:
    trailer_pattern = r"\{" if definitions_only else r"[;{]"
    pattern = re.compile(
        rf'(^|\n)(?P<indent>\s*)(?P<extern>extern\s*"C"\s+)?'
        rf'(?P<ret>[A-Za-z_][\w:\s\*&<>\[\]]*?)\b(?P<name>{re.escape(function_name)})\s*'
        rf'\((?P<params>.*?)\)\s*(?P<trailer>{trailer_pattern})',
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(code or "")
    if not match:
        return None
    signature_start = match.start("extern") if match.group("extern") else match.start("ret")
    signature_end = match.start("trailer")
    return {
        "name": match.group("name"),
        "return_type": match.group("ret").strip(),
        "params": match.group("params").strip(),
        "extern_c": bool(match.group("extern")),
        "signature_start": signature_start,
        "signature_end": signature_end,
    }

def _canonical_function_signature(signature: Optional[dict]) -> str:
    if not signature:
        return ""
    return_type = _normalize_signature_text(signature.get("return_type", ""))
    params = _normalize_signature_text(signature.get("params", ""))
    return f"{return_type} {signature.get('name', '')}({params})".strip()

def _render_function_signature(signature: dict, include_extern: bool = False) -> str:
    prefix = 'extern "C" ' if include_extern else ""
    return (
        f"{prefix}{_normalize_signature_text(signature.get('return_type', ''))} "
        f"{signature.get('name', '')}({_normalize_signature_text(signature.get('params', ''))})"
    ).strip()

def _expected_top_signature(header_code: str, testbench_code: str, function_name: str) -> Optional[dict]:
    for source, label in ((testbench_code, "testbench"), (header_code, "header")):
        signature = _extract_function_signature(source, function_name, definitions_only=False)
        if signature:
            signature["source"] = label
            return signature
    return None

def _top_signature_mismatch_reason(code: str, header_code: str, testbench_code: str,
                                   function_name: str) -> str:
    expected = _expected_top_signature(header_code, testbench_code, function_name)
    current = _extract_function_signature(code, function_name, definitions_only=True)
    if not expected or not current:
        return ""

    expected_linkage = bool(expected.get("extern_c"))
    # `extern "C"` may appear with arbitrary whitespace (`extern"C"`,
    # `extern  "C"`) and the surrounding `extern "C" { ... }` block doesn't
    # land directly on the function — match either form via regex rather than
    # a literal substring.
    current_linkage = bool(
        current.get("extern_c") or
        re.search(r'extern\s*"C"', code or "")
    )
    same_signature = _canonical_function_signature(current) == _canonical_function_signature(expected)
    same_linkage = (not expected_linkage) or current_linkage
    if same_signature and same_linkage:
        return ""

    expected_text = _render_function_signature(expected, include_extern=expected_linkage)
    current_text = _render_function_signature(current, include_extern=current_linkage)
    return (
        f"`{function_name}` uses `{current_text}`, but the benchmark testbench expects "
        f"`{expected_text}`"
    )

def _align_generated_top_signature(code: str, header_code: str, testbench_code: str,
                                   function_name: str) -> tuple[str, str]:
    expected = _expected_top_signature(header_code, testbench_code, function_name)
    current = _extract_function_signature(code, function_name, definitions_only=True)
    if not expected or not current:
        return code, ""

    notes = []
    updated = code
    has_extern_linkage = current.get("extern_c") or ('extern "C"' in (code or ""))
    needs_extern = bool(expected.get("extern_c") and not has_extern_linkage)
    if _canonical_function_signature(current) != _canonical_function_signature(expected) or needs_extern:
        replacement = _render_function_signature(expected, include_extern=needs_extern)
        updated = (
            code[:current["signature_start"]]
            + replacement
            + " "
            + code[current["signature_end"]:]
        )
        if _canonical_function_signature(current) != _canonical_function_signature(expected):
            notes.append(f"normalized `{function_name}` signature to match the {expected.get('source', 'expected')} declaration")
        if needs_extern:
            notes.append(f'added `extern "C"` linkage to `{function_name}`')

    return updated, "; ".join(notes)

def _extract_struct_names(header_code: str) -> List[str]:
    return sorted(set(re.findall(r"\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)", header_code or "")))

def _extract_prototype_names(header_code: str) -> List[str]:
    if not header_code:
        return []
    pattern = re.compile(r"^\s*(?:[A-Za-z_][\w:\s\*<>]*?)\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*;", re.MULTILINE)
    names = []
    for name in pattern.findall(header_code):
        if name not in {"if", "for", "while", "switch", "return"}:
            names.append(name)
    return sorted(set(names))

def _extract_defined_function_names(code: str) -> List[str]:
    if not code:
        return []
    pattern = re.compile(r"^\s*(?:[A-Za-z_][\w:\s\*<>\[\]]*?)\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{", re.MULTILINE)
    names = []
    for name in pattern.findall(code):
        if name not in {"if", "for", "while", "switch"}:
            names.append(name)
    return sorted(set(names))

def _extract_interface_ports(hls_code: str) -> List[str]:
    if not hls_code:
        return []
    ports = re.findall(r"#pragma\s+HLS\s+INTERFACE\s+[^\n]*?\bport\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", hls_code)
    return sorted(dict.fromkeys(ports))

def _normalize_vitis_s_axilite_bundles(hls_code: str) -> tuple[str, str]:
    """Ensure Vitis kernel-control pragmas are explicit and single-bundle.

    Vitis 2023.2 treats `m_axi ... offset=slave` ports as requiring a matching
    AXI-lite control entry. If the generated code only adds `s_axilite` for
    scalar arguments and return, synthesis can fail with HLS 214-219 because
    the pointer/array offsets are assigned a different bundle. Normalize this
    before synthesis so external datasets with native array signatures do not
    depend on the model remembering every control pragma.
    """
    if not hls_code:
        return hls_code, ""

    lines = hls_code.splitlines()
    m_axi_ports: list[str] = []
    s_axilite_ports: set[str] = set()
    updated_lines: list[str] = []
    changed_bundle = False

    for line in lines:
        m_axi = re.search(r"#pragma\s+HLS\s+INTERFACE\s+m_axi\b[^\n]*?\bport\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", line)
        if m_axi:
            port = m_axi.group(1)
            if port not in m_axi_ports:
                m_axi_ports.append(port)

        s_axilite = re.search(r"#pragma\s+HLS\s+INTERFACE\s+s_axilite\b[^\n]*?\bport\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", line)
        if s_axilite:
            s_axilite_ports.add(s_axilite.group(1))
            if re.search(r"\bbundle\s*=", line):
                normalized = re.sub(r"\bbundle\s*=\s*[A-Za-z_][A-Za-z0-9_]*", "bundle=control", line)
            else:
                normalized = line.rstrip() + " bundle=control"
            changed_bundle = changed_bundle or normalized != line
            line = normalized
        updated_lines.append(line)

    missing_ports = [port for port in m_axi_ports if port not in s_axilite_ports]
    if not missing_ports and not changed_bundle:
        return hls_code, ""

    final_lines: list[str] = []
    inserted: set[str] = set()
    for line in updated_lines:
        final_lines.append(line)
        m_axi = re.search(r"#pragma\s+HLS\s+INTERFACE\s+m_axi\b[^\n]*?\bport\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", line)
        if m_axi:
            port = m_axi.group(1)
            if port in missing_ports and port not in inserted:
                indent = re.match(r"^(\s*)", line).group(1)
                final_lines.append(f"{indent}#pragma HLS INTERFACE s_axilite port={port} bundle=control")
                inserted.add(port)

    notes = []
    if inserted:
        notes.append("added missing s_axilite control pragmas for m_axi ports: " + ", ".join(sorted(inserted)))
    if changed_bundle:
        notes.append("normalized all s_axilite pragmas to bundle=control")
    return "\n".join(final_lines) + ("\n" if hls_code.endswith("\n") else ""), "; ".join(notes)

def _strip_contract_comments_and_literals(text: str) -> str:
    """Remove syntax that must not seed the public-contract identifier set."""
    text = re.sub(r"/\*.*?\*/", " ", text or "", flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", " ", text)
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    text = re.sub(r"'(?:\\.|[^'\\])*'", "''", text)
    return text

def _normalize_contract_tokens(text: str) -> str:
    return re.sub(r"\s+", "", _strip_contract_comments_and_literals(text))

def _header_contract_definitions(header_code: str) -> dict[str, dict]:
    """Extract macro/constant definitions relevant to workload semantics.

    The paper contract deliberately ignores include paths and private helper
    declarations.  Public identifiers are selected separately from the plain
    input and golden testbench, then followed transitively through this map.
    """
    clean = _strip_contract_comments_and_literals(header_code)
    clean = re.sub(r"\\\n", "", clean)
    definitions: dict[str, dict] = {}
    macro_pattern = re.compile(
        r"^[ \t]*#[ \t]*define[ \t]+"
        r"(?P<name>[A-Za-z_]\w*)"
        r"(?P<params>\([^\n)]*\))?"
        r"[ \t]*(?P<body>[^\n]*)$",
        re.MULTILINE,
    )
    for match in macro_pattern.finditer(clean):
        name = match.group("name")
        params = match.group("params") or ""
        body = match.group("body") or ""
        definitions[name] = {
            "kind": "function_macro" if params else "macro",
            "normalized": _normalize_contract_tokens(params + body),
            "dependencies": sorted(
                set(re.findall(r"\b[A-Za-z_]\w*\b", body)) - {name}
            ),
        }

    constant_pattern = re.compile(
        r"\b(?:static\s+)?(?:constexpr\s+|const\s+)"
        r"(?P<type>[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)?"
        r"(?:\s*<[^;=]+?>)?)\s+"
        r"(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>[^;]+);",
        re.MULTILINE,
    )
    for match in constant_pattern.finditer(clean):
        name = match.group("name")
        value = match.group("value")
        definitions[name] = {
            "kind": "constant",
            "normalized": _normalize_contract_tokens(
                match.group("type") + "=" + value
            ),
            "dependencies": sorted(
                set(re.findall(r"\b[A-Za-z_]\w*\b", value)) - {name}
            ),
        }
    return definitions

def _public_header_contract_audit(
    canonical_header: str,
    candidate_header: str,
    *,
    plain_code: str,
    testbench_code: str,
) -> dict:
    """Compare the transitive public macro/constant workload contract."""
    canonical_defs = _header_contract_definitions(canonical_header)
    candidate_defs = _header_contract_definitions(candidate_header)
    public_text = _strip_contract_comments_and_literals(
        (plain_code or "") + "\n" + (testbench_code or "")
    )
    referenced = set(re.findall(r"\b[A-Za-z_]\w*\b", public_text))
    pending = sorted(referenced.intersection(canonical_defs))
    contract_names: set[str] = set()
    while pending:
        name = pending.pop()
        if name in contract_names:
            continue
        contract_names.add(name)
        for dependency in canonical_defs[name].get("dependencies", []):
            if dependency in canonical_defs and dependency not in contract_names:
                pending.append(dependency)

    differences = []
    canonical_contract = {}
    candidate_contract = {}
    for name in sorted(contract_names):
        expected = canonical_defs[name]
        actual = candidate_defs.get(name)
        canonical_contract[name] = {
            "kind": expected["kind"],
            "normalized": expected["normalized"],
        }
        if actual is not None:
            candidate_contract[name] = {
                "kind": actual["kind"],
                "normalized": actual["normalized"],
            }
        if actual is None:
            differences.append({
                "identifier": name,
                "reason_code": "public_definition_missing",
                "expected_sha256": _sha256_text(expected["normalized"]),
                "actual_sha256": "",
            })
        elif (
            actual.get("kind") != expected.get("kind")
            or actual.get("normalized") != expected.get("normalized")
        ):
            differences.append({
                "identifier": name,
                "reason_code": "public_definition_mismatch",
                "expected_sha256": _sha256_text(expected["normalized"]),
                "actual_sha256": _sha256_text(actual.get("normalized", "")),
            })

    return {
        "schema_version": "c2hls.public-workload-contract.v1",
        "passed": not differences,
        "reason_codes": sorted({item["reason_code"] for item in differences}),
        "public_identifiers": sorted(contract_names),
        "canonical_contract_sha256": _sha256_text(canonical_contract),
        "candidate_contract_sha256": _sha256_text(candidate_contract),
        "differences": differences,
    }

def _sha256_text(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
