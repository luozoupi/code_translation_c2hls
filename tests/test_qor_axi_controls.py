import re

import pytest

from qor_design_space import apply_knob_values, build_ofat_candidates, discover_qor_knobs


def adapters(code, values=(1, 2, 3, 4, 8, 16, 257)):
    return [knob for knob in discover_qor_knobs(code, stream_depth_values=values)
            if knob.scope == "axi_adapter"]


def test_bundle_shared_option_is_one_factor_and_has_public_dependencies():
    code = """void f(float *a, float *b, float *c) {
#pragma HLS INTERFACE m_axi port=a bundle=shared num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=b bundle=shared num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=c bundle=separate num_read_outstanding=16
}
"""
    knobs = adapters(code)
    assert len(knobs) == 2
    knob = next(k for k in knobs if k.adapter_bundle == "shared")
    candidate = build_ofat_candidates(code, [knob], max_candidates=1)[0]
    assert candidate["code"].count("num_read_outstanding=1\n") == 2
    assert "port=c bundle=separate num_read_outstanding=16" in candidate["code"]
    assert len(candidate["changed_knobs"]) == 1
    assert candidate["changed_knobs"][0]["affected_ports"] == ["a", "b"]
    assert knob.public()["dependent_lines"] == [3]
    assert "linked_replacements" not in knob.public()


def test_omitted_option_is_explicitly_linked_and_comments_preserved():
    code = """void f(float *a, float *b) {
#pragma HLS INTERFACE m_axi port=a bundle=x num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=b bundle=x // num_read_outstanding=999
}
"""
    knob, = adapters(code)
    assert knob.current_value is None
    output = apply_knob_values(code, [(knob, 2)])
    assert "port=b bundle=x num_read_outstanding=2 // num_read_outstanding=999" in output
    assert "port=a bundle=x num_read_outstanding=2" in output


def test_bundle_scope_does_not_cross_functions_or_parse_comment_pragmas():
    code = """/* {
#pragma HLS INTERFACE m_axi port=bogus bundle=x num_read_outstanding=16
} */
void f(float *a) {
const char *s = "}";
#pragma HLS INTERFACE m_axi port=a bundle=x num_read_outstanding=16
}
void g(float *b) {
#pragma HLS INTERFACE m_axi port=b bundle=x num_read_outstanding=16
}
"""
    knobs = adapters(code)
    assert len(knobs) == 2
    assert {k.affected_ports for k in knobs} == {("a",), ("b",)}
    output = apply_knob_values(code, [(knobs[0], 2)])
    assert "port=b bundle=x num_read_outstanding=16" in output


def test_implicit_default_bundle_and_symbolic_option_conservatism():
    code = """void f(float *a, float *b) {
#pragma HLS INTERFACE m_axi port=a num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=b num_write_outstanding=16
}
"""
    knob, = adapters(code)
    assert knob.adapter_bundle == "<default>"
    assert apply_knob_values(code, [(knob, 2)]).count("num_write_outstanding=2") == 2
    assert not adapters(code.replace("port=b num_write_outstanding=16", "port=b num_write_outstanding=DEPTH"))


@pytest.mark.parametrize("option", ["max_read_burst_length", "max_write_burst_length"])
def test_burst_domain_excludes_nonpowers_and_out_of_range(option):
    code = f"void f(float *a) {{\n#pragma HLS INTERFACE m_axi port=a {option}=16\n}}"
    knob, = adapters(code, (0, 1, 2, 3, 4, 5, 16, 128, 256, 512))
    assert knob.candidate_values == (1, 2, 4, 128, 256)
    with pytest.raises(ValueError, match="illegal value"):
        apply_knob_values(code, [(knob, 3)])


def test_linked_adapter_interacts_with_width_insertion_without_overlap():
    code = """void f(float *a, float *b) {
#pragma HLS INTERFACE m_axi port=a bundle=x num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=b bundle=x
}
"""
    knobs = discover_qor_knobs(code)
    adapter = next(k for k in knobs if k.scope == "axi_adapter")
    width = next(k for k in knobs if k.name.startswith("max_widen_bitwidth:b"))
    output = apply_knob_values(code, [(adapter, 2), (width, 128)])
    assert output.count("num_read_outstanding=2") == 2
    assert re.search(r"port=b[^\n]*max_widen_bitwidth=128", output)


def test_two_linked_options_can_be_inserted_at_the_same_offset():
    code = """void f(float *a, float *b) {
#pragma HLS INTERFACE m_axi port=a bundle=x num_read_outstanding=16 max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=b bundle=x
}
"""
    knobs = adapters(code)
    output = apply_knob_values(code, [(knob, 2) for knob in knobs])
    assert output.count("num_read_outstanding=2") == 2
    assert output.count("max_read_burst_length=2") == 2
