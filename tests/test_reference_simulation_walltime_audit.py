from scripts.audit_reference_simulation_walltime import assess, parse_log, status


def test_nested_phase_scopes_are_separate():
    result = parse_log("INFO: Finished Command csynth_design CPU user time: 5 seconds. Elapsed time: 16.69 seconds.\n"
                       "run: Time (s): cpu = 00:30:00 ; elapsed = 01:54:01 . Memory (MB): peak = 80\n"
                       "INFO: Finished Command cosim_design CPU user time: 6 seconds. Elapsed time: 6930.56 seconds.\n")
    assert result["csynth_design_wall_seconds"] == [16.69]
    assert result["cosim_design_wall_seconds"] == [6930.56]
    assert result["xsim_run_wall_seconds"] == [6841]


def test_hardware_latency_is_not_a_walltime():
    assert parse_log("kernel_runtime_us: 999999\nTime: 500000000 ns") == {
        "cosim_design_wall_seconds": [], "csynth_design_wall_seconds": [],
        "xsim_run_wall_seconds": [], "scopes_overlap_not_additive": True}


def test_cache_failed_timeout_remains_censored_timeout():
    assert status({"status":"failed", "error":"Cosim timed out after 10800s"}) == "timeout"
    assert status({"status":"failed", "error":"Output mismatch"}) == "fail"


def test_same_name_and_phase_time_do_not_imply_exact_reference_match():
    record = {"reference_kind":"unchanged_public_c_parent", "status":"pass",
              "cosim_design_wall_seconds":30, "header_sha256":"a", "testbench_sha256":"b",
              "public_plain_sha256":"c"}
    result = assess(record, {"header_sha256":"a", "public_testbench_sha256":"different", "plain_c_sha256":"c"})
    assert result["exactly_reusable_for_new_public_study"] is False
    assert "mismatch_testbench_sha256" in result["reuse_blockers"]


def test_unknown_hashes_do_not_match():
    record = {"reference_kind":"historical_reference_hls_cache", "status":"pass"}
    result = assess(record, {})
    assert result["target_file_matches"]["source_input_sha256"] is None
    assert "no_separate_completed_cosim_design_wall_measurement" in result["reuse_blockers"]
