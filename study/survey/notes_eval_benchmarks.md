# Survey notes 3: benchmarks and evaluation practice (research agent, 2026-10-21)

Unverified items marked UNVERIFIED by the agent. Condensed from its report.

## Benchmarks
- CHStone (2008), MachSuite (2014), PolyBench/C, Rosetta (FPGA'18, https://github.com/cornell-zhang/rosetta), SODA (ICCAD'18).
- HLSyn (NeurIPS'23 D&B; 42 kernels, ~42K designs, Merlin pragma spaces) https://github.com/ZongyueQin/HLSyn
- HLSFactory (2024) https://arxiv.org/abs/2405.00820
- HLS-Eval (ICLAD'25) https://arxiv.org/abs/2504.12268 — 94 designs; parse/compile/csim/synth pass@k (n=5); no QoR.
- HLS-Eval agentic (Sep 2026) https://arxiv.org/abs/2609.09526 — 85 designs; gpt-oss-120b 100% pass@10: saturated.
- ForgeHLS https://arxiv.org/abs/2507.03255 (v3: 846 kernels, >400K designs, some GPT-synthesized).
- HLStrans https://arxiv.org/html/2507.04315v1 (137 programs, >23K variants, testbenches).
- SAGE-HLS dataset https://arxiv.org/abs/2508.03558 (16.7K, from Verilog).
- Bench4HLS (DATE'26) https://arxiv.org/abs/2601.19941 (170 cases, Vitis + Catapult).
- RTL: VerilogEval v2 https://arxiv.org/abs/2408.11053 ; CVDP https://arxiv.org/abs/2506.14074 (783 problems, best codegen pass@1 ≤34%).

## Evaluation practice (pattern)
- Performance almost always from csynth/Merlin report cycles; no checked paper uses board runtime as primary (HLSPilot, FPGAgent possible exceptions).
- Only ChatHLS (https://arxiv.org/html/2507.00642) and LAAFD (https://arxiv.org/html/2602.06085v1) gate correctness on cosim.
- No paper reports post-route timing for main results (TimelyHLS gates on WNS; ChatHLS reports post-impl for 16 workloads).
- Cost in $: only LLM-DSE (https://arxiv.org/html/2505.12188v1, $11–77/kernel) and LAAFD (~$50 total). Agent Factories (https://arxiv.org/html/2603.25719) tokens: median 5.8M/run.
- Variance essentially absent (LLM-DSE 2 runs; Agent Factories 5-run mean no std, inconsistent 7.07× vs 8.27× headline; LAAFD only 1–2 of 10 SODA runs passed).
- No LLM-HLS paper compares against AutoSA; SODA only in LAAFD; ScaleHLS/POLSCA never run as baselines.
- Weak baselines: "Are LLMs any good for HLS" (https://arxiv.org/abs/2408.10428) disables pipelining; MailoHLS 127× over default Vitis; ScaleHLS 768× over unoptimized.
- Other systems with eval rows: HLSPilot https://arxiv.org/html/2408.06810v1 ; C2HLSC https://arxiv.org/abs/2412.00214 ; LIFT https://arxiv.org/html/2504.21187v1 ; TimelyHLS https://arxiv.org/html/2507.17962v1 ; Agentic-HLS 2412.01604 (QoR prediction) ; HLS-Seek https://arxiv.org/html/2605.13536v1 ; FPGAgent https://arxiv.org/abs/2608.23630 ; MailoHLS https://arxiv.org/abs/2606.07246 ; MPM-LLM4DSE https://arxiv.org/abs/2601.04801.

## Pitfalls
1. csynth latency undefined/wrong: LightningSim Table III csynth vs cosim up to −39.6% / +684.8%, "?" for 4 designs https://ar5iv.labs.arxiv.org/html/2304.11219 ; OmniSim: csynth does not model non-blocking FIFOs https://arxiv.org/html/2508.19299v1 ; LAAFD: HLS ignores inter-module DATAFLOW latency.
2. LOOP_TRIPCOUNT is report-only https://docs.amd.com/r/en-US/ug1399-vitis-hls/pragma-HLS-loop_tripcount
3. m_axi: default latency 64 (Vitis kernel flow) https://docs.amd.com/r/en-US/ug1399-vitis-hls/Options-for-Controlling-AXI4-Burst-Behavior ; forum csynth 2.66M vs cosim 6.71M.
4. Synthesizes but does not route: AutoBridge 16/43 HLS designs failed P&R baseline https://www.csl.cornell.edu/~zhiruz/pdfs/autobridge-fpga2021.pdf ; TAPA https://arxiv.org/abs/2209.02663
5. HLS resource estimates inaccurate https://arxiv.org/abs/2205.12397
6. Weak testbenches / gaming: KernelBench-Verified (GPT-5.5 1.43×→0.88× under hidden tests) https://www.alphaxiv.org/abs/2607.16241 ; AI CUDA Engineer exploits, CUDA-L1 32.8% timing loophole https://arxiv.org/pdf/2507.14111 https://arxiv.org/pdf/2510.03760 ; SpecBench reward hacking https://arxiv.org/html/2605.21384v1 ; no HLS-specific reward-hacking audit found.
7. Contamination: VeriContaminated https://arxiv.org/abs/2503.13572 ; no HLS study.
8. Best-of-N reporting (HLStrans, HLS-Seek Best@5).
9. Saturation of pass/fail HLS benchmarks.
10. Fast sims: LightningSim 99.9% avg, exact on 26/33; LightningSimV2 https://arxiv.org/html/2404.09471v2 ; OmniSim (MICRO'25) 0.09% avg error, non-blocking FIFOs https://arxiv.org/abs/2508.19299 ; FIFOAdvisor https://arxiv.org/abs/2510.20981

## Protocol recommendations
Hard correctness gate (strong csim + cosim/validated fast sim, testbench outside agent write access); tiered latency (csynth / cosim or validated sim with real m_axi / board) with csynth-to-cosim ratio; place-and-route (WNS, Fmax, latency × period); strong baselines (AutoSA, SODA, AutoDSE/HARP, hand-tuned) same toolchain; ≥3–5 runs with mean±std / pass@k unbiased; cost (tokens, $, wall, tool calls, CPU-h), budget-matched comparisons; contamination checks / perturbed kernels; full reproducibility artifacts.
