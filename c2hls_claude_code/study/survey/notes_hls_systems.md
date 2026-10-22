# Survey notes 1: LLM / agentic HLS systems (research agent, 2026-10-21)

Condensed. [post-window] = after mid-2026. UNVERIFIED as marked by the agent.

## Systems (class: SP single prompt, PIPE fixed pipeline, MA multi-agent fixed roles, AUTO general coding agent, FT, RL)
1. HLS-Repair (MLCAD'24) https://arxiv.org/abs/2407.03889 — PIPE+RAG, GPT-4T, Catapult+cosim; +23.33% vs direct; RAG +13.89%.
2. HLSRewriter (TODAES) — UNVERIFIED.
3. HLSTester https://arxiv.org/abs/2504.14641 — testbench gen (abstract only).
4. C2HLSC (TODAES) https://arxiv.org/abs/2412.00214 — PIPE hierarchical, GPT-4o-mini→4o escalation, Catapult; 10 designs × 10 runs; <$1/run; no formal ablation; area 2–3× manual.
5. HLSPilot https://arxiv.org/abs/2408.06810 — PIPE (profiling, decomposition, strategy retrieval, DSE), GPT-4, U280 board runtime; beats hand-crafted on 3, far worse on 2; DSE on/off only.
6. "Are LLMs any good for HLS?" https://arxiv.org/abs/2408.10428 — study, weak baseline.
7. RALAD https://arxiv.org/abs/2410.07356 — SP+RAG; CodeLlama-7B synthesizable 0%→30% (RAG)→50% (+annotations).
8. Agentic-HLS https://arxiv.org/abs/2412.01604 — MA QoR prediction; agent RMSE 4.21 vs fine-tuned HARP 2.82 (GNN alone better).
9. Gai et al. ASP-DAC'25 https://arxiv.org/abs/2502.13921 — FT Code-Llama-7B: functionality 0→53.2%, CoT 61.45%.
10. HLS-Eval https://arxiv.org/abs/2504.12268 ; agentic HLS-Eval [post-window] https://arxiv.org/abs/2609.09526 — mini-swe-agent saturates (gpt-oss-120b 100% pass@10); general agents suffice for simple tasks.
11. LIFT https://arxiv.org/abs/2504.21187 — FT GNN-supervised; 3.52× vs AutoDSE-24h, 2.16× vs HARP; no numeric ablation.
12. Intelligent4DSE https://arxiv.org/abs/2504.19649 (abstract only).
13. LLM-DSE https://arxiv.org/abs/2505.12188 — MA (router/specialists/arbitrator/critic), GPT-4o, Merlin 8 h; 1.87× vs AutoDSE-8h, 1.04× vs HARP-24h; $11–77/program. Ablation: zero-shot 0.16×; arbitrator only 0.69×; +specialists 0.95×; +router 1.24×; +critic 1.87×. No single-agent-same-tools control.
14. iDSE https://arxiv.org/abs/2505.22086 — PIPE (prune, warm start, evolutionary), DeepSeek-R1; ADRS 16.6× vs NSGA-II; LLM warm start 2.5× vs random.
15. MPM-LLM4DSE (DATE'26) https://arxiv.org/abs/2601.04801 — GNN+LM predictor + LLM optimizer.
16. ChatHLS (ACL'26) https://arxiv.org/abs/2507.00642 — MA+FT; repair 82.7% (+19.1 vs GPT-4o); HLSTuner 5.4×; ablation only for debugging (+8.7 fine-tuned analysis, +1.8 DPO, +8.5 iterative).
17. HLStrans https://arxiv.org/abs/2507.04315 — dataset via LLM+MCTS agent; GPT-5 %OPT 20.5→25.6 with retrieval; no MCTS vs single-shot.
18. ForgeHLS https://arxiv.org/abs/2507.03255 — dataset.
19. TimelyHLS https://arxiv.org/abs/2507.17962 — PIPE+RAG; unoptimized baseline only; no ablation.
20. HLSDebugger (ICCAD'25) https://arxiv.org/abs/2507.21485 — FT.
21. SAGE-HLS (ICCD'25) https://arxiv.org/abs/2508.03558 — FT; AST +1.28 Pass@1.
22. Spec2RTL-Agent https://arxiv.org/abs/2506.13905 — MA, NVIDIA.
23. A2H-MAS https://arxiv.org/abs/2508.10904 — MA running on Claude Code, MATLAB→HLS, board; direct→adaptation→refinement.
24. CorrectHDL https://arxiv.org/abs/2511.16395.
25. Bench4HLS (DATE'26) https://arxiv.org/abs/2601.19941.
26. "From Pragmas to Partners" (Cornell position) https://arxiv.org/abs/2602.01401.
27. LAAFD https://arxiv.org/abs/2602.06085 — MA (translator, fixers, judge, optimizer), GPT-5; 99.9% geomean of hand-tuned; SODA parity; ~$50 total; no ablation; only 1–2/10 SODA runs approved.
28. Agent Factories (IBM) https://arxiv.org/abs/2603.25719 — AUTO Claude Code (Opus 4.5/4.6) + per-sub-kernel agents + ILP; 7.07×/8.27×/6.53× inconsistent; median 5.82M tokens/run; no specialized-system or AutoDSE comparison.
29. IBM QEC agent factory [post-window] https://www.ibm.com/downloads/documents/us-en/17f55bd44d44b0e3 — coordinator+workers+verifier, Opus 5 / GLM-5.3; 204× ($299) / 126× ($105) / 52× ($26) at 100 hypotheses; gains from re-architecture, not pragmas.
30. HLS-Seek https://arxiv.org/abs/2605.13536 — RL GRPO Qwen2.5-Coder-7B with proxy reward.
31. A3D https://arxiv.org/abs/2605.15237 — MA + agentic RAG, Catapult.
32. MailoHLS https://arxiv.org/abs/2606.07246 — FT LLM+GNN.
33. AgRefactor (UCLA) https://arxiv.org/abs/2606.30949 — MA + memory + tools; memory 129→145 passes; hidden testbench: public passes 15/17 → 6/7 (weak oracles inflate ~60%); 6.51× vs AutoDSE.
34. HLS-LeVeri https://arxiv.org/abs/2606.17128 — verification agent.
35. Evidence-Driven C-to-synthesizable-C agent https://arxiv.org/abs/2606.28409 — MA, gpt-5-mini, csim→csyn→cosim; 87.85% all-pass vs C2HLSC 34.58%, HLSTrans 29.91%, ChatHLS 28.97% re-run with same LLM/budget; zero-shot 0–28%; RAG +9.35; gpt-4o-mini 74.8%, DeepSeek v4-flash 91.6% with same scaffold; 18.8K tokens, 140 s/case. Scaffold > model choice.
36. **HLSmith** [post-window] https://arxiv.org/abs/2608.06791 — Luo, Baroughi, Stachura, Chen, Vishwanath, **Z. Fang (SFU)**, Ding; HPCA'27 submission draft. MA (translator, evaluator, QoR diagnoser, optimizer, orchestrator, router) + FT + RL; 42-entry expertise library; Vitis 2023.2 csim/csyn/cosim, U280 300 MHz; HLSFactory-28; ChatHLS baseline (14/14 cosim pass vs 8; 6.91× geomean on common 8); expertise on/off 1.87–2.88× by model; agent loop feasibility 154/154 vs 18/26 zero-shot (Sonnet); flash vs multi-step mixed; open-model pilot: expertise can hurt weak base model (0.50×), fine-tune fixes; abstract 138× vs body 18.77× inconsistency; Agent Factories only in a qualitative table; no general-agent head-to-head.
37. FPGAgent [post-window] https://arxiv.org/abs/2608.23630 — MA training-free + evolutionary search, U200 board; Qwen3-Coder-30B functional 37.2% base → 66.7% full; zero-shot→full: Gemini 56.4→92.3, Sonnet 60.3→84.6; RAG never isolated.
38. HLSFactory-Agent [post-window] https://arxiv.org/abs/2609.09519 — Pi agent for dataset extraction.
39. **AHRR** (UCLA, ICCAD'26) [post-window] https://arxiv.org/abs/2609.21157 — unmodified "pi" coding agent, minimal prompt, ≤5×60 min, hidden testbench, place-and-route, U55C 3.33 ns, resources <60%; agent HLS 2.31× vs its direct RTL; improves AutoSA GEMM 1.2–1.3×, StreamHLS up to 3.3×; weaker model reaches 0.95× of strong with HLS vs 0.40× with RTL; single runs.
40. MicroEvo — UNVERIFIED.
Not HLS: MARCO https://arxiv.org/abs/2505.03906 ; NVIDIA Marco https://arxiv.org/abs/2504.01962

## Structure ablations (key numbers)
LLM-DSE component ladder; ChatHLS debug; FPGAgent (search +, reflection +, full 2×); Evidence-Driven (same-LLM re-runs of prior pipelines 29–35% vs 88%); HLSmith (expertise, loop, flash vs multi-step); AgRefactor (memory, tools, hidden tests); HLS-Repair, RALAD, iDSE, A2H-MAS; Agent Factories agent-count only; IBM QEC model-role swap at matched budget; Agentic-HLS negative (GNN alone better); AHRR abstraction ablation.

## General agents in HLS
Agent Factories, IBM QEC, AHRR, agentic HLS-Eval use general agents successfully; A2H-MAS runs on Claude Code. No paper runs a specialized HLS pipeline vs a general coding agent with same model, tools and budget. AHRR and agentic HLS-Eval suggest naive general agents are already strong on small/medium kernels.

## Weaknesses
csynth-only latency; weak oracles (AgRefactor −60%); weak/self-made baselines (AutoSA only in AHRR, SODA only in LAAFD); tiny benchmarks, unstated sizes; single runs (variance only C2HLSC, Agent Factories, HLSmith); confounded ablations (scaffold vs knowledge vs model); abstract/body inconsistencies (Agent Factories, HLSmith, RALAD, SAGE-HLS, ChatHLS, C2HLSC, HLStrans); sparse cost; resource-blind speedups; clock confounds; model churn.
