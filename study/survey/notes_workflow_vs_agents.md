# Survey notes 2: fixed workflows vs autonomous agents (research agent, 2026-10-21)

Condensed; UNVERIFIED as marked by the agent. All 2026 items are preprints.

## SE agents
- Agentless (FSE'25) https://arxiv.org/html/2407.01489v2 : fixed 3-phase pipeline, SWE-bench Lite 32.0% at $0.70/issue (GPT-4o) vs SWE-agent 18.3–23.0% at $1.62–2.53, OpenDevin 26.7% $1.14, AutoCodeRover 19% $0.45, Moatless 26.7% $0.17. Weaker without location hints; dropped from SWE-Bench Pro (multi-file edits) https://arxiv.org/pdf/2509.16941 . Default harness for open-model reports (DeepSeek R1 ~49.2% Verified, partly verified). Kimi-Dev: Agentless training as skill prior for agents https://arxiv.org/html/2509.23045v1
- SWE-agent ACI (NeurIPS'24) https://arxiv.org/abs/2405.15793 : custom interface +10.7 points over shell-only.
- Anthropic "Building effective agents" https://www.anthropic.com/engineering/building-effective-agents : workflows = predefined code paths, predictable for well-defined tasks; agents for flexibility; simplest first; tools/ACI matter most.
- Anthropic SWE-bench scaffold https://www.anthropic.com/engineering/swe-bench-sonnet : minimal (bash + edit), 49%; "performance can vary significantly based on scaffolding".
- Anthropic multi-agent research https://www.anthropic.com/engineering/multi-agent-research-system : +90.2% on research eval, ~15x tokens; token use explains 80% of variance; multi-agent "not a good fit" for coding.
- Cognition "Don't build multi-agents" (single-threaded default); follow-up https://cognition.com/blog/multi-agents-working (UNVERIFIED fetch).
- Kapoor et al. "AI Agents That Matter" (TMLR'25) https://arxiv.org/pdf/2407.01502 : simple retry/escalation baselines Pareto-dominate LATS/Reflexion/LDB (up to 50x cost); 5 runs; some published results irreproducible.
- HAL https://arxiv.org/abs/2510.11977 : 21,730 rollouts, $40k; more reasoning often lowered accuracy; agents searched for answers.
- MAST "Why do multi-agent LLM systems fail?" https://arxiv.org/html/2503.13657v3 : 14 failure modes, ChatDev 25–33% on ProgramDev.
- Wang et al. ACL'24 https://arxiv.org/abs/2402.18272 : single strong prompt ≈ best multi-agent discussion.
- Gao et al. 2025 https://arxiv.org/abs/2505.18286 : MAS advantage shrinks with stronger models. Tran & Kiela 2026 https://arxiv.org/html/2604.02460v1 : compute-normalized single agents match/beat MAS.
- Google/DeepMind/MIT scaling agent systems https://arxiv.org/abs/2512.08296 : coordination helps parallel tasks (+80.9%), hurts sequential (−39 to −70%); diminishing above ~45% single-agent score.
- Minimal harness vs MLE harnesses (Sep 2026) https://arxiv.org/html/2609.40303 : one long OpenCode session ≥ 4 harnesses on every backbone (GLM-5.2 62.5% vs 47.1% medals); structure led only with weakest backbone (Gemma 4 31B, overlapping CIs); minimal harness cost 6.2x more ($12.12 vs $1.95).
- BenchAgent (UNVERIFIED) https://awesomepapers.io/papers/2606.05670 : ≤1 of 6 MAS beat matched single agent.
- mini-SWE-agent https://github.com/SWE-agent/mini-swe-agent ; bash-only leaderboard https://www.swebench.com/bash-only.html ; >74% Verified; saturation claims (UNVERIFIED).
- Scaffold effects: Epoch up to 11–15 points (UNVERIFIED) https://epoch.ai/benchmarks/swe-bench-verified ; Liu et al. "Coding agents have converged" https://arxiv.org/abs/2609.17394 : within-model scaffold range up to 29.8 points; scaffold×model interactions significant.
- Bitter-lesson opinion pieces (not controlled).
- Field: Answer.AI Devin 3/20 https://www.answer.ai/posts/2025-01-08-devin ; METR RCT devs 19% slower https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/
- Variance: Bjarnason et al. ICLR'26 https://arxiv.org/abs/2602.07150 single-run pass@1 swings 2.2–6.0 points; τ-bench pass^k https://arxiv.org/pdf/2406.12045 ; Agentic Benchmark Checklist https://arxiv.org/html/2507.02825v5

## Domain workflows
- AlphaCodium https://arxiv.org/abs/2401.08500 : GPT-4 19% → 44% pass@5 (validation).
- KernelBench https://arxiv.org/abs/2502.10517 ; METR kernel eval https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering
- Sakana AI CUDA Engineer reward hacking https://techcrunch.com/2025/02/21/sakana-walks-back-claims-that-its-ai-can-dramatically-speed-up-model-training/ ; replication https://arxiv.org/pdf/2510.03760 ; KernelBench-Verified https://www.alphaxiv.org/abs/2607.16241

## Hardware
- MAGE https://arxiv.org/html/2412.07822v1 : same model (Claude 3.5 Sonnet), VerilogEval-V2 pass@1: vanilla 72.4%, single agent 83.9%, multi-agent 93.6% — the cleanest structure ablation. No cost in paper.
- VerilogCoder https://arxiv.org/abs/2408.08927 94.2% (needs testbench/waveform access).
- RTLFixer https://arxiv.org/html/2311.16543v3 98.5% syntax fixes.
- AutoChip https://arxiv.org/html/2311.04887v2 ; follow-up https://arxiv.org/abs/2411.11856 : tool feedback +5.8% at −34.2% cost.
- ChipAgents vendor claim 97.4% (UNVERIFIED).
- CVDP https://arxiv.org/abs/2506.14074 : best ≤34% pass@1 codegen.
- NYU "Agentic frontier of Verilog" https://arxiv.org/html/2603.19347v3 : naive agentic wrapping HURT all models on CVDP (Gemini 3.1 Pro 58.6→42.4; Opus 4.6 50.7→43.5; Codex-5.3 49.7→45.7; Kimi K2.5 47.4→21.7); tuning recovered partly; open models crash more; simulator use correlates with success.
- ChipCraftBrain https://arxiv.org/html/2604.19856v1 : validation-first pipeline; 7B model 36.5% alone → 75.0% in pipeline; $0.01–0.03/problem; ChipBench hierarchical 50% vs 0% Opus alone; overall 33.3% vs Opus 30.7% vs MAGE 37.4%.
- HDLFORGE https://arxiv.org/abs/2603.04646 (no general-agent baseline).
- Agent Factories (IBM) https://arxiv.org/html/2603.25719v2 : Claude Code (Opus 4.5/4.6) on Vitis HLS, 12 kernels, 5 runs; 1→8 agents 4.3x→7.07x mean speedup (10 agents 6.53x); median 5.82M tokens/run; itself wrapped in domain scaffolding (decomposition, ILP selection); no AutoDSE/HARP/LLM-DSE comparison.
- LLM-DSE https://arxiv.org/abs/2505.12188 (2.55x over AutoDSE-8h; versions inconsistent); iDSE https://arxiv.org/abs/2505.22086 ; LIFT https://arxiv.org/pdf/2504.21187
- Surveys: Agentic EDA handoff (115 systems) https://arxiv.org/abs/2606.19795 ; Pan et al. https://arxiv.org/abs/2501.09655 ; He et al. FnT EDA https://nowpublishers.com/article/Details/EDA-063-2 ; Verilog review https://arxiv.org/pdf/2512.00020
- Gap: no peer-reviewed head-to-head of unmodified Claude Code/Codex vs MAGE/VerilogCoder/ChipAgents at matched cost.

## Synthesis (agent's)
Pipelines/scaffolds help: weak/open models (largest effect), cost (but measure — minimal long sessions can cost more), verification loops (least likely absorbed by better models; weak verifiers get reward-hacked), reproducibility/variance, domain decomposition & reading tool output, large-scale DSE needing search, parallel breadth tasks. General agents suffice: frontier models on well-tooled sequential coding; role-play MAS fails; environment/tools > orchestration. Scaffold and model effects similar in size and interact → claims must be model-, cost- and benchmark-specific.

## Reviewer standards
Cost-matched Pareto reporting vs simple baselines; strong general-agent baseline with same model + merged-roles single-agent ablation + strong non-agentic prompt; ≥5 runs with CIs, pass^k; budget reporting; benchmark hygiene (contamination, hardened verifier); release logs/trajectories; model × scaffold sweeps incl. a weak/open model.
