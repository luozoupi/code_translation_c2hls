from test import AgentOrchestrator

if __name__ == "__main__":

    key=""
    max_completion_tokens=1000
    gpt_model="meta-llama/Llama-3.3-70B-Instruct"
    turns_limitation=3
    idx=0
    orchestrator = AgentOrchestrator(
        max_completion_tokens=max_completion_tokens,
        gpt_model=gpt_model,
        turns_limitation=turns_limitation,
        idx=idx)


    orchestrator.run_phase_c()

    print(orchestrator.history)