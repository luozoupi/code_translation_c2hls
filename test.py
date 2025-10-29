from openai import OpenAI
import os
import json
import re
import logging
import subprocess
import glob
from typing import List, Tuple, Dict, Optional

# Import prompts and constants
try:
    from prompt_f2c_output_comparison import *
except ImportError:
    from utils.prompt_f2c_output_comparison import *

# Constants
DEFAULT_MODEL_ID = "gpt-4"
TIMEOUT_LIMIT = 60  # timeout limit in seconds
start_sample = 0  # Default value, can be overridden

# Parse first-line JSON tags (repair intent tags)
def parse_repair_tags(reply: str) -> list:
    if not reply:
        return []
    first = reply.splitlines()[0].strip()
    try:
        data = json.loads(first)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

def extract_codes_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None
    fortran_code, cpp_code = None, None
    # Find all fenced blocks
    fence_pattern = re.compile(r"```(\w+)?\s*(.*?)```", re.DOTALL)
    for lang, body in fence_pattern.findall(text):
        lang_l = (lang or "").lower()
        if lang_l in ("fortran", "f90", "f95", "f03", "f08"):
            fortran_code = body.strip()
        elif lang_l in ("cpp", "c++", "cc", "cxx"):
            cpp_code = body.strip()
    return fortran_code, cpp_code

def run_fortran_only(fortran_folder, fortran_code_exe, timeout_seconds=TIMEOUT_LIMIT):
    """
    Minimal helper: compile & run ONLY the Fortran program used as golden baseline.
    """
    os.makedirs(fortran_folder, exist_ok=True)
    fortran_file_path = os.path.join(fortran_folder, 'test.f90')
    with open(fortran_file_path, 'w') as file:
        file.write(fortran_code_exe)

    fortran_compile_cmd = f'/usr/bin/gfortran -fopenmp -J {fortran_folder} -o {fortran_folder}/test_fortran {fortran_file_path}'
    fortran_compile_process = subprocess.run(fortran_compile_cmd, shell=True, capture_output=True, timeout=timeout_seconds)
    fortran_stdout = fortran_compile_process.stdout.decode('utf-8', 'replace')
    fortran_stderr = fortran_compile_process.stderr.decode('utf-8', 'replace')
    if fortran_compile_process.returncode != 0:
        return (fortran_stdout, fortran_stderr, False)

    fortran_run_cmd = f'{fortran_folder}/test_fortran'
    try:
        fortran_run_process = subprocess.run(fortran_run_cmd, shell=True, capture_output=True, timeout=timeout_seconds)
        fortran_stdout = fortran_run_process.stdout.decode('utf-8', 'replace')
        fortran_stderr = fortran_run_process.stderr.decode('utf-8', 'replace')
        return (fortran_stdout, fortran_stderr, fortran_run_process.returncode == 0)
    except subprocess.TimeoutExpired as e:
        # Kill the process if it's still running
        if hasattr(e, 'cmd') and e.cmd:
            try:
                import psutil
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['cmdline'] and any('test_fortran' in str(cmd) for cmd in proc.info['cmdline']):
                            proc.kill()
                            logging.warning(f"Killed hanging Fortran process: {proc.info['pid']}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except ImportError:
                logging.warning("psutil not available, cannot kill hanging processes")
        return ("", "It seems that the program hangs! Fortran execution timed out.", False)
    finally:
        # Clean up .mod files
        mod_files = glob.glob(f'{fortran_folder}/*.mod')
        for file in mod_files:
            os.remove(file)

def add_to_json(history, file_path='dialogues.json'):
    """
    Adds the conversation history to a JSON file.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                dialogues = json.load(file)
            except json.JSONDecodeError:
                dialogues = []
    else:
        dialogues = []

    new_id = 1 if not dialogues else dialogues[-1]["id"] + 1

    new_dialogue = {
        "id": new_id,
        "messages": history
    }
    dialogues.append(new_dialogue)
    
    with open(file_path, 'w') as file:
        json.dump(dialogues, file, indent=4)

def run_codes(fortran_folder, f_code_exe, cpp_folder, c_code_exe, timeout_seconds=TIMEOUT_LIMIT):
    """
    Compiles and runs Fortran and C++ code and captures their output.
    """
    fortran_file_path = os.path.join(fortran_folder, 'test.f90')
    with open(fortran_file_path, 'w') as file:
        file.write(f_code_exe)

    cpp_file_path = os.path.join(cpp_folder, 'test.cpp')
    with open(cpp_file_path, 'w') as file:
        file.write(c_code_exe)

    fortran_compile_cmd = f'/usr/bin/gfortran -fopenmp -J {fortran_folder} -o {fortran_folder}/test {fortran_file_path}'
    fortran_compile_process = subprocess.run(fortran_compile_cmd, 
                                                shell=True, 
                                                capture_output=True,
                                                timeout=timeout_seconds)
    fortran_stderr = fortran_compile_process.stderr.decode('utf-8', 'replace')
    fortran_stdout = fortran_compile_process.stdout.decode('utf-8', 'replace')
    if fortran_compile_process.returncode != 0:
        fortran_p_f = False
    else:
        fortran_p_f = True  
        fortran_run_cmd = f'{fortran_folder}/test'
        try:
            fortran_run_process = subprocess.run(fortran_run_cmd, 
                                                    shell=True, 
                                                    capture_output=True,
                                                    timeout=timeout_seconds)
            fortran_stdout = fortran_run_process.stdout.decode('utf-8', 'replace')
            fortran_stderr = fortran_run_process.stderr.decode('utf-8', 'replace')
        except subprocess.TimeoutExpired:
            fortran_p_f = False
            fortran_stdout = ''
            fortran_stderr = 'It seems that the program hangs. Fortran execution timed out.'
        # delete generated .mod files
        mod_files = glob.glob(f'{fortran_folder}/*.mod')
        for file in mod_files:
            os.remove(file)

    cpp_compile_cmd = f'g++ -fopenmp {cpp_file_path} -o {cpp_folder}/test'
    cpp_compile_process = subprocess.run(cpp_compile_cmd, 
                                            shell=True, 
                                            capture_output=True,
                                            timeout=timeout_seconds)
    cpp_stderr = cpp_compile_process.stderr.decode('utf-8', 'replace')
    cpp_stdout = cpp_compile_process.stdout.decode('utf-8', 'replace')
    if cpp_compile_process.returncode != 0:
        cpp_p_f = False
    else:
        cpp_p_f = True  
        cpp_run_cmd = f'{cpp_folder}/test'
        try:
            cpp_run_process = subprocess.run(cpp_run_cmd, 
                                                shell=True, 
                                                capture_output=True,
                                                timeout=timeout_seconds)    
            cpp_stdout = cpp_run_process.stdout.decode('utf-8', 'replace')
            cpp_stderr = cpp_run_process.stderr.decode('utf-8', 'replace')
        except subprocess.TimeoutExpired as e:
            # Kill the process if it's still running
            try:
                import psutil
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['cmdline'] and any('test' in str(cmd) and 'cpp_' in str(cmd) for cmd in proc.info['cmdline']):
                            proc.kill()
                            logging.warning(f"Killed hanging C++ process: {proc.info['pid']}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except ImportError:
                logging.warning("psutil not available, cannot kill hanging processes")
            cpp_p_f = False
            cpp_stdout = ''
            cpp_stderr = 'It seems that the program hangs! C++ execution timed out.'
    return fortran_stdout, fortran_stderr, fortran_p_f, cpp_stdout, cpp_stderr, cpp_p_f

def update_code_from_history(f_code_exe, c_code_exe, history):
    """
    Update Fortran and C++ code from the history of previous interactions.
    """
    Str_Exe = history[-1]["content"]
    Str_Exe = Str_Exe.encode().decode('unicode_escape')
    fortran_code, cpp_code = extract_codes_from_text(Str_Exe)
    if fortran_code:
        f_code_exe = fortran_code
    if cpp_code:
        c_code_exe = cpp_code
    return f_code_exe, c_code_exe



class AgentOrchestrator:
    """
    Two-phase pipeline orchestrator for Fortran to C++ translation and verification.
    """

    def __init__(self, max_completion_tokens, gpt_model=DEFAULT_MODEL_ID, turns_limitation=3, idx=0):
        self.key = os.getenv('OPENAI_API_KEY', None)
        self.max_completion_tokens = max_completion_tokens
        self.gpt_model = gpt_model
        self.turns_limitation = turns_limitation
        self.idx = idx
        self.base_url = os.getenv('OPENAI_BASE_URL', "https://api.openai.com/v1")
        self.client = OpenAI(base_url=self.base_url, api_key=self.key)

        self.qer_messages = []
        self.ser_messages = []
        self.history = []
        self.fortran_baseline = None

    def _fur_modification(self, modification_prompt, max_completion_tokens=4096*2):
        """
        Modifies the code based on the provided prompt and updates the history and messages.
        """
        m_ser = {
            "role": "user",
            "content": modification_prompt
        }
        self.history.append(m_ser)
        self.ser_messages.append(m_ser)

        # Use OpenAI client to call API
        response = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=self.ser_messages,
            max_tokens=max_completion_tokens
        )
        ser_answer = response.choices[0].message.content

        m_ser_gpt = {
            "role": "assistant",
            "content": ser_answer
        }
        self.ser_messages.append(m_ser_gpt)
        m_his = {
            "role": "assistant",
            "content": f"{ser_answer}"
        }
        self.history.append(m_his)

    def _initialize_phase_a(self, fortran_code):
        """Initialize Phase A with system and user prompts."""
        # System prompt
        m_sys = {"role": "system", "content": Instruction_qer}
        self.qer_messages.append(m_sys)
        self.history.append(m_sys)

        # User prompt: request Fortran testbench using provided source
        logging.info("=== [Phase A] Incoming source (fortran_code) ===\n%s\n", fortran_code)
        m_userA = {"role": "user", "content":
                   q_generate_fortran_bench_first +
                   "\n\nHere is the provided Fortran source to wrap and test:\n```fortran\n" + (fortran_code or "") + "\n```"}
        self.qer_messages.append(m_userA)
        self.history.append(m_userA)

    def _generate_initial_fortran_code(self):
        """Generate initial Fortran code from the model."""
        # Ask model
        ansA = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=self.qer_messages,
            max_tokens=self.max_completion_tokens
        )
        ansA = ansA.choices[0].message.content

        self.qer_messages.append({"role": "assistant", "content": f"{ansA}"})
        self.history.append({"role": "assistant", "content": f"{ansA}"})

        tags = parse_repair_tags(ansA)
        if tags:
            logging.info("[Phase A] tags=%s", tags)

        # Extract Fortran code
        fortran_code, _ = extract_codes_from_text(ansA)
        if fortran_code:
            logging.info("=== [Phase A] Initial Fortran code extracted (len=%d) ===", len(fortran_code))
        else:
            # Ask for a clean single-file fortran
            prompt = ft_cf_further_modification.format(cpp_compile_result="Return a SINGLE-FILE ```fortran block only.")
            self._fur_modification(prompt)
            reply = self.history[-1]["content"]
            tags = parse_repair_tags(reply)
            if tags:
                logging.info("[Phase A] tags=%s", tags)
            fortran_code, _ = update_code_from_history(fortran_code or "", "", self.history)
            if not fortran_code:
                self.history.append({"role": "system", "content": f"[Phase A] FAIL: no Fortran fenced block after correction. idx={self.idx}"})
                return None

        return fortran_code

    def _debug_fortran_code(self, fortran_code):
        """Debug loop for Phase A - compile and run Fortran code."""
        fortran_folder = f"../sandbox/fortran_{start_sample}"
        os.makedirs(fortran_folder, exist_ok=True)

        for turn in range(self.turns_limitation):
            out, err, ok = run_fortran_only(fortran_folder, fortran_code, timeout_seconds=TIMEOUT_LIMIT)
            logging.info("=== [Phase A] Debug run (turn=%d) ===\nPass: %s\nStdout:\n%s\nStderr:\n%s\n",
                        turn, ok, out, err)

            if ok and out and out.strip():
                logging.info("=== [Phase A] SUCCESS: Fortran program runs and produces output")
                return fortran_code, True

            # Determine modification prompt based on error
            if "missing terminating \" character" in (err or ""):
                modification_prompt = missing_terminating
            elif ("undefined reference to" in (err or "")) or ("No such file or directory" in (err or "")):
                modification_prompt = combine_header_files_fortran.format(compile_result=f"Fortran Compile Stderr:\n{err}")
            else:
                modification_prompt = ft_cf_further_modification.format(cpp_compile_result=f"Fortran Stdout: {out}\nFortran Stderr: {err}")

            self._fur_modification(modification_prompt)
            reply = self.history[-1]["content"]
            tags = parse_repair_tags(reply)
            if tags:
                logging.info("[Phase A][turn=%d] tags=%s", turn, tags)
            fortran_code, _ = update_code_from_history(fortran_code, "", self.history)

        return fortran_code, False

    def run_phase_a(self, fortran_code):
        """
        Phase A: Fortran testbench generation & debug.
        Returns: (success_bool)
        """
        self._initialize_phase_a(fortran_code)

        fortran_code = self._generate_initial_fortran_code()
        if fortran_code is None:
            return False

        fortran_code, phaseA_pass = self._debug_fortran_code(fortran_code)

        if not phaseA_pass:
            self.history.append({"role": "system", "content": f"[Phase A] FAIL: exceeded turns_limitation={self.turns_limitation} without valid testbench. idx={self.idx}"})
            return False

        self.fortran_baseline = fortran_code
        return True

    def _initialize_phase_b(self):
        """Initialize Phase B with user prompt for C++ translation."""
        m_userB = {"role": "user", "content":
                   q_translate_to_cpp_same_test +
                   "\n\nHere is the validated Fortran program:\n```fortran\n" + self.fortran_baseline + "\n```"}
        self.qer_messages.append(m_userB)
        self.history.append(m_userB)

        # Ask model
        ansB = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=self.qer_messages,
            max_tokens=self.max_completion_tokens
        )
        ansB = ansB.choices[0].message.content

        self.qer_messages.append({"role": "assistant", "content": f"{ansB}"})
        self.history.append({"role": "assistant", "content": f"{ansB}"})

        tags = parse_repair_tags(ansB)
        if tags:
            logging.info("[Phase B][init] tags=%s", tags)

        # Initialize C++ code from reply
        _, cpp_code = extract_codes_from_text(ansB)
        self.ser_messages = [{"role": "user", "content": f"{ansB}"}]

        return cpp_code

    def _compare_outputs(self, fortran_stdout, cpp_stdout, cpp_code):
        """
        Compare Fortran and C++ outputs using programmatic and AI comparison.
        Returns: (success, comparison_method, should_continue, updated_cpp_code)
        """
        if not fortran_stdout or not cpp_stdout:
            return False, "", False, cpp_code

        # 1. Programmatic comparison
        success, comparison_method = programmatic_output_compare(fortran_stdout, cpp_stdout)

        if success:
            logging.info(f"[Phase B] SUCCESS: Programmatic comparison ({comparison_method}) shows equivalent outputs")
            return True, comparison_method, False, cpp_code

        # 2. AI comparison
        logging.info("[Phase B] Programmatic comparison failed, trying AI comparison")

        output_comparison_prompt = output_comparison_analysis.format(
            cpp_code=self.fortran_baseline,
            cuda_code=cpp_code or "",
            cpp_output=fortran_stdout,
            cuda_output=cpp_stdout
        )

        try:
            comparison_result = generate_str_answer_gpt(output_comparison_prompt, max_completion_tokens=512, gpt_model=self.gpt_model)
            logging.info("=== [Phase B] AI Output Comparison ===\n%s", comparison_result)

            # Parse YES/NO from AI reply
            first_line = comparison_result.strip().split('\n')[0].strip().upper()
            if first_line.startswith('YES'):
                logging.info("[Phase B] SUCCESS: AI confirmed outputs are equivalent")
                return True, "ai_comparison", False, cpp_code
            else:
                logging.info("[Phase B] AI determined outputs are different, attempting fix")

                # Ask AI to fix the C++ code
                fix_prompt = output_mismatch_fix.format(
                    cpp_code=self.fortran_baseline,
                    cuda_code=cpp_code or "",
                    cpp_output=fortran_stdout,
                    cuda_output=cpp_stdout
                )
                self._fur_modification(fix_prompt)
                reply = self.history[-1]["content"]
                tags = parse_repair_tags(reply)
                if tags:
                    logging.info("[Phase B] Fix attempt tags=%s", tags)
                _, cpp_code = update_code_from_history(self.fortran_baseline, cpp_code or "", self.history)
                return False, "", True, cpp_code

        except Exception as e:
            logging.error(f"[Phase B] Error in AI comparison: {e}")
            # Fallback to simple string comparison
            if fortran_stdout.strip() == cpp_stdout.strip():
                logging.info("[Phase B] SUCCESS: Simple string comparison shows identical outputs")
                return True, "simple_string", False, cpp_code
            else:
                logging.info("[Phase B] Simple string comparison shows different outputs")
                # Try to fix with general modification
                modification_prompt = ff_ct_further_modification.format(cuda_compile_result=f"C++ Stdout: {cpp_stdout}\nC++ Stderr: {cpp_stderr}") + \
                                      f"\n\nMake the C++ program produce the same output as the Fortran program:\n{fortran_stdout}"
                self._fur_modification(modification_prompt)
                reply = self.history[-1]["content"]
                tags = parse_repair_tags(reply)
                if tags:
                    logging.info("[Phase B] General fix tags=%s", tags)
                _, cpp_code = update_code_from_history(self.fortran_baseline, cpp_code or "", self.history)
                return False, "", True, cpp_code

    def _debug_and_compare_cpp(self, cpp_code):
        """Debug loop for Phase B - compile, run, and compare C++ code with Fortran baseline."""
        fortran_folder = f"../sandbox/fortran_{start_sample}"
        cpp_folder = f"../sandbox/cpp_{start_sample}"
        os.makedirs(fortran_folder, exist_ok=True)
        os.makedirs(cpp_folder, exist_ok=True)

        for turn in range(self.turns_limitation):
            logging.info("[Phase B] Running modification %dth turn", turn)

            # Provide current codes for the unit-test runner
            init_msg = {"role": "assistant", "content": Init_solver_prompt.format(cpp_code=self.fortran_baseline, cuda_code=cpp_code or "")}
            self.ser_messages = self.ser_messages + [init_msg]

            fortran_stdout, fortran_stderr, fortran_ok, cpp_stdout, cpp_stderr, cpp_ok = run_codes(
                fortran_folder, self.fortran_baseline, cpp_folder, cpp_code or ""
            )
            logging.info("=== [Phase B] Compile/Run Summary ===\nFortran pass: %s\nC++ pass: %s\n", fortran_ok, cpp_ok)
            logging.info("Fortran stdout:\n%s\nFortran stderr:\n%s\n", fortran_stdout, fortran_stderr)
            logging.info("C++ stdout:\n%s\nC++ stderr:\n%s\n", cpp_stdout, cpp_stderr)

            # Do not modify Fortran in Phase B
            if not fortran_ok:
                logging.error("[Phase B] Unexpected: Fortran baseline failed. Phase B should NOT modify Fortran.")
                modification_prompt = ff_ct_further_modification.format(cuda_compile_result=f"C++ Stdout: {cpp_stdout}\nC++ Stderr: {cpp_stderr}") + \
                                      f"\n\nNOTE: Do not modify the Fortran program; fix the C++ program to match the validated Fortran baseline."
                self._fur_modification(modification_prompt)
                reply = self.history[-1]["content"]
                tags = parse_repair_tags(reply)
                if tags:
                    logging.info("[Phase B][turn=%d] tags=%s", turn, tags)
                _, cpp_code = update_code_from_history(self.fortran_baseline, cpp_code or "", self.history)
                continue

            if not cpp_ok:
                if ("undefined reference to" in (cpp_stderr or "")) or ("No such file or directory" in (cpp_stderr or "")):
                    modification_prompt = combine_header_files_cpp.format(compile_result=f"C++ Compile Stderr:{cpp_stderr}")
                else:
                    modification_prompt = ff_ct_further_modification.format(cuda_compile_result=f"C++ Stdout: {cpp_stdout}\nC++ Stderr: {cpp_stderr}")
                self._fur_modification(modification_prompt)
                reply = self.history[-1]["content"]
                tags = parse_repair_tags(reply)
                if tags:
                    logging.info("[Phase B][turn=%d] tags=%s", turn, tags)
                _, cpp_code = update_code_from_history(self.fortran_baseline, cpp_code or "", self.history)
                continue

            # Both run ok - compare outputs
            success, comparison_method, should_continue, cpp_code = self._compare_outputs(fortran_stdout, cpp_stdout, cpp_code)

            if should_continue:
                continue

            if success:
                return cpp_code, True

        return cpp_code, False

    def _save_results(self, cpp_code_final):
        """Save the final Fortran and C++ code to files."""
        os.makedirs("F2C-Translator/data/f2c_test", exist_ok=True)
        with open(f"F2C-Translator/data/f2c_test/fortran_change_gemini_llama_4_scout_{start_sample+self.idx}.f90", "w", encoding="utf-8") as ffortran:
            ffortran.write(self.fortran_baseline)
        with open(f"F2C-Translator/data/f2c_test/cpp_change_gemini_llama_4_scout_{start_sample+self.idx}.cpp", "w", encoding="utf-8") as fcpp:
            fcpp.write(cpp_code_final or "")

    def run_phase_b(self):
        """
        Phase B: C++ translation & debug (Fortran frozen).
        Returns: (success_bool)
        """
        cpp_code = self._initialize_phase_b()

        cpp_code, phaseB_pass = self._debug_and_compare_cpp(cpp_code)

        if not phaseB_pass:
            self.history.append({"role": "system", "content": f"[FAIL] idx={self.idx} Phase B not converged within {self.turns_limitation} turns."})
            return False

        # Seal the deal
        end_prompt = end_prompt_
        self._fur_modification(end_prompt)
        _ = self.history[-1]["content"]

        # Update final codes from history (ignore any Fortran change)
        _, cpp_code_final = update_code_from_history(self.fortran_baseline, cpp_code or "", self.history)

        self._save_results(cpp_code_final or cpp_code)

        self.history.append({"role": "system", "content": f"[SUCCESS] idx={self.idx} saved fortran/cpp pair. Phase A & B passed."})
        return True

    def run(self, fortran_code):
        """
        Main orchestration method that runs both Phase A and Phase B.
        Returns: (history, success_bool)
        """
        # Run Phase A
        for i in range(self.turns_limitation):
            if not self.run_phase_a(fortran_code):
                return self.history, False

        for i in range(self.turns_limitation):
            if not self.run_phase_b():
                return self.history, False
                
        return self.history, True


def Ai_chat_with_Ai(key, fortran_code, max_completion_tokens, gpt_model=DEFAULT_MODEL_ID, turns_limitation=3, idx=0):
    """
    Two-phase pipeline with strict logging and latest mismatch policy.
    Returns: (history, success_bool)

    This function now delegates to the AgentOrchestrator class.
    """
    orchestrator = AgentOrchestrator(
        max_completion_tokens=max_completion_tokens,
        gpt_model=gpt_model,
        turns_limitation=turns_limitation,
        idx=idx
    )
    return orchestrator.run(fortran_code)


if __name__ == "__main__":
    fortran_code = """program main
    implicit none
    integer :: i, n, sum
    n = 10
    sum = 0
    do i = 1, n
        sum = sum + i
    end do
    if (sum /= 55) then
        write (*,*) 'Mismatch at trial ', 1
        write (*,*) 'Expected: ', 55
        write (*,*) 'Actual: ', sum
        stop 1
    end if
    write(*,*) "Sum of first 10 natural numbers is: ", sum
end program main
"""
    output = Ai_chat_with_Ai(key="", fortran_code=fortran_code, max_completion_tokens=1000, gpt_model="llama3:latest", turns_limitation=3, idx=0)
    print(output)