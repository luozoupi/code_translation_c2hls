"""
Prompts and constants for Fortran to C++ translation and verification.
"""

# System instruction for the query agent
Instruction_qer = """You are an expert in Fortran and C++ programming. Your task is to help generate test programs, translate code between languages, and debug compilation/runtime issues. Always provide complete, single-file programs that can be compiled and run independently."""

# Phase A: Request Fortran testbench generation
q_generate_fortran_bench_first = """Generate a complete, self-contained Fortran program that tests the provided Fortran source code. The program should:
1. Include all necessary modules and declarations
2. Call the functions/subroutines with test inputs
3. Print clear output showing the test results
4. Be a single file that can be compiled with gfortran
5. Include the source code inline (not as a separate module)

Provide the code in a ```fortran code fence."""

# Phase B: Request C++ translation
q_translate_to_cpp_same_test = """Now translate the validated Fortran program to C++. The C++ program should:
1. Perform exactly the same operations as the Fortran program
2. Produce identical output
3. Be a complete, self-contained single file
4. Use standard C++ (C++11 or later)
5. Include proper headers and namespace declarations

Provide the code in a ```cpp code fence."""

# Modification prompt for Fortran/C++ code
ft_cf_further_modification = """The code has compilation or runtime issues. Here are the results:

{cpp_compile_result}

Please fix the issues and provide the corrected code in appropriate code fences (```fortran or ```cpp)."""

# Prompt for combining header files (Fortran)
combine_header_files_fortran = """The Fortran code has missing file dependencies or undefined references:

{compile_result}

Please provide a single, self-contained Fortran file that includes all necessary code inline without external dependencies. Use ```fortran code fence."""

# Prompt for combining header files (C++)
combine_header_files_cpp = """The C++ code has missing header files or undefined references:

{compile_result}

Please provide a single, self-contained C++ file that includes all necessary code inline without external dependencies. Use ```cpp code fence."""

# Prompt for missing terminating character
missing_terminating = """The code has a missing terminating quote character. Please fix the string literals and provide the corrected code."""

# Prompt for C++ modification
ff_ct_further_modification = """The C++ code has compilation or runtime issues:

{cuda_compile_result}

Please fix the C++ code and provide it in a ```cpp code fence. Do not modify the Fortran baseline."""

# Initial solver prompt
Init_solver_prompt = """Here are the current code versions:

Fortran code:
```fortran
{cpp_code}
```

C++ code:
```cpp
{cuda_code}
```"""

# Output comparison analysis prompt
output_comparison_analysis = """Compare the outputs of the Fortran and C++ programs below and determine if they are equivalent.

Fortran code:
```fortran
{cpp_code}
```

C++ code:
```cpp
{cuda_code}
```

Fortran output:
{cpp_output}

C++ output:
{cuda_output}

Answer with YES if the outputs are functionally equivalent (ignoring minor formatting differences like whitespace), or NO if they are different. Start your response with YES or NO on the first line, then explain your reasoning."""

# Output mismatch fix prompt
output_mismatch_fix = """The Fortran and C++ programs produce different outputs. Please fix the C++ code to match the Fortran output.

Fortran code:
```fortran
{cpp_code}
```

C++ code:
```cpp
{cuda_code}
```

Fortran output:
{cpp_output}

C++ output:
{cuda_output}

Provide the corrected C++ code in a ```cpp code fence."""

# End prompt
end_prompt_ = """Great! The translation is successful. Please provide the final, cleaned-up version of both the Fortran and C++ code in code fences."""
