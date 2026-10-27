Restructure this C kernel for high-level synthesis.

Preserve the function signature, integer widths, memory layout and observable
results. Keep the supplied interface. Use bounded loops and storage that the
HLS compiler can determine; avoid allocation, recursion and operating-system
calls. Consider memory access, dependencies, reuse and parallel work. Add HLS
directives only where they make sense for this kernel.

Return one complete C++ source file, without explanations or a testbench.
