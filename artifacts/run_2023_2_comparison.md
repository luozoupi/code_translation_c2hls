# 2023.2 + xcu280 vs reference JSONL

Vitis 2023.2 / part xcu280-fsvh2892-2L-e / clock 3.33 ns

Reference: same Vitis + xcu280, dev.llm4hls.com 2023_port


| bench | variant | metric | reference | ours | delta% |
|---|---|---|---:|---:|---:|
| nw | baseline | lut | 36684 | 36626 | -0.2% |
| nw | baseline | ff | 17924 | 17921 | -0.0% |
| nw | baseline | bram | 69 | 69 | +0.0% |
| nw | baseline | dsp | 0 | 0 | — |
| nw | tiling | lut | 9768 | 9686 | -0.8% |
| nw | tiling | ff | 8986 | 8983 | -0.0% |
| nw | tiling | bram | 165 | 165 | +0.0% |
| nw | tiling | dsp | 0 | 0 | — |
| nw | pipeline | lut | 361642 | 358914 | -0.8% |
| nw | pipeline | ff | 198961 | 198958 | -0.0% |
| nw | pipeline | bram | 734 | 734 | +0.0% |
| nw | pipeline | dsp | 0 | 0 | — |
| nw | unroll | lut | 168746 | 165634 | -1.8% |
| nw | unroll | ff | 71793 | 71790 | -0.0% |
| nw | unroll | bram | 862 | 862 | +0.0% |
| nw | unroll | dsp | 0 | 0 | — |
| nw | doublebuffer | lut | 373720 | 370972 | -0.7% |
| nw | doublebuffer | ff | 198354 | 198351 | -0.0% |
| nw | doublebuffer | bram | 990 | 990 | +0.0% |
| nw | doublebuffer | dsp | 0 | 0 | — |
| nw | coalescing | lut | 484657 | 480765 | -0.8% |
| nw | coalescing | ff | 214815 | 214812 | -0.0% |
| nw | coalescing | bram | 1480 | 1480 | +0.0% |
| nw | coalescing | dsp | 0 | 0 | — |
| pathfinder | baseline | latency_ns | 7045000 | 7039000 | -0.1% |
| pathfinder | baseline | lut | 6875 | 6837 | -0.6% |
| pathfinder | baseline | ff | 7990 | 7987 | -0.0% |
| pathfinder | baseline | bram | 35 | 35 | +0.0% |
| pathfinder | baseline | dsp | 17 | 17 | +0.0% |
| pathfinder | tiling | latency_ns | 10531000 | 10522000 | -0.1% |
| pathfinder | tiling | lut | 23260 | 23216 | -0.2% |
| pathfinder | tiling | ff | 11605 | 11602 | -0.0% |
| pathfinder | tiling | bram | 94 | 94 | +0.0% |
| pathfinder | tiling | dsp | 0 | 0 | — |
| pathfinder | unroll | latency_ns | 7087000 | 7081000 | -0.1% |
| pathfinder | unroll | lut | 32160 | 32116 | -0.1% |
| pathfinder | unroll | ff | 15376 | 15373 | -0.0% |
| pathfinder | unroll | bram | 94 | 94 | +0.0% |
| pathfinder | unroll | dsp | 0 | 0 | — |
| pathfinder | doublebuffer | latency_ns | 3521000 | 3518000 | -0.1% |
| pathfinder | doublebuffer | lut | 44638 | 44572 | -0.1% |
| pathfinder | doublebuffer | ff | 16791 | 16788 | -0.0% |
| pathfinder | doublebuffer | bram | 158 | 158 | +0.0% |
| pathfinder | doublebuffer | dsp | 0 | 0 | — |
| pathfinder | coalescing | latency_ns | 73626 | 73560 | -0.1% |
| pathfinder | coalescing | lut | 112189 | 112111 | -0.1% |
| pathfinder | coalescing | ff | 39948 | 39945 | -0.0% |
| pathfinder | coalescing | bram | 30 | 30 | +0.0% |
| pathfinder | coalescing | dsp | 0 | 0 | — |
| knn | baseline | latency_ns | 3496000 | 3493000 | -0.1% |
| knn | baseline | lut | 5802 | 5786 | -0.3% |
| knn | baseline | ff | 8012 | 8009 | -0.0% |
| knn | baseline | bram | 30 | 30 | +0.0% |
| knn | baseline | dsp | 14 | 14 | +0.0% |
| knn | tiling | latency_ns | 14253000 | 14240000 | -0.1% |
| knn | tiling | lut | 6118 | 6082 | -0.6% |
| knn | tiling | ff | 7900 | 7897 | -0.0% |
| knn | tiling | bram | 33 | 33 | +0.0% |
| knn | tiling | dsp | 14 | 14 | +0.0% |
| knn | pipeline | latency_ns | 14253000 | 14240000 | -0.1% |
| knn | pipeline | lut | 6118 | 6082 | -0.6% |
| knn | pipeline | ff | 7900 | 7897 | -0.0% |
| knn | pipeline | bram | 33 | 33 | +0.0% |
| knn | pipeline | dsp | 14 | 14 | +0.0% |
| knn | unroll | latency_ns | 13482000 | 13469000 | -0.1% |
| knn | unroll | lut | 13203 | 13151 | -0.4% |
| knn | unroll | ff | 43839 | 43836 | -0.0% |
| knn | unroll | bram | 31 | 31 | +0.0% |
| knn | unroll | dsp | 28 | 28 | +0.0% |
| knn | doublebuffer | latency_ns | 5801000 | 5796000 | -0.1% |
| knn | doublebuffer | lut | 27758 | 27700 | -0.2% |
| knn | doublebuffer | ff | 76651 | 76648 | -0.0% |
| knn | doublebuffer | bram | 32 | 32 | +0.0% |
| knn | doublebuffer | dsp | 28 | 28 | +0.0% |
| knn | coalescing | latency_ns | 875000 | 874000 | -0.1% |
| knn | coalescing | lut | 23346 | 23288 | -0.2% |
| knn | coalescing | ff | 101850 | 101847 | -0.0% |
| knn | coalescing | bram | 30 | 30 | +0.0% |
| knn | coalescing | dsp | 224 | 224 | +0.0% |
