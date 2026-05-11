# 2023.2 + xcu280 vs reference JSONL

Vitis 2023.2 / part xcu280-fsvh2892-2L-e / clock 3.33 ns

Reference: same Vitis + xcu280, dev.llm4hls.com 2023_port


| bench | variant | metric | reference | ours | delta% |
|---|---|---|---:|---:|---:|
| nw | baseline | lut | 36684 | 6143 | -83.3% |
| nw | baseline | ff | 17924 | 4099 | -77.1% |
| nw | baseline | bram | 69 | 41 | -40.6% |
| nw | baseline | dsp | 0 | 0 | — |
| nw | tiling | lut | 9768 | 5421 | -44.5% |
| nw | tiling | ff | 8986 | 2830 | -68.5% |
| nw | tiling | bram | 165 | 137 | -17.0% |
| nw | tiling | dsp | 0 | 0 | — |
| nw | pipeline | lut | 361642 | 352858 | -2.4% |
| nw | pipeline | ff | 198961 | 191637 | -3.7% |
| nw | pipeline | bram | 734 | 706 | -3.8% |
| nw | pipeline | dsp | 0 | 0 | — |
| nw | unroll | lut | 168746 | 159578 | -5.4% |
| nw | unroll | ff | 71793 | 64469 | -10.2% |
| nw | unroll | bram | 862 | 834 | -3.2% |
| nw | unroll | dsp | 0 | 0 | — |
| nw | doublebuffer | lut | 373720 | 366734 | -1.9% |
| nw | doublebuffer | ff | 198354 | 192198 | -3.1% |
| nw | doublebuffer | bram | 990 | 962 | -2.8% |
| nw | doublebuffer | dsp | 0 | 0 | — |
| nw | coalescing | lut | 484657 | 480081 | -0.9% |
| nw | coalescing | ff | 214815 | 214430 | -0.2% |
| nw | coalescing | bram | 1480 | 1480 | +0.0% |
| nw | coalescing | dsp | 0 | 0 | — |
| pathfinder | baseline | latency_ns | 7045000 | 7038000 | -0.1% |
| pathfinder | baseline | lut | 6875 | 3253 | -52.7% |
| pathfinder | baseline | ff | 7990 | 2447 | -69.4% |
| pathfinder | baseline | bram | 35 | 7 | -80.0% |
| pathfinder | baseline | dsp | 17 | 17 | +0.0% |
| pathfinder | tiling | latency_ns | 10531000 | 10521000 | -0.1% |
| pathfinder | tiling | lut | 23260 | 19692 | -15.3% |
| pathfinder | tiling | ff | 11605 | 6186 | -46.7% |
| pathfinder | tiling | bram | 94 | 66 | -29.8% |
| pathfinder | tiling | dsp | 0 | 0 | — |
| pathfinder | unroll | latency_ns | 7087000 | 7080000 | -0.1% |
| pathfinder | unroll | lut | 32160 | 28592 | -11.1% |
| pathfinder | unroll | ff | 15376 | 9957 | -35.2% |
| pathfinder | unroll | bram | 94 | 66 | -29.8% |
| pathfinder | unroll | dsp | 0 | 0 | — |
| pathfinder | doublebuffer | latency_ns | 3521000 | 3504000 | -0.5% |
| pathfinder | doublebuffer | lut | 44638 | 40734 | -8.7% |
| pathfinder | doublebuffer | ff | 16791 | 10349 | -38.4% |
| pathfinder | doublebuffer | bram | 158 | 130 | -17.7% |
| pathfinder | doublebuffer | dsp | 0 | 0 | — |
| pathfinder | coalescing | latency_ns | 73626 | 59714 | -18.9% |
| pathfinder | coalescing | lut | 112189 | 110792 | -1.2% |
| pathfinder | coalescing | ff | 39948 | 39689 | -0.6% |
| pathfinder | coalescing | bram | 30 | 30 | +0.0% |
| pathfinder | coalescing | dsp | 0 | 0 | — |
| knn | baseline | latency_ns | 3496000 | 13967000 | +299.5% |
| knn | baseline | lut | 5802 | 2047 | -64.7% |
| knn | baseline | ff | 8012 | 2296 | -71.3% |
| knn | baseline | bram | 30 | 2 | -93.3% |
| knn | baseline | dsp | 14 | 5 | -64.3% |
| knn | tiling | latency_ns | 14253000 | 14240000 | -0.1% |
| knn | tiling | lut | 6118 | 2995 | -51.0% |
| knn | tiling | ff | 7900 | 3423 | -56.7% |
| knn | tiling | bram | 33 | 5 | -84.8% |
| knn | tiling | dsp | 14 | 14 | +0.0% |
| knn | pipeline | latency_ns | 14253000 | 14247000 | -0.0% |
| knn | pipeline | lut | 6118 | 1609 | -73.7% |
| knn | pipeline | ff | 7900 | 2292 | -71.0% |
| knn | pipeline | bram | 33 | 3 | -90.9% |
| knn | pipeline | dsp | 14 | 14 | +0.0% |
| knn | unroll | latency_ns | 13482000 | 12487000 | -7.4% |
| knn | unroll | lut | 13203 | 8022 | -39.2% |
| knn | unroll | ff | 43839 | 36943 | -15.7% |
| knn | unroll | bram | 31 | 1 | -96.8% |
| knn | unroll | dsp | 28 | 28 | +0.0% |
| knn | doublebuffer | latency_ns | 5801000 | 7031000 | +21.2% |
| knn | doublebuffer | lut | 27758 | 22559 | -18.7% |
| knn | doublebuffer | ff | 76651 | 135291 | +76.5% |
| knn | doublebuffer | bram | 32 | 2 | -93.8% |
| knn | doublebuffer | dsp | 28 | 28 | +0.0% |
| knn | coalescing | latency_ns | 875000 | 485000 | -44.6% |
| knn | coalescing | lut | 23346 | 18611 | -20.3% |
| knn | coalescing | ff | 101850 | 162532 | +59.6% |
| knn | coalescing | bram | 30 | 0 | — |
| knn | coalescing | dsp | 224 | 224 | +0.0% |
