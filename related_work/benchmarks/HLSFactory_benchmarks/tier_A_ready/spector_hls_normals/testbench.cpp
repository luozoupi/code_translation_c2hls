#include "normals.h"
#include "params.h"
#include <stdio.h>

int main() {
    static int vmap[rows * cols * 3];
    static int nmap[rows * cols * 3];
    for (int i = 0; i < rows * cols; i++) {
        const int r = i / cols;
        const int c = i % cols;
        vmap[i * 3 + 0] = r + 1;
        vmap[i * 3 + 1] = c + 1;
        vmap[i * 3 + 2] = (r * 3 + c) % 17 + 1;
    }
    for (int i = 0; i < rows * cols * 3; i++)
        nmap[i] = 0;
    normals(vmap, nmap);
    printf("normals csim done\n");
    return 0;
}
