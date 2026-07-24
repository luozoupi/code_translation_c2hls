#include "fpga_temp_matching.h"

buffer row_buf = {};
window win_buf = {};

void SAD_MATCH(axis_t *INPUT, axis_t *OUTPUT) {
    unsigned char templ[tmpsize] = {0};

    
    

    axis_t cur;
    int cur_data;
    int out[size];
    int i = 0, j = 0, k = 0, o = 0, m = 0, n = 0, l = 0;

    // Pull next pixel into buffer
    for (i = 0; i < size; i++) {
        
        cur = *INPUT;
        INPUT++;
        cur_data = cur.data;
        row_buf.buf[tmpdim - 1][k] = cur_data;

        // shift window
        for (m = 0; m < tmpdim - 1; m++) {
            
            for (n = 0; n < tmpdim; n++) {
                win_buf.win[n][m] = win_buf.win[n][m + 1];
            }
        }

        // pull column from buffer into window
        for (l = 0; l < tmpdim; l++) {
            
            win_buf.win[l][tmpdim - 1] = row_buf.buf[l][k];
        }

        // SAD (Sum of Absolute Differences)
        int y, z, sad = 0;
        int absl = 0;
        for (y = 0; y < tmpdim; y++) {
            
            for (z = 0; z < tmpdim; z++) {
                absl = win_buf.win[y][z] - templ[z + tmpdim * y] > 0
                           ? win_buf.win[y][z] - templ[z + tmpdim * y]
                           : win_buf.win[y][z] - templ[z + tmpdim * y] * -1;
                sad += absl;
            }
        }

        out[i] = (sad < thre ? 1 : 0);

        // if the buffer row is filled, shift buffer row up by 1
        k++;
        if (k == indim) {
            k = 0;
            for (j = 0; j < tmpdim - 1; j++) {
                
                for (o = 0; o < indim; o++) {
                    row_buf.buf[j][o] = row_buf.buf[j + 1][o];
                }
            }
        }

        cur.last = 0;
        cur.data = out[i];
        if (i == size - 1) {
            cur.last = 1;
        } else
            cur.last = 0;
        *OUTPUT++ = cur;
    }
}