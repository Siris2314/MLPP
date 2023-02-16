#ifndef _MAX_AVERAGE_HPP
#define _MAX_AVERAGE_HPP

#include "Config.hpp"


namespace internal{
    inline int find_max_val(const Scalar* x, const int n){
        int max_index = 0;
        for(int i = 1; i<n; i++){
            max_index = (x[i] > x[max_index]) ? i : max_index;
        }
        return max_index;
    }


    inline Scalar find_block_max(const Scalar* x, const int nrow, const int ncol, const int col_stride, int& loc){

        loc = find_max_val(x, nrow);
        Scalar max_val = x[loc];

        x+=col_stride;
        int loc_next = find_max_val(x, nrow);
        Scalar val_next = x[loc_next];
        if(val_next > max_val) {loc = col_stride + loc_next; max_val = val_next;}
        if(ncol == 2) return max_val;

        for(int i = 2; i<ncol; i++){
            x+= col_stride;
            loc_next = find_max_val(x, nrow);
            val_next = x[loc_next];
            if(val_next > max_val) {loc = i*col_stride + loc_next; max_val = val_next;}
        }


        return max_val;



    }

    inline Scalar sum_row(const Scalar* x, const int n){
        Scalar c = 0;
        for(int i = 0; i<n; i++){
            c += x[i];
        }

        return c;
    }

    inline Scalar average_block(const Scalar* x, const int nrow, const int ncol, const int col_stride, int& loc){
        Scalar sum = 0;
        for(int i = 0; i<ncol; i++){
            x += col_stride;
            sum += sum_row(x, nrow);
        }
         return sum / (ncol*nrow);
    }
};



#endif