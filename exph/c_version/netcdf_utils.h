#ifndef NETCDF_UTILS_H
#define NETCDF_UTILS_H

#include <netcdf.h>

#include "types.h"

int read_nc_real(int ncid, const char* varname, real_t** data,
                 size_t* total_size);
int read_nc_complex(int ncid, const char* varname, complex_t** data,
                    size_t* total_size);
int read_nc_int(int ncid, const char* varname, int_type** data,
                size_t* total_size);

// Writing wrappers
int create_nc_file(const char* filename, int* ncid);
int write_nc_real_1d(int ncid, const char* varname, const char* dimname, real_t* data, size_t size);
int write_nc_complex_nd(int ncid, const char* varname, int ndims, const char** dimnames, const size_t* dimsizes, complex_t* data);

#endif
