#include "netcdf_utils.h"

#include <stdio.h>
#include <stdlib.h>

// Basic wrapper to read a variable of type NC_DOUBLE or NC_FLOAT into a contiguous real_t array
int read_nc_real(int ncid, const char* varname, real_t** data,
                   size_t* total_size)
{
    int varid;
    if (nc_inq_varid(ncid, varname, &varid) != NC_NOERR)
    {
        return -1;
    }

    nc_type var_type;
    if (nc_inq_vartype(ncid, varid, &var_type) != NC_NOERR) {
        return -1;
    }

    if (sizeof(real_t) == sizeof(double) && var_type != NC_DOUBLE) {
        fprintf(stderr, "Error: Code compiled in double precision, but %s is not NC_DOUBLE.\n", varname);
        exit(1);
    }
    if (sizeof(real_t) == sizeof(float) && var_type != NC_FLOAT) {
        fprintf(stderr, "Error: Code compiled in single precision, but %s is not NC_FLOAT.\n", varname);
        exit(1);
    }

    int ndims;
    int dimids[NC_MAX_VAR_DIMS];
    if (nc_inq_var(ncid, varid, NULL, NULL, &ndims, dimids, NULL) != NC_NOERR)
    {
        return -1;
    }

    size_t size = 1;
    for (int_type i = 0; i < ndims; i++)
    {
        size_t dimlen;
        nc_inq_dimlen(ncid, dimids[i], &dimlen);
        size *= dimlen;
    }

    *data = malloc(size * sizeof(real_t));
    if (sizeof(real_t) == sizeof(double)) {
        if (nc_get_var_double(ncid, varid, (double*)*data) != NC_NOERR)
        {
            free(*data);
            return -1;
        }
    } else {
        if (nc_get_var_float(ncid, varid, (float*)*data) != NC_NOERR)
        {
            free(*data);
            return -1;
        }
    }

    if (total_size)
    {
        *total_size = size;
    }
    return 0;
}

// NetCDF often stores complex arrays with an extra dimension of size 2 at the
// end (real, imag)
int read_nc_complex(int ncid, const char* varname, complex_t** data,
                    size_t* total_size)
{
    real_t* raw_data;
    size_t raw_size;
    if (read_nc_real(ncid, varname, &raw_data, &raw_size) != 0)
    {
        return -1;
    }

    size_t cmplx_size = raw_size / 2;
    *data = malloc(cmplx_size * sizeof(complex_t));
    for (size_t i = 0; i < cmplx_size; i++)
    {
        (*data)[i] = raw_data[2 * i] + I * raw_data[2 * i + 1];
    }

    free(raw_data);
    if (total_size)
    {
        *total_size = cmplx_size;
    }
    return 0;
}

int read_nc_int(int ncid, const char* varname, int_type** data,
                size_t* total_size)
{
    int varid;
    if (nc_inq_varid(ncid, varname, &varid) != NC_NOERR)
    {
        return -1;
    }

    int ndims;
    int dimids[NC_MAX_VAR_DIMS];
    if (nc_inq_var(ncid, varid, NULL, NULL, &ndims, dimids, NULL) != NC_NOERR)
    {
        return -1;
    }

    size_t size = 1;
    for (int_type i = 0; i < ndims; i++)
    {
        size_t dimlen;
        nc_inq_dimlen(ncid, dimids[i], &dimlen);
        size *= dimlen;
    }

    int* temp_data = malloc(size * sizeof(int));
    if (nc_get_var_int(ncid, varid, temp_data) != NC_NOERR)
    {
        free(temp_data);
        return -1;
    }

    *data = malloc(size * sizeof(int_type));
    for (size_t i = 0; i < size; i++)
    {
        (*data)[i] = (int_type)temp_data[i];
    }
    free(temp_data);

    if (total_size)
    {
        *total_size = size;
    }
    return 0;
}

int create_nc_file(const char* filename, int* ncid) {
    if (nc_create(filename, NC_CLOBBER | NC_NETCDF4, ncid) != NC_NOERR) {
        return -1;
    }
    return 0;
}

// Assumes we will enter define mode, add var, exit define mode, and write data.
int write_nc_real_1d(int ncid, const char* varname, const char* dimname, real_t* data, size_t size) {
    int dimid, varid;
    nc_redef(ncid);
    if (nc_inq_dimid(ncid, dimname, &dimid) != NC_NOERR) {
        nc_def_dim(ncid, dimname, size, &dimid);
    }
    
    nc_type var_type = (sizeof(real_t) == sizeof(double)) ? NC_DOUBLE : NC_FLOAT;
    nc_def_var(ncid, varname, var_type, 1, &dimid, &varid);
    nc_enddef(ncid);

    if (var_type == NC_DOUBLE) {
        nc_put_var_double(ncid, varid, (const double*)data);
    } else {
        nc_put_var_float(ncid, varid, (const float*)data);
    }
    return 0;
}

int write_nc_complex_nd(int ncid, const char* varname, int ndims, const char** dimnames, const size_t* dimsizes, complex_t* data) {
    int dimids[NC_MAX_VAR_DIMS];
    nc_redef(ncid);
    
    for (int i = 0; i < ndims; i++) {
        if (nc_inq_dimid(ncid, dimnames[i], &dimids[i]) != NC_NOERR) {
            nc_def_dim(ncid, dimnames[i], dimsizes[i], &dimids[i]);
        }
    }
    
    // Add complex dimension
    int cplx_dimid;
    if (nc_inq_dimid(ncid, "complex", &cplx_dimid) != NC_NOERR) {
        nc_def_dim(ncid, "complex", 2, &cplx_dimid);
    }
    dimids[ndims] = cplx_dimid;
    
    nc_type var_type = (sizeof(real_t) == sizeof(double)) ? NC_DOUBLE : NC_FLOAT;
    int varid;
    nc_def_var(ncid, varname, var_type, ndims + 1, dimids, &varid);
    nc_enddef(ncid);

    size_t total_size = 1;
    for (int i = 0; i < ndims; i++) total_size *= dimsizes[i];

    real_t* raw_data = malloc(total_size * 2 * sizeof(real_t));
    for (size_t i = 0; i < total_size; i++) {
        raw_data[2*i] = creal(data[i]);
        raw_data[2*i+1] = cimag(data[i]);
    }

    if (var_type == NC_DOUBLE) {
        nc_put_var_double(ncid, varid, (const double*)raw_data);
    } else {
        nc_put_var_float(ncid, varid, (const float*)raw_data);
    }
    
    free(raw_data);
    return 0;
}