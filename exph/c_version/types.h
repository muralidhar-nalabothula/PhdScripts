#ifndef TYPES_H
#define TYPES_H

#include <complex.h>

typedef long long int_type;

#ifdef USE_DOUBLE_PRECISION
typedef double real_t;
typedef double complex complex_t;
#else
typedef float real_t;
typedef float complex complex_t;
#endif

#endif
