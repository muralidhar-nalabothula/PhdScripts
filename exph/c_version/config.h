#ifndef CONFIG_H
#define CONFIG_H

#include "types.h"

#define MAX_PATH 512

struct Config
{
    char SAVE_dir[MAX_PATH];
    char elph_file[MAX_PATH];
    char Dmat_file[MAX_PATH];
    char dipole_file[MAX_PATH];
    double omega_one_ph_range[3];  // min, max, numpoints
    int_type num_omega_two_ph_freq;
    double* omega_two_ph_freq;  // array
    double broading;
    int_type npol;
    int_type bands[2];  // min, max
    int_type one_ph;
    int_type two_ph;
};

int parse_config(const char* filename, struct Config* conf);

#endif
