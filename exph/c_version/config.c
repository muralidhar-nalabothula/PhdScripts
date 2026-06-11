#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int parse_config(const char* filename, struct Config* conf)
{
    FILE* f = fopen(filename, "r");
    if (!f)
    {
        return -1;
    }

    char line[1024];
    conf->num_omega_two_ph_freq = 0;
    conf->omega_two_ph_freq = NULL;

    while (fgets(line, sizeof(line), f))
    {
        char key[256];
        char val[512];
        if (sscanf(line, "%255s = %511[^\n]", key, val) == 2)
        {
            if (strcmp(key, "SAVE_dir") == 0)
            {
                strcpy(conf->SAVE_dir, val);
            }
            else if (strcmp(key, "elph_file") == 0)
            {
                strcpy(conf->elph_file, val);
            }
            else if (strcmp(key, "Dmat_file") == 0)
            {
                strcpy(conf->Dmat_file, val);
            }
            else if (strcmp(key, "dipole_file") == 0)
            {
                strcpy(conf->dipole_file, val);
            }
            else if (strcmp(key, "omega_one_ph_range") == 0)
            {
                sscanf(val, "%lf %lf %lf", &conf->omega_one_ph_range[0],
                       &conf->omega_one_ph_range[1],
                       &conf->omega_one_ph_range[2]);
            }
            else if (strcmp(key, "broading") == 0)
            {
                conf->broading = atof(val);
            }
            else if (strcmp(key, "npol") == 0)
            {
                conf->npol = atoll(val);
            }
            else if (strcmp(key, "bands") == 0)
            {
                sscanf(val, "%lld %lld", &conf->bands[0], &conf->bands[1]);
            }
            else if (strcmp(key, "one_ph") == 0)
            {
                conf->one_ph = atoll(val);
            }
            else if (strcmp(key, "two_ph") == 0)
            {
                conf->two_ph = atoll(val);
            }
        }
    }
    fclose(f);
    return 0;
}
