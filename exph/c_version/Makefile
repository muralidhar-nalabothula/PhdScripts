CC = gcc-15
#CFLAGS = #-g -fsanitize=address -I/Users/murali/softwares/core/include
CFLAGS = -Wall -O3 -std=c99 -fopenmp -DWITH_OPENMP -I/Users/murali/softwares/core/include
LDFLAGS = -L/Users/murali/softwares/core/lib -lnetcdf -lm

SRC = main.c config.c netcdf_utils.c symmetry.c raman_kernels.c raman_kernels_twoph.c read_nc_data.c
OBJ = $(SRC:.c=.o)
TARGET = raman_calc

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
