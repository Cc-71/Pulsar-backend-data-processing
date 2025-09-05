# Pulsar-backend-data-processing
A high-performance multi-GPU system for real-time pulsar radio signal processing. Captures UDP data via optimized multi-threaded reception, processes with GPU-accelerated FFT, stores in FITS format.Features high-concurrency lock-free buffers, multi-channel accumulation, parallel computing, batch storage, and performance monitoring. 

# Sender filess
send_test2.py - A file for simulating packet transmission from the sender side
# Receiver files
.cu, .h, and .cpp files are backend data processing files
.cu: CUDA source files containing GPU-accelerated code for parallel data processing (e.g., FFT computations in your pulsar system).
.h: Header files with function declarations, macros, and data structure definitions shared across the backend.
.cpp: C++ source files implementing core backend logic like data parsing, network handling, and integration with GPU processing.



# Compile CUDA files
nvcc -std=c++17 -O3 -c cuda_fft.cu -o cuda_fft.o

# Compile main program
g++ -std=c++17 -O3 -I/usr/local/cuda/include -c main.cpp -o main.o

# Link
g++ main.o cuda_fft.o -o udp_processor \
    -L/usr/local/cuda/lib64 -lcudart -lcufft \
    -lboost_system -lcfitsio -lpthread

# Run .py file
python send_test2.py
// First compile source files to generate object files (.o)
// Use nvcc to link object files and libraries
// Generate the final executable file
