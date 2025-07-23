# NVIDIA Ada GPU architecture
# 80 GB of global memory
# 300 SMs
# 32 threads per warp
# Per SM
# - max warps = 48
# - max thread blocks = 24
# - 64_000 32-bit registers = 256 KB but only 251.9 KB when distributed evenly over the 1536 threads. So each thread block has 251.9 / 24 = 10.49583 KB
# - 100 KB shared memory = 4.16 KB per thread block = 65 bytes per thread
#
# Total threads on GPU:
# 300 * 1536 = 460_800
#
# Total number of molecules that fit in global memory when a molecule has 8.3 MB of data.
# 80 GB / 8.3 MB = 9638.554
#
# Number of threads available per molecule
# 460_800 / 9638 = 47.81
#
# The L2 cache is 98_304 KB
# There are 300 * 24 = 7200 thread blocks, so each has:
# 98_304 / 7200 = 13.653 KB
#
# 47 threads fit within two warps, so for one molecule, this case we need:
# 1 thread block with 2 warps
# This thread block has this much when combining shared + L2 + register memory per thread block:
# 4.16 KB + 10.49583 KB + 13.653 KB = 28.30883 KB
# 
# And it has 11.1 MB of global memory. So the molecule which is 8.3 MB fits.
# 80_000 / (300 * 24) = 80_000 / 7200 = 11.1111 MB


import math

# Constants for NVIDIA Ada GPU architecture
MEM_GLOBAL = 80_000 # MB
SM_COUNT = 300
THREADS_PER_WARP = 32
L2_CACHE = 98_304 # KB

## Per SM constants ##
MAX_WARPS = 48
MAX_THREAD_BLOCKS = 24
REGISTER_COUNT = 64_000 # 32-bit registers
BITS_PER_REGISTER = 32
MEM_SHARED = 100 # KB
######################

SM_THREADS_TOTAL = MAX_WARPS * THREADS_PER_WARP

THREAD_BLOCKS_TOTAL = SM_COUNT * MAX_THREAD_BLOCKS
MOLECULES_PER_BLOCK = 1

def compute(bytes_per_molecule):
    # Total number of molecules that fit in global memory
    molecule_capacity_global = math.floor(MEM_GLOBAL / (bytes_per_molecule / 1_000_000))
    THREADS_TOTAL = THREADS_PER_WARP * MAX_WARPS * SM_COUNT
    # Number of threads available per molecule
    threads_per_molecule = math.ceil((math.floor(THREADS_TOTAL / molecule_capacity_global)*molecule_capacity_global)/molecule_capacity_global)

    # Number of warps needed per molecule
    warps_per_molecule = math.ceil(threads_per_molecule / THREADS_PER_WARP)
    used_warps_per_sm = MAX_WARPS - (MAX_WARPS % warps_per_molecule)
    warps_per_block = warps_per_molecule * MOLECULES_PER_BLOCK
    sm_threads_used = used_warps_per_sm * THREADS_PER_WARP # Total thread count per SM
    total_blocks_used_per_sm = MAX_THREAD_BLOCKS - (MAX_THREAD_BLOCKS % warps_per_block) # Total blocks used on GPU
    total_blocks_used = SM_COUNT * total_blocks_used_per_sm
    total_blocks_used = min(molecule_capacity_global, total_blocks_used)
    total_threads_used = sm_threads_used * SM_COUNT # Total thread count on GPU
    mem_register_per_block = math.floor(REGISTER_COUNT / sm_threads_used) * (BITS_PER_REGISTER / 8) * sm_threads_used / 1000

    l2_cache_per_block = L2_CACHE / total_blocks_used_per_sm
    mem_shared_per_block = MEM_SHARED / total_blocks_used_per_sm
    mem_global_per_block = MEM_GLOBAL / total_blocks_used

    mem_non_global_per_block = mem_shared_per_block + mem_register_per_block + l2_cache_per_block
    mem_per_block_total = (mem_non_global_per_block / 1000) + mem_global_per_block

    print(f"Global memory: {MEM_GLOBAL / 1000} GB")
    print(f"L2 cache: {L2_CACHE} KB")
    print(f"Total SMs: {SM_COUNT}")
    print(f"Threads per warp: {THREADS_PER_WARP}")

    print()

    print(f"Warps per SM: {MAX_WARPS}")
    print(f"Thread blocks per SM: {MAX_THREAD_BLOCKS}")
    print(f"Registers per SM: {REGISTER_COUNT} {BITS_PER_REGISTER}-bit")
    print(f"Shared memory per SM: {MEM_SHARED} KB")
    print(f"Total number of threads per SM: {SM_THREADS_TOTAL}")
    print(f"Total number of threads used per SM: {sm_threads_used}")

    print()

    print(f"Total number of threads on GPU: {THREADS_TOTAL}")
    print(f"Total number of threads used on GPU: {total_threads_used}")
    print(f"Total number of thread blocks on GPU: {THREAD_BLOCKS_TOTAL}")
    print(f"Total number of thread blocks used on GPU: {total_blocks_used}")

    print()

    print(f"L2 cache per thread block: {l2_cache_per_block} KB")
    print(f"Register memory per thread block: {mem_register_per_block} KB")
    print(f"Shared memory per thread block: {mem_shared_per_block} KB")
    print(f"Global memory per thread block: {mem_global_per_block} MB")
    print(f"Total memory per thread block: {mem_per_block_total} MB")

    print()

    print(f"Warps per molecule: {warps_per_molecule}")


compute(8300000)
