# NVIDIA Ada GPU architecture
# 80 GB of global memory
# 128 SMs
# 32 threads per warp
# Per SM
# - max warps = 48
# - max blocks = 24
# - 64_000 32-bit registers = 256 KB but only 251.9 KB when distributed evenly over the 1536 threads. So each block has 251.9 / 24 = 10.49583 KB
# - 100 KB shared memory = 4.16 KB per block = 65 bytes per thread
#
# Total threads on GPU:
# 128 * 1536 = 196_608
#
# Total number of molecules that fit in global memory when a molecule has 8.3 MB of data.
# 80 GB / 8.3 MB = 9638.554
#
# Number of threads available per molecule
# 196_608 / 9638 = 20.399
#
# The L2 cache is 98_304 KB
# There are 128 * 24 = 3072 blocks, so each has:
# 98_304 / 3072 = 32 KB
#
# 20 threads fit within one warp, so for one molecule, in this case we need:
# 1 block with 1 warp, so each block can do 2 molecules
# This block has this much when combining shared + L2 + register memory per block:
# 4.16 KB + 10.49583 KB + 32 KB = 46.65583 KB
# 
# And it has 26.04166 MB of global memory. So the molecule which is 8.3 MB fits.
# 80_000 / (128 * 24) = 80_000 / 3072 = 26.04166 MB


import math

# Constants for NVIDIA Ada GPU architecture
MEM_GLOBAL = 80_000 # MB
SM_COUNT = 128
THREADS_PER_WARP = 32
L2_CACHE = 98_304 # KB

## Per SM constants ##
MAX_WARPS = 48
MAX_BLOCKS = 24
REGISTER_COUNT = 64_000 # 32-bit registers
BITS_PER_REGISTER = 32
MEM_SHARED = 100 # KB
######################

SM_THREADS_TOTAL = MAX_WARPS * THREADS_PER_WARP

BLOCKS_TOTAL = SM_COUNT * MAX_BLOCKS

def compute(bytes_per_molecule):
    # Total number of molecules that fit in global memory
    mb_per_molecule = bytes_per_molecule / 1_000_000
    molecule_capacity_global = math.floor(MEM_GLOBAL / mb_per_molecule)
    THREADS_TOTAL = THREADS_PER_WARP * MAX_WARPS * SM_COUNT

    mem_global_per_thread = MEM_GLOBAL / THREADS_TOTAL;

    mem_shared_per_thread = (MEM_SHARED * SM_COUNT) / THREADS_TOTAL;
    mem_register_per_thread = (REGISTER_COUNT * BITS_PER_REGISTER * SM_COUNT) / THREADS_TOTAL / 8 / 1_000_000; # MB
    mem_l2_cache_per_thread = L2_CACHE / 1000 / THREADS_TOTAL

    mem_non_global_per_thread = mem_shared_per_thread + mem_register_per_thread + mem_l2_cache_per_thread
    mem_per_thread_total = mem_non_global_per_thread + mem_global_per_thread

    print(f"mem global per thread: {mem_global_per_thread} MB")
    print(f"mem non-global per thread: {mem_non_global_per_thread} MB")
    print(f"mem per thread total: {mem_per_thread_total} MB")
    print(f"mb per molecule: {mb_per_molecule} MB")


    # Number of threads needed per molecule
    #threads_per_molecule = math.floor(THREADS_TOTAL / molecule_capacity_global)
    threads_per_molecule = math.ceil(mb_per_molecule / mem_per_thread_total)

    print(f"threads per molecule: {threads_per_molecule}")

    # Number of warps needed per molecule
    warps_per_molecule = math.ceil(threads_per_molecule / THREADS_PER_WARP)
    used_warps_per_sm = MAX_WARPS - (MAX_WARPS % warps_per_molecule)
    molecules_per_block = math.ceil(2 / warps_per_molecule)
    warps_per_block = warps_per_molecule * molecules_per_block
    sm_threads_used = used_warps_per_sm * THREADS_PER_WARP
    total_blocks_used_per_sm = ((MAX_WARPS / MAX_BLOCKS) / warps_per_block) * MAX_BLOCKS
    total_blocks_used =  total_blocks_used_per_sm * SM_COUNT
    total_threads_used = sm_threads_used * SM_COUNT
    mem_register_per_block = math.floor(REGISTER_COUNT / sm_threads_used) * (BITS_PER_REGISTER / 8) * sm_threads_used / 1000

    l2_cache_per_block = L2_CACHE / total_blocks_used_per_sm
    mem_shared_per_block = MEM_SHARED / total_blocks_used_per_sm
    mem_global_per_block = MEM_GLOBAL / total_blocks_used

    mem_non_global_per_block = mem_shared_per_block + mem_register_per_block + l2_cache_per_block
    mem_per_block_total = (mem_non_global_per_block / 1000) + mem_global_per_block

    print(f"| Global memory | L2 cache | Total SMs | Threads per warp | Threads Available | Threads Used | Blocks Available | Blocks Used |")
    print(f"|    {MEM_GLOBAL / 1000} GB    | {L2_CACHE} KB |    {SM_COUNT}    |        {THREADS_PER_WARP}        |      {THREADS_TOTAL}       |    {total_threads_used}    |       {BLOCKS_TOTAL}       |    {total_blocks_used}     |")

    print()

    #print(f"Global memory: {MEM_GLOBAL / 1000} GB")
    #print(f"L2 cache: {L2_CACHE} KB")
    #print(f"Total SMs: {SM_COUNT}")
    #print(f"Threads per warp: {THREADS_PER_WARP}")

    #print()

    print(f"#################################### Per SM ####################################")
    print(f"| Warps | Blocks | Registers | Shared Memory | Threads Available | Threads Used |")
    print(f"|  {MAX_WARPS}   |   {MAX_BLOCKS}   |   {REGISTER_COUNT}   |    {MEM_SHARED} KB     |       {SM_THREADS_TOTAL}        |     {sm_threads_used}     |")

    print()

    #print(f"Warps per SM: {MAX_WARPS}")
    #print(f"Blocks per SM: {MAX_BLOCKS}")
    #print(f"Registers per SM: {REGISTER_COUNT} {BITS_PER_REGISTER}-bit")
    #print(f"Shared memory per SM: {MEM_SHARED} KB")
    #print(f"Threads available per SM: {SM_THREADS_TOTAL}")
    #print(f"Threads used per SM: {sm_threads_used}")

    #print()

    #print(f"Threads available on GPU: {THREADS_TOTAL}")
    #print(f"Threads used on GPU: {total_threads_used}")
    #print(f"Blocks available on GPU: {BLOCKS_TOTAL}")
    #print(f"Blocks used on GPU: {total_blocks_used}")

    #print()


    print(f"########################### Per Block ###########################")
    print(f"| L2 Cache | Register Mem | Shared Mem | Global Mem | Total mem |")
    print(f"| {"%.1f" % l2_cache_per_block} KB|  {"%.3f" % mem_register_per_block} KB  |  {"%.3f" % mem_shared_per_block} KB  |  {"%.3f" % mem_global_per_block} MB | {"%.3f" % mem_per_block_total} MB |")

    print()

    #print(f"L2 cache per block: {l2_cache_per_block} KB")
    #print(f"Register memory per block: {mem_register_per_block} KB")
    #print(f"Shared memory per block: {mem_shared_per_block} KB")
    #print(f"Global memory per block: {mem_global_per_block} MB")
    #print(f"Total memory per block: {mem_per_block_total} MB")

    #print()

    print(f"Warps per molecule: {warps_per_molecule}")
    print(f"Molecules per block: {molecules_per_block}")


compute(8300000)
