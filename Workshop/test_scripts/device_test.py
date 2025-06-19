import torch
import time
import platform
import os
import argparse


def test_device(device_name, size=5000, iterations=10, warmup=5, synchronize=True):
    """
    Test performance on a specified device

    Args:
        device_name: 'cpu', 'cuda', or 'mps'
        size: matrix dimension
        iterations: number of matmul operations to time
        warmup: number of warmup operations
        synchronize: whether to synchronize GPU after operations
    """
    device = torch.device(device_name)
    print(f"Testing on {device_name} (size={size}x{size}, iterations={iterations})...")

    # Create matrices
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = torch.matmul(a, b)

    # Synchronize after warmup
    if synchronize:
        if device_name == "cuda":
            torch.cuda.synchronize()
        elif device_name == "mps":
            torch.mps.synchronize()

    # Benchmark
    print(f"Benchmarking ({iterations} iterations)...")
    start = time.time()
    for _ in range(iterations):
        c = torch.matmul(a, b)

    # Synchronize before stopping timer
    if synchronize:
        if device_name == "cuda":
            torch.cuda.synchronize()
        elif device_name == "mps":
            torch.mps.synchronize()

    end = time.time()
    total_time = end - start
    per_iteration = total_time / iterations
    return total_time, per_iteration


def auto_size_for_gpu(device_name):
    """Auto-select matrix size based on GPU memory"""
    if device_name == "cpu":
        return 5000  # Default for CPU

    try:
        if device_name == "cuda":
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            if gpu_mem > 20:  # High-end GPU (A100, etc.)
                return 15000
            elif gpu_mem > 10:  # Mid-range GPU (RTX 3080, etc.)
                return 10000
            else:  # Lower-end GPU
                return 8000
        elif device_name == "mps":
            # For Apple Silicon, use system memory as a proxy
            # This is imperfect since unified memory is shared
            system_mem = (
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
            )
            if system_mem > 32:  # M1 Max, M2 Ultra, etc.
                return 8000
            elif system_mem > 16:  # M1 Pro, M2, etc.
                return 7000
            else:  # Base M1, etc.
                return 6000
    except:
        # Fallback if detection fails
        return 5000

    return 5000  # Default fallback


def run_all_benchmarks(args):
    """Run benchmarks on all available devices"""
    devices_to_test = ["cpu"]
    benchmarks = {}

    size = args.size
    iterations = args.iter
    warmup = args.warmup

    # Check CUDA availability
    if torch.cuda.is_available():
        devices_to_test.append("cuda")
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

        if size <= 0:
            size = auto_size_for_gpu("cuda")

    # Check MPS availability
    elif (
        hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        devices_to_test.append("mps")
        print("MPS available on Apple Silicon GPU")

        if size <= 0:
            size = auto_size_for_gpu("mps")

    # If no size specified and no GPU detected, use default
    if size <= 0:
        size = auto_size_for_gpu("cpu")

    print(f"\nUsing matrix size: {size}x{size}\n")

    # Run tests
    for device in devices_to_test:
        try:
            total_time, per_iter_time = test_device(device, size, iterations, warmup)
            benchmarks[device] = (total_time, per_iter_time)
        except Exception as e:
            print(f"Error testing {device}: {e}")

    # Print results
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)

    cpu_total = benchmarks.get("cpu", (0, 0))[0]

    for device, (total, per_iter) in benchmarks.items():
        print(
            f"{device.upper()} total time: {total:.4f}s ({per_iter:.4f}s per iteration)"
        )

        if device != "cpu" and cpu_total > 0:
            speedup = cpu_total / total
            print(f"{device.upper()} speedup over CPU: {speedup:.2f}x")

            if device == "cuda" and speedup < 10:
                print("Note: CUDA speedup is lower than expected for a high-end GPU.")
                print(
                    "Possible reasons: small matrices, CPU thread contention, PCIe transfer overhead"
                )
            elif device == "mps" and speedup < 3:
                print("Note: MPS speedup is lower than expected for Apple Silicon.")
                print(
                    "Possible reasons: power saving mode, thermal throttling, background processes"
                )

    print("=" * 50)
    print(f"System: {platform.system()} {platform.release()} {platform.machine()}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Performance Benchmark")
    parser.add_argument(
        "--size",
        type=int,
        default=0,
        help="Matrix size (default: auto-detected based on GPU)",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=10,
        help="Number of iterations for benchmark (default: 10)",
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of warmup iterations (default: 5)"
    )

    args = parser.parse_args()
    run_all_benchmarks(args)
