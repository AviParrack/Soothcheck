#!/usr/bin/env python
"""
Check environment differences that might affect streaming.
Run this on both GPUs to compare environments.
"""

import sys
import os
import platform
import subprocess
import json
from pathlib import Path

def run_command(cmd, capture_output=True):
    """Run a shell command and return the output."""
    try:
        if isinstance(cmd, str):
            cmd = cmd.split()
        result = subprocess.run(cmd, capture_output=capture_output, text=True)
        return result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr.strip()}"
    except Exception as e:
        return f"Error: {e}"

def check_python_environment():
    """Check Python version and key packages."""
    info = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
    }
    
    # Check key packages
    packages_to_check = [
        "torch", "transformers", "fastapi", "uvicorn", "threading"
    ]
    
    for package in packages_to_check:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            info[f"{package}_version"] = version
            if package == "torch":
                try:
                    info["torch_cuda_available"] = module.cuda.is_available()
                    if module.cuda.is_available():
                        info["torch_cuda_version"] = module.version.cuda
                        info["torch_gpu_count"] = module.cuda.device_count()
                        info["torch_current_device"] = module.cuda.current_device()
                        info["torch_device_name"] = module.cuda.get_device_name()
                except Exception as e:
                    info["torch_cuda_error"] = str(e)
        except ImportError:
            info[f"{package}_version"] = "not installed"
    
    return info

def check_gpu_info():
    """Check GPU information."""
    gpu_info = {}
    
    # NVIDIA GPU info
    nvidia_smi = run_command("nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader,nounits")
    if not nvidia_smi.startswith("Error"):
        gpu_info["nvidia_smi"] = nvidia_smi
    else:
        gpu_info["nvidia_smi"] = "Not available"
    
    # CUDA version
    nvcc_version = run_command("nvcc --version")
    gpu_info["nvcc_version"] = nvcc_version
    
    # Driver version
    nvidia_driver = run_command("cat /proc/driver/nvidia/version")
    gpu_info["nvidia_driver"] = nvidia_driver
    
    return gpu_info

def check_system_resources():
    """Check system resources."""
    resources = {}
    
    # Memory
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        resources["memory_info"] = meminfo[:500]  # First 500 chars
    except:
        resources["memory_info"] = "Could not read /proc/meminfo"
    
    # CPU info
    cpu_info = run_command("lscpu")
    resources["cpu_info"] = cpu_info[:1000] if not cpu_info.startswith("Error") else "lscpu not available"
    
    # Load average
    try:
        resources["load_average"] = os.getloadavg()
    except:
        resources["load_average"] = "Not available"
    
    # Disk space
    disk_usage = run_command("df -h /")
    resources["disk_usage"] = disk_usage
    
    return resources

def check_network_environment():
    """Check network-related environment."""
    network = {}
    
    # Check if we're in a container or special environment
    if os.path.exists('/.dockerenv'):
        network["container"] = "Docker"
    elif os.environ.get('KUBERNETES_SERVICE_HOST'):
        network["container"] = "Kubernetes"
    else:
        network["container"] = "None detected"
    
    # Environment variables that might affect networking
    env_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY', 'CURL_CA_BUNDLE', 'REQUESTS_CA_BUNDLE']
    for var in env_vars:
        if var in os.environ:
            network[var] = os.environ[var]
    
    return network

def check_file_limits():
    """Check file descriptor limits and other system limits."""
    limits = {}
    
    # File descriptor limits
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        limits["file_descriptors"] = {"soft": soft, "hard": hard}
        
        # Memory limits
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        limits["virtual_memory"] = {"soft": soft, "hard": hard}
        
        # Process limits
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        limits["processes"] = {"soft": soft, "hard": hard}
        
    except Exception as e:
        limits["error"] = str(e)
    
    return limits

def main():
    """Main function to gather all environment information."""
    
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    print(f"Hostname: {platform.node()}")
    print(f"Timestamp: {subprocess.check_output(['date']).decode().strip()}")
    print()
    
    # Gather all information
    info = {
        "hostname": platform.node(),
        "timestamp": subprocess.check_output(['date']).decode().strip(),
        "python_environment": check_python_environment(),
        "gpu_info": check_gpu_info(),
        "system_resources": check_system_resources(),
        "network_environment": check_network_environment(),
        "system_limits": check_file_limits(),
    }
    
    # Print formatted output
    for section, data in info.items():
        print(f"\n=== {section.upper().replace('_', ' ')} ===")
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 200:
                    print(f"{key}: {value[:200]}...")
                else:
                    print(f"{key}: {value}")
        else:
            print(data)
    
    # Save to file for comparison
    output_file = f"environment_check_{platform.node()}_{int(__import__('time').time())}.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(info, f, indent=2, default=str)
        print(f"\n✓ Environment info saved to: {output_file}")
    except Exception as e:
        print(f"\n✗ Could not save to file: {e}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Key information for debugging
    python_env = info["python_environment"]
    print(f"Python: {python_env.get('python_version', 'unknown').split()[0]}")
    print(f"PyTorch: {python_env.get('torch_version', 'unknown')}")
    print(f"Transformers: {python_env.get('transformers_version', 'unknown')}")
    print(f"CUDA Available: {python_env.get('torch_cuda_available', 'unknown')}")
    print(f"GPU: {python_env.get('torch_device_name', 'unknown')}")

if __name__ == "__main__":
    main() 