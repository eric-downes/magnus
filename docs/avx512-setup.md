# AVX-512 Development Setup for MAGNUS

## Hetzner Cloud Recommendations

### Required CPU Features
For AVX-512 support, you need Intel CPUs from:
- **Skylake-X** (2017+): Full AVX-512 Foundation, CD, VL, DQ, BW
- **Cascade Lake** (2019+): Adds AVX-512 VNNI
- **Ice Lake** (2019+): Adds more AVX-512 extensions
- **Sapphire Rapids** (2023+): Latest AVX-512 features

### Recommended Hetzner Options

#### Option 1: Dedicated Server (Best Performance)
**AX102** or **AX162** from Hetzner Dedicated
- Intel Xeon W-2145 (Skylake) or better
- 8-16 cores with AVX-512
- 128-256GB RAM for large matrix tests
- ~€140-250/month
- Full AVX-512 support guaranteed

#### Option 2: Cloud VM - CCX Series (Good Balance)
**CCX32** or **CCX42**
- Dedicated vCPU (no sharing)
- 8-16 vCPUs 
- 32-64GB RAM
- ~€100-200/month
- Based on AMD EPYC - **Note: AMD EPYC does NOT have AVX-512!**

#### Option 3: Cloud VM - CX Series (Budget Testing)
**CX41** or **CX51**
- Shared vCPU (Intel Xeon)
- 8-16 vCPUs
- 16-32GB RAM  
- ~€30-60/month
- May or may not have AVX-512 depending on host

### ⚠️ IMPORTANT: Verification Required

**Most Hetzner Cloud VMs use AMD EPYC processors which do NOT support AVX-512!**

You need to either:
1. Get a **dedicated Intel server** (AX series)
2. Use **Intel-based cloud instances** (less common)
3. Try **other providers** with Intel Ice Lake/Cascade Lake

### Alternative Providers with AVX-512

#### AWS EC2
- **m6i.2xlarge** or larger: Intel Ice Lake with AVX-512
- **c6i.4xlarge**: Compute optimized with AVX-512
- ~$0.30-0.70/hour

#### Google Cloud (GCP)
- **n2-standard-8**: Intel Cascade Lake
- **c2-standard-8**: Intel Cascade Lake optimized
- ~$0.35-0.50/hour

#### Azure
- **Standard_D8s_v5**: Intel Ice Lake
- **Standard_F8s_v2**: Intel Cascade Lake
- ~$0.30-0.40/hour

## Setup Script for Linux Development

```bash
#!/bin/bash
# setup-avx512-dev.sh

# Check CPU features
echo "=== Checking CPU Features ==="
if lscpu | grep -q avx512f; then
    echo "✅ AVX-512 Foundation detected"
    lscpu | grep -E "avx512" | head -5
else
    echo "❌ No AVX-512 support detected!"
    echo "This CPU does not support AVX-512 instructions."
    exit 1
fi

# Install development tools
echo "=== Installing Development Tools ==="
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    htop \
    linux-tools-common \
    linux-tools-generic \
    linux-tools-$(uname -r) \
    valgrind

# Install Rust
echo "=== Installing Rust ==="
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
rustup default stable
rustup component add rust-src

# Install performance tools
echo "=== Installing Performance Tools ==="
cargo install flamegraph
cargo install cargo-criterion

# Clone MAGNUS repository
echo "=== Cloning MAGNUS ==="
git clone https://github.com/yourusername/magnus.git
cd magnus

# Create AVX-512 feature detection test
cat > test_avx512.c << 'EOF'
#include <stdio.h>
#include <immintrin.h>

int main() {
    if (__builtin_cpu_supports("avx512f")) {
        printf("AVX-512 Foundation: YES\n");
    }
    if (__builtin_cpu_supports("avx512dq")) {
        printf("AVX-512 DQ: YES\n");
    }
    if (__builtin_cpu_supports("avx512bw")) {
        printf("AVX-512 BW: YES\n");
    }
    if (__builtin_cpu_supports("avx512vl")) {
        printf("AVX-512 VL: YES\n");
    }
    
    // Test actual AVX-512 instruction
    __m512i a = _mm512_set1_epi32(1);
    __m512i b = _mm512_set1_epi32(2);
    __m512i c = _mm512_add_epi32(a, b);
    
    printf("AVX-512 instruction test: PASSED\n");
    return 0;
}
EOF

echo "=== Testing AVX-512 Support ==="
gcc -mavx512f test_avx512.c -o test_avx512
./test_avx512

echo "=== Setup Complete ==="
echo "Run benchmarks with:"
echo "  cd magnus"
echo "  BENCH_TIER=quick cargo bench --bench tiered_benchmark"
```

## Development VM Recommendation

### For MAGNUS Testing: AWS EC2 m6i.4xlarge
- **vCPUs**: 16 (Intel Ice Lake)
- **RAM**: 64 GB
- **AVX-512**: Full support including VNNI
- **Cost**: ~$0.77/hour (spot: ~$0.30/hour)
- **Storage**: 100GB gp3 SSD

### Setup Commands
```bash
# Launch instance (using AWS CLI)
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \  # Ubuntu 22.04
    --instance-type m6i.4xlarge \
    --key-name your-key \
    --security-groups your-sg \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]'

# Or use spot instance for 60% savings
aws ec2 request-spot-instances \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification file://spot-spec.json
```

## Testing AVX-512 on the VM

```bash
# After setup, verify AVX-512
cat /proc/cpuinfo | grep avx512

# Run MAGNUS benchmarks
cd magnus
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Quick test
BENCH_TIER=quick cargo bench --bench tiered_benchmark

# Large matrix benchmark (primary use case)
BENCH_TIER=large cargo bench --bench tiered_benchmark
```

## Cost Optimization Tips

1. **Use Spot Instances**: 60-70% cheaper for development
2. **Stop When Not Using**: Don't leave running overnight
3. **Use Smaller Instance for Initial Development**: m6i.2xlarge for coding, m6i.4xlarge for benchmarks
4. **Consider Dedicated Hosts**: If testing for >1 week, monthly dedication might be cheaper

## Recommended Approach

Since Hetzner Cloud mostly uses AMD EPYC (no AVX-512), I recommend:

1. **Development**: AWS EC2 `m6i.2xlarge` spot instance (~$0.15/hour)
2. **Benchmarking**: AWS EC2 `m6i.4xlarge` spot instance (~$0.30/hour)
3. **Alternative**: Google Cloud `c2-standard-8` with Intel Cascade Lake

This gives you:
- Guaranteed AVX-512 support
- Hourly billing (no monthly commitment)
- Easy to scale up/down
- Good performance for large matrix tests