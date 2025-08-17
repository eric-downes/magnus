#!/bin/bash

# MAGNUS AVX-512 Development Environment Setup
# Run this on a fresh Ubuntu 22.04 instance with AVX-512 support

set -e  # Exit on error

echo "================================================"
echo "  MAGNUS AVX-512 Development Setup"
echo "================================================"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo "Please run as a normal user (not root)"
   exit 1
fi

# Check CPU features
echo "=== Step 1: Checking CPU Features ==="
if lscpu | grep -q avx512f; then
    echo "✅ AVX-512 Foundation detected"
    echo ""
    echo "Available AVX-512 features:"
    lscpu | grep -E "avx512" || echo "Basic AVX-512 support"
    echo ""
else
    echo "❌ No AVX-512 support detected!"
    echo ""
    echo "This CPU does not support AVX-512 instructions."
    echo "You need an Intel CPU with AVX-512 support:"
    echo "  - Intel Skylake-X or newer"
    echo "  - Intel Xeon Scalable (Cascade Lake, Ice Lake)"
    echo ""
    echo "Current CPU:"
    lscpu | grep "Model name"
    exit 1
fi

# Update system
echo "=== Step 2: Updating System ==="
sudo apt-get update
sudo apt-get upgrade -y

# Install development tools
echo "=== Step 3: Installing Development Tools ==="
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    htop \
    pkg-config \
    libssl-dev \
    linux-tools-common \
    linux-tools-generic \
    linux-tools-$(uname -r) \
    valgrind \
    gdb

# Install Rust if not present
echo "=== Step 4: Installing Rust ==="
if ! command -v rustc &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
else
    echo "Rust already installed: $(rustc --version)"
fi

# Update Rust and add components
source $HOME/.cargo/env
rustup default stable
rustup update
rustup component add rust-src
rustup component add llvm-tools-preview

# Install performance analysis tools
echo "=== Step 5: Installing Performance Tools ==="
cargo install flamegraph || true
cargo install cargo-criterion || true
cargo install cargo-asm || true

# Create test directory
echo "=== Step 6: Creating Test Programs ==="
mkdir -p ~/avx512-tests
cd ~/avx512-tests

# Create C test for AVX-512
cat > test_avx512.c << 'EOF'
#include <stdio.h>
#include <immintrin.h>
#include <stdint.h>

void print_m512i(__m512i v) {
    int32_t* vals = (int32_t*)&v;
    printf("[");
    for(int i = 0; i < 16; i++) {
        printf("%d", vals[i]);
        if(i < 15) printf(", ");
    }
    printf("]\n");
}

int main() {
    printf("=== AVX-512 Feature Test ===\n\n");
    
    // Check CPU support
    if (__builtin_cpu_supports("avx512f")) {
        printf("✅ AVX-512 Foundation (F)\n");
    }
    if (__builtin_cpu_supports("avx512dq")) {
        printf("✅ AVX-512 Doubleword/Quadword (DQ)\n");
    }
    if (__builtin_cpu_supports("avx512bw")) {
        printf("✅ AVX-512 Byte/Word (BW)\n");
    }
    if (__builtin_cpu_supports("avx512vl")) {
        printf("✅ AVX-512 Vector Length (VL)\n");
    }
    if (__builtin_cpu_supports("avx512cd")) {
        printf("✅ AVX-512 Conflict Detection (CD)\n");
    }
    
    printf("\n=== Testing AVX-512 Instructions ===\n");
    
    // Test basic AVX-512 arithmetic
    __m512i a = _mm512_set1_epi32(10);
    __m512i b = _mm512_set1_epi32(20);
    __m512i c = _mm512_add_epi32(a, b);
    
    printf("Vector A (16x32-bit): ");
    print_m512i(a);
    printf("Vector B (16x32-bit): ");
    print_m512i(b);
    printf("A + B = ");
    print_m512i(c);
    
    // Test gather operation (important for sparse matrices)
    int32_t data[32];
    for(int i = 0; i < 32; i++) data[i] = i * 10;
    
    __m512i indices = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
    __m512i gathered = _mm512_i32gather_epi32(indices, data, 4);
    
    printf("\nGather test result: ");
    print_m512i(gathered);
    
    printf("\n✅ All AVX-512 tests passed!\n");
    return 0;
}
EOF

# Create Rust test for AVX-512
cat > test_avx512.rs << 'EOF'
#![feature(stdsimd)]
use std::arch::x86_64::*;

fn main() {
    if is_x86_feature_detected!("avx512f") {
        println!("✅ AVX-512F detected in Rust");
        
        unsafe {
            // Test AVX-512 operations
            let a = _mm512_set1_epi32(100);
            let b = _mm512_set1_epi32(200);
            let c = _mm512_add_epi32(a, b);
            
            // Extract and verify result
            let result = _mm512_extract_epi32::<0>(c);
            assert_eq!(result, 300);
            println!("AVX-512 arithmetic test passed: 100 + 200 = {}", result);
        }
    } else {
        println!("❌ AVX-512F not available");
    }
}
EOF

# Compile and run C test
echo ""
echo "=== Step 7: Testing AVX-512 Support ==="
gcc -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl test_avx512.c -o test_avx512
./test_avx512

# Clone MAGNUS if not exists
echo ""
echo "=== Step 8: Setting up MAGNUS ==="
cd ~
if [ ! -d "magnus" ]; then
    if [ -d "/Users/eric/compute/magnus" ]; then
        # If running from Mac, copy the project
        echo "Copying MAGNUS from local directory..."
        cp -r /Users/eric/compute/magnus ~/magnus
    else
        echo "Creating new MAGNUS directory..."
        mkdir -p ~/magnus
        echo "Please copy your MAGNUS project to ~/magnus"
    fi
fi

# Create build script
cat > ~/magnus/build-avx512.sh << 'EOF'
#!/bin/bash
# Build MAGNUS with AVX-512 optimizations

echo "Building MAGNUS with AVX-512 optimizations..."

# Set CPU-specific optimizations
export RUSTFLAGS="-C target-cpu=native -C target-feature=+avx512f,+avx512dq,+avx512bw,+avx512vl"

# Build release version
cargo build --release

echo "Build complete!"
echo ""
echo "Run benchmarks with:"
echo "  BENCH_TIER=quick cargo bench --bench tiered_benchmark"
echo "  BENCH_TIER=large cargo bench --bench tiered_benchmark"
EOF

chmod +x ~/magnus/build-avx512.sh

# Create benchmark script
cat > ~/magnus/bench-avx512.sh << 'EOF'
#!/bin/bash
# Benchmark MAGNUS with AVX-512

export RUSTFLAGS="-C target-cpu=native -C target-feature=+avx512f,+avx512dq,+avx512bw,+avx512vl"

echo "=== MAGNUS AVX-512 Benchmarks ==="
echo "CPU: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo "AVX-512: $(lscpu | grep avx512 | head -1)"
echo ""

# Quick sanity check
echo "Running quick benchmark (30s)..."
BENCH_TIER=quick cargo bench --bench tiered_benchmark 2>&1 | grep -E "time:|thrpt:"

# Save results
mkdir -p bench-results
BENCH_TIER=quick cargo bench --bench tiered_benchmark > bench-results/quick-$(date +%Y%m%d-%H%M%S).txt 2>&1

echo ""
echo "Quick benchmark complete. For full benchmarks run:"
echo "  BENCH_TIER=large cargo bench --bench tiered_benchmark"
EOF

chmod +x ~/magnus/bench-avx512.sh

# Final instructions
echo ""
echo "================================================"
echo "  Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. cd ~/magnus"
echo "2. Copy your magnus source files here if needed"
echo "3. Run: ./build-avx512.sh"
echo "4. Run: ./bench-avx512.sh"
echo ""
echo "For development:"
echo "  cd ~/magnus"
echo "  cargo test --test large_matrix_quick --release  # Quick tests"
echo "  BENCH_TIER=quick cargo bench --bench tiered_benchmark  # Quick bench"
echo ""
echo "System info:"
lscpu | grep -E "Model name|Core|Thread|Cache|avx512" | head -10
echo ""
echo "Happy AVX-512 development!"