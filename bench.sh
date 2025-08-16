#!/bin/bash

# MAGNUS Benchmark Runner
# 
# Usage: ./bench.sh [tier]
#   tier: quick (default), standard, large, full
#
# Examples:
#   ./bench.sh           # Run quick tier (30s)
#   ./bench.sh large     # Run large matrix benchmarks (5min)
#   ./bench.sh full      # Run everything (10+ min)

TIER=${1:-quick}

echo "=========================================="
echo "  MAGNUS Benchmark Suite"
echo "  Running tier: $TIER"
echo "=========================================="
echo ""

case $TIER in
    quick)
        echo "Quick sanity check (30-60 seconds)..."
        echo "Testing basic functionality and catching regressions."
        echo ""
        BENCH_TIER=quick cargo bench --bench tiered_benchmark
        ;;
    
    standard)
        echo "Standard benchmark suite (2-3 minutes)..."
        echo "Testing normal use cases with matrices up to 1.5M nnz."
        echo ""
        BENCH_TIER=standard cargo bench --bench tiered_benchmark
        ;;
    
    large)
        echo "Large matrix focus (5 minutes)..."
        echo "Testing primary use case with 2-3M+ nnz matrices."
        echo ""
        BENCH_TIER=large cargo bench --bench tiered_benchmark
        ;;
    
    full)
        echo "Full benchmark suite (10+ minutes)..."
        echo "Running all benchmarks including stress tests."
        echo ""
        BENCH_TIER=full cargo bench --bench tiered_benchmark
        ;;
    
    test)
        echo "Running large matrix correctness tests..."
        cargo test --test large_matrix --release
        ;;
    
    test-all)
        echo "Running all tests including memory-intensive ones..."
        cargo test --test large_matrix --release -- --ignored
        ;;
    
    *)
        echo "Unknown tier: $TIER"
        echo "Valid options: quick, standard, large, full, test, test-all"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "  Benchmark complete!"
echo "=========================================="