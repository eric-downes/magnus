#!/bin/bash

# MAGNUS Unified Test & Benchmark Script v2
# Integrates Buckingham π testing with traditional benchmarks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run tests for a tier
run_tier_tests() {
    local tier=$1
    print_info "Running $tier tier tests..."
    
    export TEST_TIER=$tier
    
    case $tier in
        quick)
            # Quick tests - under 30 seconds
            print_info "Running critical π configurations (5 configs)..."
            cargo test --lib test_tiers --release
            cargo test --test large_matrix_quick --release
            cargo run --bin explore_dimensions -- benchmark 2>/dev/null | head -20
            ;;
            
        commit)
            # Commit tests - under 10 minutes
            print_info "Running smart π sample (50 configs)..."
            cargo test --lib --release
            cargo test --test large_matrix_quick --release
            BENCH_TIER=commit cargo bench --bench tiered_benchmark --no-fail-fast
            ;;
            
        pr)
            # PR tests - under 30 minutes  
            print_info "Running extended π sample (500 configs)..."
            cargo test --lib --release
            cargo test --test large_matrix --release
            BENCH_TIER=pr cargo bench --bench tiered_benchmark
            BENCH_TIER=pr cargo bench --bench comparison_benchmark
            ;;
            
        release)
            # Release tests - under 2 hours
            print_info "Running full π exploration (2,673 configs)..."
            cargo test --all --release
            cargo test --test large_matrix --release -- --ignored
            BENCH_TIER=release cargo bench
            cargo run --bin explore_dimensions -- full
            ;;
    esac
    
    print_success "$tier tier tests completed"
}

# Function to run benchmarks for a tier
run_tier_bench() {
    local tier=$1
    print_info "Running $tier tier benchmarks..."
    
    export BENCH_TIER=$tier
    
    case $tier in
        quick)
            # Quick benchmark - 30 seconds
            cargo bench --bench quick_performance
            cargo bench --bench tiered_benchmark -- quick_comparison
            ;;
            
        commit)
            # Commit benchmark - replaces "standard"
            cargo bench --bench tiered_benchmark
            cargo bench --bench comparison_benchmark -- medium
            ;;
            
        pr)
            # PR benchmark - between commit and release
            cargo bench --bench large_matrix_bench
            cargo bench --bench comparison_benchmark
            cargo bench --bench tiered_benchmark -- large
            ;;
            
        release)
            # Release benchmark - everything
            cargo bench --all
            ;;
    esac
    
    print_success "$tier tier benchmarks completed"
}

# Function to show tier info
show_tier_info() {
    cat << EOF
MAGNUS Test & Benchmark Tiers v2
=================================

Tier System (Integrated with Buckingham π Testing):

┌─────────┬──────────┬─────────────┬──────────────┬─────────────────┐
│  Tier   │   Time   │  π Configs  │ Matrix Sizes │    Use Case     │
├─────────┼──────────┼─────────────┼──────────────┼─────────────────┤
│ quick   │    30s   │      5      │   0.5-5K     │ TDD, pre-commit │
│ commit  │   10min  │     50      │   1-15K      │ Every commit    │
│ pr      │   30min  │    500      │   5-50K      │ Pull requests   │
│ release │    2hr   │   2,673     │  10-200K     │ Release/tags    │
└─────────┴──────────┴─────────────┴──────────────┴─────────────────┘

Commands:
  quick    - Run quick tier (30s) - 5 critical π configs
  commit   - Run commit tier (10min) - 50 smart π configs  
  pr       - Run PR tier (30min) - 500 interesting π configs
  release  - Run release tier (2hr) - 2,673 full π exploration
  
  test-quick    - Tests only for quick tier
  test-commit   - Tests only for commit tier
  test-pr       - Tests only for PR tier
  test-release  - Tests only for release tier
  
  bench-quick   - Benchmarks only for quick tier
  bench-commit  - Benchmarks only for commit tier
  bench-pr      - Benchmarks only for PR tier
  bench-release - Benchmarks only for release tier
  
  info     - Show this help message
  status   - Show current tier configuration

Environment Variables:
  TEST_TIER   - Set default test tier (quick|commit|pr|release)
  BENCH_TIER  - Set default benchmark tier (quick|commit|pr|release)

Examples:
  ./bench_v2.sh quick           # TDD - rapid feedback
  ./bench_v2.sh commit          # Before committing
  ./bench_v2.sh pr              # Before merging PR
  ./bench_v2.sh release         # Before release

EOF
}

# Function to show current status
show_status() {
    print_info "Current Configuration:"
    echo "  TEST_TIER:  ${TEST_TIER:-not set (defaults to quick)}"
    echo "  BENCH_TIER: ${BENCH_TIER:-not set (defaults to quick)}"
    echo ""
    
    # Try to run a quick Rust check to show π config counts
    if command -v cargo &> /dev/null; then
        cargo run --bin explore_dimensions -- demo 2>/dev/null | grep -A 5 "Critical π-Configurations" || true
    fi
}

# Main command handling
case "${1:-info}" in
    # Full tier runs (test + bench)
    quick)
        run_tier_tests quick
        run_tier_bench quick
        ;;
    commit)
        run_tier_tests commit
        run_tier_bench commit
        ;;
    pr)
        run_tier_tests pr
        run_tier_bench pr
        ;;
    release)
        run_tier_tests release
        run_tier_bench release
        ;;
        
    # Test-only commands
    test-quick)
        run_tier_tests quick
        ;;
    test-commit)
        run_tier_tests commit
        ;;
    test-pr)
        run_tier_tests pr
        ;;
    test-release)
        run_tier_tests release
        ;;
        
    # Bench-only commands
    bench-quick)
        run_tier_bench quick
        ;;
    bench-commit)
        run_tier_bench commit
        ;;
    bench-pr)
        run_tier_bench pr
        ;;
    bench-release)
        run_tier_bench release
        ;;
        
    # Info commands
    info|help)
        show_tier_info
        ;;
    status)
        show_status
        ;;
        
    # Legacy compatibility
    test)
        print_warning "Legacy 'test' command mapped to 'quick'"
        run_tier_tests quick
        ;;
    standard)
        print_warning "Legacy 'standard' command mapped to 'commit'"
        run_tier_tests commit
        run_tier_bench commit
        ;;
    large)
        print_warning "Legacy 'large' command mapped to 'pr'"
        run_tier_tests pr
        run_tier_bench pr
        ;;
    full)
        print_warning "Legacy 'full' command mapped to 'release'"
        run_tier_tests release
        run_tier_bench release
        ;;
        
    *)
        print_error "Unknown command: $1"
        echo ""
        show_tier_info
        exit 1
        ;;
esac

print_success "All operations completed successfully!"