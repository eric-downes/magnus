//! Simple demo comparing AVX512 and generic sort accumulators

use magnus::accumulator::{avx512::Avx512Accumulator, sort::SortAccumulator, Accumulator};
use std::time::Instant;

fn main() {
    println!("AVX512 Accumulator Performance Demo");
    println!("====================================\n");

    // Check AVX512 availability
    println!(
        "AVX512 available: {}",
        magnus::accumulator::avx512::is_avx512_available()
    );
    println!(
        "AVX512CD available: {}\n",
        magnus::accumulator::avx512::is_avx512cd_available()
    );

    // Test different sizes
    let sizes = vec![16, 32, 64, 128, 256, 512, 1024];
    let iterations = 1000;

    for &size in &sizes {
        // Generate test data with some duplicates
        let mut indices = Vec::with_capacity(size);
        let mut values = Vec::with_capacity(size);

        for i in 0..size {
            indices.push(i % (size / 2 + 1)); // Create some duplicates
            values.push((i as f32) * 0.5);
        }

        // Test AVX512 accumulator
        let start = Instant::now();
        for _ in 0..iterations {
            let mut acc = Avx512Accumulator::new(size);
            for (idx, val) in indices.iter().zip(values.iter()) {
                acc.accumulate(*idx, *val);
            }
            let _ = acc.extract_result();
        }
        let avx512_time = start.elapsed();

        // Test generic sort accumulator
        let start = Instant::now();
        for _ in 0..iterations {
            let mut acc = SortAccumulator::new(size);
            for (idx, val) in indices.iter().zip(values.iter()) {
                acc.accumulate(*idx, *val);
            }
            let _ = acc.extract_result();
        }
        let generic_time = start.elapsed();

        // Print results
        let speedup = generic_time.as_secs_f64() / avx512_time.as_secs_f64();
        println!(
            "Size {:4}: AVX512: {:8.3}ms, Generic: {:8.3}ms, Speedup: {:.2}x",
            size,
            avx512_time.as_secs_f64() * 1000.0 / iterations as f64,
            generic_time.as_secs_f64() * 1000.0 / iterations as f64,
            speedup
        );
    }
}
