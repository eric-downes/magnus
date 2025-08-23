//! Debug test for bitonic sort

#![cfg(target_arch = "x86_64")]

use magnus::accumulator::avx512::*;
use magnus::accumulator::Accumulator;

#[test]
fn debug_bitonic_duplicates() {
    let mut acc = Avx512Accumulator::new(16);
    
    // Add scattered duplicates - exactly the test case that's failing
    acc.accumulate(10, 1.0);
    acc.accumulate(5, 2.0);
    acc.accumulate(15, 3.0);
    acc.accumulate(5, 4.0);
    acc.accumulate(10, 5.0);
    acc.accumulate(15, 6.0);
    
    println!("Input: 6 elements with duplicates at indices 5, 10, 15");
    
    let (indices, values) = acc.extract_result();
    
    println!("Output indices: {:?}", indices);
    println!("Output values: {:?}", values);
    
    // What we expect:
    println!("Expected indices: [5, 10, 15]");
    println!("Expected values: [6.0, 6.0, 9.0]");
    
    // The bitonic sort should sort to: [5, 5, 10, 10, 15, 15]
    // Then accumulation should merge to: [5, 10, 15] with [6.0, 6.0, 9.0]
    
    assert_eq!(indices.len(), 3, "Should have 3 unique indices");
    assert_eq!(indices, vec![5, 10, 15]);
    assert_eq!(values, vec![6.0, 6.0, 9.0]);
}