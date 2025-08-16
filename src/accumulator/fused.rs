//! Fused sort+accumulate implementation for high-duplicate scenarios
//!
//! This module provides a specialized implementation that combines sorting
//! and accumulation in a single pass, optimized for cases with >75% duplicates.


/// Fused sort and accumulate for high-duplicate scenarios
/// 
/// This implementation is optimized for cases where we expect >75% duplicates.
/// It uses a modified merge sort that accumulates duplicates during the merge phase.
pub fn fused_sort_accumulate(
    col_indices: Vec<usize>,
    values: Vec<f32>,
) -> (Vec<usize>, Vec<f32>) {
    if col_indices.is_empty() {
        return (Vec::new(), Vec::new());
    }
    
    if col_indices.len() == 1 {
        return (col_indices, values);
    }
    
    // Create paired data for sorting
    let mut pairs: Vec<(usize, f32)> = col_indices.into_iter()
        .zip(values.into_iter())
        .collect();
    
    // Perform merge sort with accumulation
    let result = merge_sort_accumulate(&mut pairs);
    
    // Unzip the result
    let (indices, vals): (Vec<_>, Vec<_>) = result.into_iter().unzip();
    (indices, vals)
}

/// Merge sort with duplicate accumulation
fn merge_sort_accumulate(pairs: &mut [(usize, f32)]) -> Vec<(usize, f32)> {
    let len = pairs.len();
    
    // Base case: small arrays use insertion sort with accumulation
    if len <= 16 {
        return insertion_sort_accumulate(pairs);
    }
    
    // Divide
    let mid = len / 2;
    let left = merge_sort_accumulate(&mut pairs[..mid]);
    let right = merge_sort_accumulate(&mut pairs[mid..]);
    
    // Merge with accumulation
    merge_with_accumulation(left, right)
}

/// Insertion sort with accumulation for small arrays
fn insertion_sort_accumulate(pairs: &mut [(usize, f32)]) -> Vec<(usize, f32)> {
    // First, sort the array
    pairs.sort_unstable_by_key(|&(idx, _)| idx);
    
    // Then accumulate duplicates
    let mut result = Vec::with_capacity(pairs.len() / 4); // Expect high duplicates
    
    let mut current_idx = pairs[0].0;
    let mut current_sum = pairs[0].1;
    
    for &(idx, val) in &pairs[1..] {
        if idx == current_idx {
            current_sum += val;
        } else {
            result.push((current_idx, current_sum));
            current_idx = idx;
            current_sum = val;
        }
    }
    
    result.push((current_idx, current_sum));
    result
}

/// Merge two sorted sequences with duplicate accumulation
fn merge_with_accumulation(
    left: Vec<(usize, f32)>,
    right: Vec<(usize, f32)>,
) -> Vec<(usize, f32)> {
    let mut result = Vec::with_capacity(left.len() + right.len());
    let mut i = 0;
    let mut j = 0;
    
    // Current accumulation state
    let mut current: Option<(usize, f32)> = None;
    
    while i < left.len() || j < right.len() {
        // Select next element from left or right
        let next = if j >= right.len() || (i < left.len() && left[i].0 <= right[j].0) {
            let elem = left[i];
            i += 1;
            elem
        } else {
            let elem = right[j];
            j += 1;
            elem
        };
        
        // Accumulate or output
        match current {
            None => {
                current = Some(next);
            }
            Some((idx, sum)) if idx == next.0 => {
                // Accumulate duplicate
                current = Some((idx, sum + next.1));
            }
            Some(prev) => {
                // Different index, output previous and start new
                result.push(prev);
                current = Some(next);
            }
        }
    }
    
    // Don't forget the last element
    if let Some(last) = current {
        result.push(last);
    }
    
    result
}

/// Generic version for any numeric type
pub fn fused_sort_accumulate_generic<T>(
    col_indices: Vec<usize>,
    values: Vec<T>,
) -> (Vec<usize>, Vec<T>)
where
    T: Copy + std::ops::Add<Output = T>,
{
    if col_indices.is_empty() {
        return (Vec::new(), Vec::new());
    }
    
    // Create pairs and sort
    let mut pairs: Vec<(usize, T)> = col_indices.into_iter()
        .zip(values.into_iter())
        .collect();
    
    pairs.sort_unstable_by_key(|&(idx, _)| idx);
    
    // Accumulate duplicates in a single pass
    let mut result_indices = Vec::with_capacity(pairs.len() / 4); // Expect 75% duplicates
    let mut result_values = Vec::with_capacity(pairs.len() / 4);
    
    let mut current_idx = pairs[0].0;
    let mut current_sum = pairs[0].1;
    
    for &(idx, val) in &pairs[1..] {
        if idx == current_idx {
            current_sum = current_sum + val;
        } else {
            result_indices.push(current_idx);
            result_values.push(current_sum);
            current_idx = idx;
            current_sum = val;
        }
    }
    
    result_indices.push(current_idx);
    result_values.push(current_sum);
    
    (result_indices, result_values)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fused_sort_accumulate_no_duplicates() {
        let indices = vec![3, 1, 4, 2];
        let values = vec![3.0, 1.0, 4.0, 2.0];
        
        let (sorted_idx, sorted_val) = fused_sort_accumulate(indices, values);
        
        assert_eq!(sorted_idx, vec![1, 2, 3, 4]);
        assert_eq!(sorted_val, vec![1.0, 2.0, 3.0, 4.0]);
    }
    
    #[test]
    fn test_fused_sort_accumulate_high_duplicates() {
        // 75% duplicates scenario
        let indices = vec![1, 2, 1, 2, 1, 2, 1, 3];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        
        let (sorted_idx, sorted_val) = fused_sort_accumulate(indices, values);
        
        assert_eq!(sorted_idx, vec![1, 2, 3]);
        assert_eq!(sorted_val, vec![16.0, 12.0, 8.0]); // 1+3+5+7=16, 2+4+6=12
    }
    
    #[test]
    fn test_merge_with_accumulation() {
        let left = vec![(1, 2.0), (3, 4.0), (3, 1.0)];
        let right = vec![(2, 3.0), (3, 5.0), (4, 6.0)];
        
        let result = merge_with_accumulation(left, right);
        
        assert_eq!(result, vec![(1, 2.0), (2, 3.0), (3, 10.0), (4, 6.0)]);
    }
    
    #[test]
    fn test_high_duplicate_performance() {
        // Create scenario with 80% duplicates
        let n = 1000;
        let unique_values = 200;
        
        let mut indices = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);
        
        for i in 0..n {
            indices.push(i % unique_values);
            values.push(i as f32);
        }
        
        // Shuffle to make it realistic
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut combined: Vec<_> = indices.iter().zip(values.iter()).map(|(&i, &v)| (i, v)).collect();
        combined.shuffle(&mut rng);
        
        let (shuffled_indices, shuffled_values): (Vec<_>, Vec<_>) = combined.into_iter().unzip();
        
        let (sorted_idx, sorted_val) = fused_sort_accumulate(shuffled_indices, shuffled_values);
        
        // Should have exactly 200 unique indices
        assert_eq!(sorted_idx.len(), unique_values);
        
        // Should be sorted
        for window in sorted_idx.windows(2) {
            assert!(window[0] < window[1]);
        }
    }
}