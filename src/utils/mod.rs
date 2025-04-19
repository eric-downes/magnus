//! Utility functions and helpers

// This is a placeholder for now.
// Will be expanded as needed during implementation.

/// Computes an exclusive prefix sum (scan) for a vector
pub fn exclusive_scan(input: &[usize]) -> Vec<usize> {
    let mut result = Vec::with_capacity(input.len() + 1);
    let mut sum = 0;
    
    result.push(0); // First element is always 0
    
    for &val in input {
        sum += val;
        result.push(sum);
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_exclusive_scan() {
        let input = vec![1, 2, 3, 4];
        let expected = vec![0, 1, 3, 6, 10];
        assert_eq!(exclusive_scan(&input), expected);
        
        let input = vec![0, 0, 5, 0];
        let expected = vec![0, 0, 0, 5, 5];
        assert_eq!(exclusive_scan(&input), expected);
    }
}