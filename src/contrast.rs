//! contrast.rs — CONTRAST: ΔR = R(tap) - R(base)
//!
//! The fundamental equation of contrast imaging:
//! R(base) = baseline response (no perturbation)
//! R(tap)  = response to perturbation
//! ΔR = R(tap) - R(base) = information neither contains alone
//!
//! Returns a ContrastMap with:
//!   - hyperperfused pathways (amplified by tap)
//!   - hypoperfused pathways (suppressed by tap)
//!   - dead spots (absorb perturbation, don't reradiate)
//!   - stable frequencies (persist across both)

use super::resonance::ResonanceSignature;

/// Contrast map — which pathways changed under perturbation
#[derive(Debug, Clone)]
pub struct ContrastMap {
    /// ΔR = R(tap) - R(base) for each dimension
    pub delta_r: Vec<f64>,

    /// Indices of hyperperfused pathways (amplified by tap)
    pub hyperperfused: Vec<usize>,

    /// Indices of hypoperfused pathways (suppressed by tap)
    pub hypoperfused: Vec<usize>,

    /// Indices of dead spots (absorb perturbation, don't reradiate)
    pub dead_spots: Vec<usize>,

    /// Indices of stable frequencies (persist across both)
    pub stable_frequencies: Vec<usize>,
}

impl Default for ContrastMap {
    fn default() -> Self {
        Self {
            delta_r: vec![],
            hyperperfused: vec![],
            hypoperfused: vec![],
            dead_spots: vec![],
            stable_frequencies: vec![],
        }
    }
}

/// Result of a contrast computation
#[derive(Debug, Clone)]
pub struct ContrastResult {
    pub base_signature: ResonanceSignature,
    pub tap_signature: ResonanceSignature,
    pub contrast_map: ContrastMap,
    pub similarity_score: f64,
}

impl ContrastResult {
    /// Get summary statistics
    pub fn summary(&self) -> ContrastSummary {
        ContrastSummary {
            hyperperfused_count: self.contrast_map.hyperperfused.len(),
            hypoperfused_count: self.contrast_map.hypoperfused.len(),
            dead_spot_count: self.contrast_map.dead_spots.len(),
            stable_count: self.contrast_map.stable_frequencies.len(),
            similarity: self.similarity_score,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ContrastSummary {
    pub hyperperfused_count: usize,
    pub hypoperfused_count: usize,
    pub dead_spot_count: usize,
    pub stable_count: usize,
    pub similarity: f64,
}

/// Contrast engine — computes difference maps
#[derive(Debug, Clone)]
pub struct ContrastEngine;

impl ContrastEngine {
    /// Compute contrast map: ΔR = R(tap) - R(base)
    pub fn contrast(base: &ResonanceSignature, tap: &ResonanceSignature) -> ContrastResult {
        // Compute ΔR vector (tap - base)
        let base_vec = base.as_vector();
        let tap_vec = tap.as_vector();

        // Pad to same length
        let len = base_vec.len().max(tap_vec.len());
        let mut delta_r = vec![0.0; len];
        for i in 0..len {
            let b = base_vec.get(i).unwrap_or(&0.0);
            let t = tap_vec.get(i).unwrap_or(&0.0);
            delta_r[i] = t - b;
        }

        // Find hyperperfused (ΔR > threshold — amplified by tap)
        let hyper_threshold = 0.1;
        let hyperperfused: Vec<usize> = delta_r
            .iter()
            .enumerate()
            .filter(|(_, &d)| d > hyper_threshold)
            .map(|(i, _)| i)
            .collect();

        // Find hypoperfused (ΔR < -threshold — suppressed by tap)
        let hypoperfused: Vec<usize> = delta_r
            .iter()
            .enumerate()
            .filter(|(_, &d)| d < -hyper_threshold)
            .map(|(i, _)| i)
            .collect();

        // Find dead spots (|ΔR| <= epsilon — absorbs perturbation)
        let epsilon = 0.01;
        let dead_spots: Vec<usize> = delta_r
            .iter()
            .enumerate()
            .filter(|(_, &d)| d.abs() <= epsilon)
            .map(|(i, _)| i)
            .collect();

        // Find stable frequencies (persist across both)
        // = dimensions where both base and tap are non-zero and similar magnitude
        let stable_threshold = 0.05;
        let stable_frequencies: Vec<usize> = (0..len)
            .filter(|&i| {
                let b = base_vec.get(i).unwrap_or(&0.0);
                let t = tap_vec.get(i).unwrap_or(&0.0);
                b.abs() > stable_threshold && t.abs() > stable_threshold && (b - t).abs() < hyper_threshold
            })
            .collect();

        let contrast_map = ContrastMap {
            delta_r,
            hyperperfused,
            hypoperfused,
            dead_spots,
            stable_frequencies,
        };

        let similarity = Self::similarity(base, tap);

        ContrastResult {
            base_signature: base.clone(),
            tap_signature: tap.clone(),
            contrast_map,
            similarity_score: similarity,
        }
    }

    /// Identify hyperperfused pathways (amplified by tap)
    pub fn hyperperfused(delta_r: &[f64], threshold: f64) -> Vec<usize> {
        delta_r
            .iter()
            .enumerate()
            .filter(|(_, &d)| d > threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Identify hypoperfused pathways (suppressed by tap)
    pub fn hypoperfused(delta_r: &[f64], threshold: f64) -> Vec<usize> {
        delta_r
            .iter()
            .enumerate()
            .filter(|(_, &d)| d < -threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Identify dead spots (absorb perturbation, don't reradiate)
    /// Dead spots have |ΔR| < epsilon
    pub fn dead_spots(delta_r: &[f64]) -> Vec<usize> {
        let epsilon = 0.01;
        delta_r
            .iter()
            .enumerate()
            .filter(|(_, &d)| d.abs() <= epsilon)
            .map(|(i, _)| i)
            .collect()
    }

    /// Compute similarity between two signatures (cosine similarity)
    pub fn similarity(a: &ResonanceSignature, b: &ResonanceSignature) -> f64 {
        let vec_a = a.as_vector();
        let vec_b = b.as_vector();

        let dot = vec_a.iter().zip(vec_b.iter()).map(|(x, y)| x * y).sum::<f64>();
        let norm_a = vec_a.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-10);
        let norm_b = vec_b.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-10);

        dot / (norm_a * norm_b)
    }

    /// Compute weighted contrast (emphasize certain dimensions)
    pub fn weighted_contrast(base: &ResonanceSignature, tap: &ResonanceSignature, weights: &[f64]) -> Vec<f64> {
        let base_vec = base.as_vector();
        let tap_vec = tap.as_vector();
        let len = base_vec.len().max(tap_vec.len());

        (0..len)
            .map(|i| {
                let w = weights.get(i).unwrap_or(&1.0);
                let b = base_vec.get(i).unwrap_or(&0.0);
                let t = tap_vec.get(i).unwrap_or(&0.0);
                w * (t - b)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signature(freq: Vec<f64>, imp: f64, ent: f64) -> ResonanceSignature {
        ResonanceSignature {
            frequency_spectrum: freq,
            decay_rate: 0.1,
            harmonic_content: vec![0.5, 0.3, 0.1, 0.05],
            impedance: imp,
            entropy: ent,
            logprob_variance: 0.3,
            quality_score: 0.7,
        }
    }

    #[test]
    fn test_contrast_basic() {
        let base = make_signature(vec![0.5, 0.3, 0.2], 0.3, 2.0);
        let tap = make_signature(vec![0.6, 0.2, 0.15], 0.2, 2.5);

        let result = ContrastEngine::contrast(&base, &tap);

        assert!(!result.contrast_map.delta_r.is_empty());
        // New tap has higher quality (lower impedance), should show contrast
        assert!(result.similarity_score > 0.0 && result.similarity_score <= 1.0);
    }

    #[test]
    fn test_dead_spot_detection() {
        // Create signatures with similar values (dead spot = no change)
        let base = make_signature(vec![0.5, 0.5, 0.5], 0.5, 2.0);
        let tap = make_signature(vec![0.5, 0.5, 0.5], 0.5, 2.0);

        let result = ContrastEngine::contrast(&base, &tap);

        // Identical signatures = mostly dead spots
        // (similarity = 1.0, delta_r near zero)
        assert!(result.similarity_score > 0.99);
    }

    #[test]
    fn test_hyper_hypo_perfused() {
        let delta = vec![0.5, 0.2, -0.3, -0.05, 0.01, 0.8];

        let hyper = ContrastEngine::hyperperfused(&delta, 0.1);
        let hypo = ContrastEngine::hypoperfused(&delta, 0.1);

        assert_eq!(hyper, vec![0, 1, 5]); // > 0.1
        assert_eq!(hypo, vec![2]); // < -0.1
    }

    #[test]
    fn test_dead_spots() {
        let delta = vec![0.005, 0.2, -0.003, 0.8, 0.009, -0.010];
        let dead = ContrastEngine::dead_spots(&delta);

        // |delta| <= 0.01
        assert!(dead.contains(&0)); // 0.005
        assert!(dead.contains(&2)); // -0.003
        assert!(dead.contains(&4)); // 0.009
        assert!(dead.contains(&5)); // -0.010
        assert!(!dead.contains(&1)); // 0.2
        assert!(!dead.contains(&3)); // 0.8
    }

    #[test]
    fn test_similarity() {
        let sig1 = make_signature(vec![1.0, 0.5, 0.3], 0.2, 1.5);
        let sig2 = make_signature(vec![1.0, 0.5, 0.3], 0.2, 1.5);
        let sig3 = make_signature(vec![0.1, 0.1, 0.1], 0.9, 5.0);

        let sim_identical = ContrastEngine::similarity(&sig1, &sig2);
        let sim_different = ContrastEngine::similarity(&sig1, &sig3);

        // Identical signatures should have similarity close to 1.0
        assert!((sim_identical - 1.0).abs() < 0.001);
        // Very different signatures should have lower similarity
        // Note: fixed fields (decay, entropy, quality) contribute, so don't expect << 1
        assert!(sim_different < sim_identical);
    }

    #[test]
    fn test_contrast_summary() {
        let base = make_signature(vec![0.5, 0.3, 0.2], 0.3, 2.0);
        let tap = make_signature(vec![0.7, 0.1, 0.05], 0.2, 3.0);

        let result = ContrastEngine::contrast(&base, &tap);
        let summary = result.summary();

        assert!(summary.similarity > 0.0);
    }
}