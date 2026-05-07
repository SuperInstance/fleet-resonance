//! test_contrast.rs — Tests for contrast map computation

use fleet_resonance::contrast::{ContrastEngine, ContrastMap, ContrastResult};
use fleet_resonance::resonance::ResonanceSignature;

fn make_signature(
    freq: Vec<f64>,
    imp: f64,
    ent: f64,
    decay: f64,
    variance: f64,
    quality: f64,
) -> ResonanceSignature {
    ResonanceSignature {
        frequency_spectrum: freq,
        decay_rate: decay,
        harmonic_content: vec![0.5, 0.3, 0.1, 0.05],
        impedance: imp,
        entropy: ent,
        logprob_variance: variance,
        quality_score: quality,
    }
}

#[test]
fn test_contrast_basic() {
    let base = make_signature(vec![0.5, 0.3, 0.2], 0.3, 2.0, 0.1, 0.3, 0.7);
    let tap = make_signature(vec![0.6, 0.2, 0.15], 0.2, 2.5, 0.15, 0.25, 0.8);

    let result = ContrastEngine::contrast(&base, &tap);

    assert!(!result.contrast_map.delta_r.is_empty());
    assert!(result.similarity_score > 0.0);
    assert!(result.similarity_score <= 1.0);
}

#[test]
fn test_contrast_identical_signatures() {
    let sig = make_signature(vec![0.5, 0.3, 0.2], 0.3, 2.0, 0.1, 0.3, 0.7);

    let result = ContrastEngine::contrast(&sig, &sig);

    // Identical signatures should have very high similarity
    assert!(result.similarity_score > 0.99);
    // Delta R should be near zero (mostly dead spots)
    let near_zero = result.contrast_map.delta_r.iter().filter(|&&d| d.abs() < 0.01).count();
    assert!(near_zero > result.contrast_map.delta_r.len() / 2);
}

#[test]
fn test_contrast_dead_spot_detection() {
    // Create signatures with similar values (dead spot)
    let base = make_signature(vec![0.5, 0.5, 0.5], 0.5, 2.0, 0.1, 0.3, 0.5);
    let tap = make_signature(vec![0.5, 0.5, 0.5], 0.5, 2.0, 0.1, 0.3, 0.5);

    let result = ContrastEngine::contrast(&base, &tap);

    // Should have dead spots (low delta_r)
    assert!(!result.contrast_map.dead_spots.is_empty());
}

#[test]
fn test_contrast_hyperperfused() {
    let base = make_signature(vec![0.3, 0.3, 0.3], 0.5, 2.0, 0.1, 0.3, 0.5);
    let tap = make_signature(vec![0.8, 0.8, 0.8], 0.3, 2.5, 0.2, 0.2, 0.8);

    let result = ContrastEngine::contrast(&base, &tap);

    // Should have hyperperfused pathways (high positive delta_r)
    assert!(!result.contrast_map.hyperperfused.is_empty());
}

#[test]
fn test_contrast_hypoperfused() {
    let base = make_signature(vec![0.8, 0.8, 0.8], 0.3, 2.0, 0.2, 0.2, 0.8);
    let tap = make_signature(vec![0.2, 0.2, 0.2], 0.6, 3.0, 0.05, 0.5, 0.3);

    let result = ContrastEngine::contrast(&base, &tap);

    // Should have hypoperfused pathways (high negative delta_r)
    assert!(!result.contrast_map.hypoperfused.is_empty());
}

#[test]
fn test_hyperperfused_threshold() {
    let delta = vec![0.5, 0.2, -0.3, -0.05, 0.01, 0.8, -0.15, 0.3];

    let hyper = ContrastEngine::hyperperfused(&delta, 0.1);
    assert_eq!(hyper, vec![0, 1, 5, 7]); // Values > 0.1

    let hyper_strict = ContrastEngine::hyperperfused(&delta, 0.25);
    assert_eq!(hyper_strict, vec![0, 5, 7]); // Values > 0.25
}

#[test]
fn test_hypoperfused_threshold() {
    let delta = vec![0.5, 0.2, -0.3, -0.05, 0.01, 0.8, -0.15, 0.3];

    let hypo = ContrastEngine::hypoperfused(&delta, 0.1);
    assert_eq!(hypo, vec![2, 6]); // Values < -0.1

    let hypo_strict = ContrastEngine::hypoperfused(&delta, 0.25);
    assert_eq!(hypo_strict, vec![2]); // Values < -0.25
}

#[test]
fn test_dead_spots_epsilon() {
    // Values with absolute value < 0.01 are dead spots
    let delta = vec![0.005, 0.2, -0.003, 0.8, 0.009, 0.008, 0.5, -0.5];
    let dead = ContrastEngine::dead_spots(&delta);

    // |delta| < 0.01
    assert!(dead.contains(&0)); // 0.005
    assert!(dead.contains(&2)); // -0.003
    assert!(dead.contains(&4)); // 0.009
    assert!(dead.contains(&5)); // 0.008
    assert!(!dead.contains(&1)); // 0.2
    assert!(!dead.contains(&3)); // 0.8
}

#[test]
fn test_similarity_identical() {
    let sig1 = make_signature(vec![1.0, 0.5, 0.3], 0.2, 1.5, 0.1, 0.2, 0.8);
    let sig2 = make_signature(vec![1.0, 0.5, 0.3], 0.2, 1.5, 0.1, 0.2, 0.8);

    let sim = ContrastEngine::similarity(&sig1, &sig2);
    assert!((sim - 1.0).abs() < 0.001);
}

#[test]
fn test_similarity_different() {
    let sig1 = make_signature(vec![1.0, 0.5, 0.3], 0.2, 1.5, 0.1, 0.2, 0.8);
    let sig2 = make_signature(vec![0.1, 0.1, 0.1], 0.9, 5.0, 0.5, 0.8, 0.2);

    let sim = ContrastEngine::similarity(&sig1, &sig2);
    // Very different signatures should have lower similarity
    assert!(sim < 0.98, "Very different signatures should have lower similarity, got {}", sim);
}

#[test]
fn test_contrast_summary() {
    let base = make_signature(vec![0.5, 0.3, 0.2], 0.3, 2.0, 0.1, 0.3, 0.7);
    let tap = make_signature(vec![0.7, 0.1, 0.05], 0.2, 3.0, 0.15, 0.25, 0.8);

    let result = ContrastEngine::contrast(&base, &tap);
    let summary = result.summary();

    assert!(summary.similarity > 0.0 && summary.similarity <= 1.0);
}

#[test]
fn test_weighted_contrast() {
    let base = make_signature(vec![0.5, 0.3, 0.2], 0.3, 2.0, 0.1, 0.3, 0.7);
    let tap = make_signature(vec![0.7, 0.1, 0.05], 0.2, 3.0, 0.15, 0.25, 0.8);

    // Weight first dimension heavily
    let weights = vec![10.0, 1.0, 1.0, 1.0, 1.0];
    let weighted = ContrastEngine::weighted_contrast(&base, &tap, &weights);

    // Base decay_rate=0.1, Tap decay_rate=0.15, diff=0.05, weighted=0.5
    assert!(weighted[0].abs() > 0.4, "Weighted first dim should be ~0.5, got {}", weighted[0]);
}

#[test]
fn test_contrast_stable_frequencies() {
    let base = make_signature(vec![0.5, 0.5, 0.5], 0.3, 2.0, 0.1, 0.3, 0.7);
    let tap = make_signature(vec![0.5, 0.6, 0.4], 0.3, 2.1, 0.1, 0.3, 0.7);

    let result = ContrastEngine::contrast(&base, &tap);

    // First dimension (0.5 vs 0.5) should be stable
    assert!(result.contrast_map.stable_frequencies.contains(&0));
}

#[test]
fn test_contrast_map_default() {
    let map = ContrastMap::default();
    assert!(map.delta_r.is_empty());
    assert!(map.hyperperfused.is_empty());
    assert!(map.hypoperfused.is_empty());
    assert!(map.dead_spots.is_empty());
    assert!(map.stable_frequencies.is_empty());
}

#[test]
fn test_contrast_result_clone() {
    let base = make_signature(vec![0.5, 0.3, 0.2], 0.3, 2.0, 0.1, 0.3, 0.7);
    let tap = make_signature(vec![0.7, 0.1, 0.05], 0.2, 3.0, 0.15, 0.25, 0.8);

    let result = ContrastEngine::contrast(&base, &tap);
    let cloned = result.clone();

    assert_eq!(cloned.similarity_score, result.similarity_score);
    assert_eq!(cloned.contrast_map.delta_r.len(), result.contrast_map.delta_r.len());
}