//! test_response.rs — Tests for response recording

use fleet_resonance::probe::Response;
use fleet_resonance::response::{ResponseRecorder, TokenDistribution};

#[test]
fn test_record_distribution() {
    let resp = Response::synthetic("Hello world".to_string(), Some(42));
    let dist = ResponseRecorder::record_distribution(&resp);

    assert!(dist.entropy >= 0.0);
    assert!(dist.max_prob > 0.0);
    assert!(dist.concentration >= 0.0 && dist.concentration <= 1.0);
}

#[test]
fn test_extract_confidence_high() {
    // Low variance logprobs = high confidence
    let mut resp = Response::synthetic("test".to_string(), None);
    resp.logprobs = vec![-0.1, -0.1, -0.1, -0.1];
    resp.token_probs = vec![0.904, 0.904, 0.904, 0.904];

    let conf = ResponseRecorder::extract_confidence(&resp);
    // With std_dev=0, conf = 1/(1+1) = 0.5
    assert!(conf >= 0.5, "Low variance should give high confidence, got {}", conf);
}

#[test]
fn test_extract_confidence_low() {
    // High variance logprobs = low confidence
    let mut resp = Response::synthetic("test".to_string(), None);
    resp.logprobs = vec![-5.0, 0.0, -10.0, 2.0];
    resp.token_probs = vec![0.006, 1.0, 0.00005, 7.39];

    let conf = ResponseRecorder::extract_confidence(&resp);
    assert!(conf < 0.5, "High variance should give low confidence");
}

#[test]
fn test_measure_decay_single_response() {
    let resp = Response::synthetic("test response".to_string(), None);
    let decay = ResponseRecorder::measure_decay(&[resp]);
    assert!(decay >= 0.0);
}

#[test]
fn test_measure_decay_multiple_responses() {
    let responses = vec![
        Response::synthetic("Response 1".to_string(), Some(1)),
        Response::synthetic("Response 2".to_string(), Some(2)),
        Response::synthetic("Response 3".to_string(), Some(3)),
    ];
    let decay = ResponseRecorder::measure_decay(&responses);
    assert!(decay >= 0.0);
}

#[test]
fn test_extract_trajectory() {
    let mut resp = Response::synthetic("test".to_string(), None);
    resp.logprobs = vec![-1.0, -2.0, -3.0, -4.0];

    let traj = ResponseRecorder::extract_trajectory(&resp);
    assert!(traj.is_some());

    let traj = traj.unwrap();
    assert_eq!(traj.len(), 4);
    // Trajectory should be cumulative and normalized
    assert!(traj.last().copied().unwrap_or(0.0) <= 1.0);
}

#[test]
fn test_kl_divergence_identical() {
    let p = vec![0.5, 0.3, 0.2];
    let kl = ResponseRecorder::kl_divergence(&p, &p);
    assert!(kl.abs() < 0.001, "KL divergence of identical distributions should be ~0");
}

#[test]
fn test_kl_divergence_different() {
    let p = vec![0.5, 0.3, 0.2];
    let q = vec![0.2, 0.2, 0.6];
    let kl = ResponseRecorder::kl_divergence(&p, &q);
    assert!(kl > 0.0, "KL divergence of different distributions should be positive");
    assert!(kl.is_finite());
}

#[test]
fn test_cosine_similarity_identical() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    assert!((ResponseRecorder::cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    assert!((ResponseRecorder::cosine_similarity(&a, &b) - 0.0).abs() < 0.001);
}

#[test]
fn test_cosine_similarity_opposite() {
    let a = vec![1.0, 1.0, 1.0];
    let b = vec![-1.0, -1.0, -1.0];
    assert!((ResponseRecorder::cosine_similarity(&a, &b) - (-1.0)).abs() < 0.001);
}

#[test]
fn test_token_distribution_synthetic() {
    let resp = Response::synthetic("Hello world".to_string(), None);
    let dist = TokenDistribution::from_response(&resp);

    assert!(!dist.tokens.is_empty());
    assert!(dist.entropy > 0.0);
}

#[test]
fn test_token_distribution_empty_logprobs() {
    let resp = Response {
        content: "test".to_string(),
        tokens: vec![],
        logprobs: vec![],
        token_probs: vec![],
        finish_reason: "stop".to_string(),
        model: "test".to_string(),
        seed: None,
    };

    let dist = TokenDistribution::from_response(&resp);
    // Should use fallback for empty logprobs
    assert!(dist.max_prob > 0.0);
}