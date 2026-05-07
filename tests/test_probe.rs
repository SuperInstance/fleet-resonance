//! test_probe.rs — Tests for probe injection

use fleet_resonance::probe::{LLMModel, Probe, ProbeConfig, ProbeRunner, ProbeType, Response};

mod mock_model {
    use super::*;

    pub struct MockModel;

    impl LLMModel for MockModel {
        fn generate(&self, prompt: &str, seed: Option<u64>, _config: &ProbeConfig) -> anyhow::Result<Response> {
            Ok(Response::synthetic(
                format!("Response to '{}' with seed {:?}", prompt, seed),
                seed,
            ))
        }
    }
}

#[test]
fn test_probe_seed_fibulation() {
    use mock_model::MockModel;

    let runner = ProbeRunner::new(MockModel, ProbeConfig::default());
    let bundle = runner
        .seed_fibulation("What is 2+2?", vec![1, 2, 3])
        .expect("seed_fibulation should succeed");

    assert_eq!(bundle.perturbations.len(), 2);
    assert_eq!(bundle.probe_type, ProbeType::SeedProbe);
}

#[test]
fn test_probe_prompt_fibulation() {
    use mock_model::MockModel;

    let runner = ProbeRunner::new(MockModel, ProbeConfig::default());

    let bundle = runner
        .prompt_fibulation(vec!["What is 2+2?".to_string(), "What is 3+3?".to_string()], 42)
        .expect("prompt_fibulation should succeed");

    assert_eq!(bundle.perturbations.len(), 1);
    assert_eq!(bundle.probe_type, ProbeType::PromptProbe);
}

#[test]
fn test_probe_attention_mask() {
    let probe = Probe::AttentionMaskProbe {
        prompt: "Test prompt".to_string(),
        seed: 42,
        mask: vec![0, 1, 2],
    };

    match probe {
        Probe::AttentionMaskProbe { mask, .. } => {
            assert_eq!(mask, vec![0, 1, 2]);
        }
        _ => panic!("Expected AttentionMaskProbe"),
    }
}

#[test]
fn test_probe_token_mask() {
    let probe = Probe::TokenMaskProbe {
        prompt: "Test prompt".to_string(),
        seed: 42,
        suppress_tokens: vec![10, 20, 30],
    };

    match probe {
        Probe::TokenMaskProbe { suppress_tokens, .. } => {
            assert_eq!(suppress_tokens, vec![10, 20, 30]);
        }
        _ => panic!("Expected TokenMaskProbe"),
    }
}

#[test]
fn test_response_synthetic() {
    let resp = Response::synthetic("Hello world".to_string(), Some(42));

    assert_eq!(resp.content, "Hello world");
    assert_eq!(resp.seed, Some(42));
    assert!(!resp.tokens.is_empty());
}

#[test]
fn test_probe_config_default() {
    let config = ProbeConfig::default();

    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.max_tokens, 256);
    assert_eq!(config.top_p, 1.0);
}

#[test]
fn test_probe_joint_fibulation() {
    use mock_model::MockModel;

    let runner = ProbeRunner::new(MockModel, ProbeConfig::default());

    let bundles = runner
        .joint_fibulation(vec!["Q1".to_string(), "Q2".to_string()], vec![1, 2])
        .expect("joint_fibulation should succeed");

    // 2 prompts × 2 seeds = 4 bundles
    assert_eq!(bundles.len(), 4);
}