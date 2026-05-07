//! probe.rs — TAP: Perturbation injection
//!
//! Four probe types mirror how a luthier taps an instrument:
//! - PromptProbe: vary tap location (which semantic domain)
//! - SeedProbe: vary tap intensity (which truths persist)
//! - AttentionMaskProbe: squeeze the body (isolate subsystems)
//! - TokenMaskProbe: damp specific strings (suppress pathways)

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Probe type enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum ProbeType {
    PromptProbe,
    SeedProbe,
    AttentionMaskProbe,
    TokenMaskProbe,
}

/// The probe — a controlled perturbation to inject
#[derive(Debug, Clone)]
pub enum Probe {
    /// Vary prompt, fixed seed — which semantic domains does the model resonate with?
    PromptProbe {
        prompts: Vec<String>,
        seed: u64,
    },
    /// Vary seed, fixed prompt — which truths persist across random initializations?
    SeedProbe {
        prompt: String,
        seeds: Vec<u64>,
    },
    /// Squeeze attention — which subsystems are coupled?
    AttentionMaskProbe {
        prompt: String,
        seed: u64,
        mask: Vec<usize>,
    },
    /// Suppress specific tokens — which tokens dampen or amplify pathways?
    TokenMaskProbe {
        prompt: String,
        seed: u64,
        suppress_tokens: Vec<u32>,
    },
}

/// Configuration for probe execution
#[derive(Debug, Clone)]
pub struct ProbeConfig {
    pub temperature: f64,
    pub max_tokens: usize,
    pub top_p: f64,
    pub frequency_penalty: f64,
    pub presence_penalty: f64,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            max_tokens: 256,
            top_p: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

/// The model's response to a probe
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub content: String,
    pub tokens: Vec<u32>,
    pub logprobs: Vec<f64>,
    pub token_probs: Vec<f64>,
    pub finish_reason: String,
    pub model: String,
    pub seed: Option<u64>,
}

impl Response {
    /// Create a synthetic response for testing
    pub fn synthetic(content: String, seed: Option<u64>) -> Self {
        let tokens: Vec<u32> = content.bytes().map(|b| b as u32).collect();
        let logprobs: Vec<f64> = (0..tokens.len())
            .map(|i| -((i as f64 + 1.0).ln()))
            .collect();
        let token_probs: Vec<f64> = logprobs.iter().map(|lp| lp.exp()).collect();

        Self {
            content,
            tokens,
            logprobs,
            token_probs,
            finish_reason: "stop".to_string(),
            model: "synthetic".to_string(),
            seed,
        }
    }
}

/// Bundle of responses from a fibulation experiment
#[derive(Debug, Clone)]
pub struct ResponseBundle {
    pub base: Response,
    pub perturbations: Vec<Response>,
    pub probe_type: ProbeType,
}

/// Trait for LLM models that can be probed
pub trait LLMModel: Send + Sync {
    fn generate(&self, prompt: &str, seed: Option<u64>, config: &ProbeConfig) -> Result<Response>;
    fn get_logprobs(&self, prompt: &str, response: &str) -> Result<Vec<f64>> {
        let _ = (prompt, response);
        Ok(vec![])
    }
}

/// OpenAI-compatible API client
/// Note: Requires running with a compatible runtime or using blocking wrapper
pub struct OpenAICompatibleClient {
    base_url: String,
    api_key: String,
    model: String,
}

impl OpenAICompatibleClient {
    pub fn new(base_url: String, api_key: String, model: String) -> Self {
        Self {
            base_url,
            api_key,
            model,
        }
    }

    /// Make a blocking API call (use with tokio::runtime::Builder::build().unwrap().block_on)
    pub fn blocking_generate(
        &self,
        prompt: &str,
        seed: Option<u64>,
        config: &ProbeConfig,
    ) -> Result<Response> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to create HTTP client: {}", e))?;

        let request = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "seed": seed,
            "logprobs": true,
            "top_logprobs": 10,
        });

        let resp = client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .map_err(|e| anyhow::anyhow!("Request failed: {}", e))?;

        let chat_resp: serde_json::Value = resp
            .json()
            .map_err(|e| anyhow::anyhow!("Failed to parse response: {}", e))?;

        let content = chat_resp["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let finish_reason = chat_resp["choices"][0]["finish_reason"]
            .as_str()
            .unwrap_or("stop")
            .to_string();

        let model_name = chat_resp["model"]
            .as_str()
            .unwrap_or(&self.model)
            .to_string();

        // Extract logprobs if available
        let (logprobs, token_probs) = if let Some(lps) = chat_resp["choices"][0]["logprobs"].as_array() {
            let lps: Vec<f64> = lps
                .iter()
                .filter_map(|lp| lp["logprob"].as_f64())
                .collect();
            let probs: Vec<f64> = lps.iter().map(|&lp| lp.exp()).collect();
            (lps, probs)
        } else {
            (vec![], vec![])
        };

        Ok(Response {
            content,
            tokens: vec![],
            logprobs,
            token_probs,
            finish_reason,
            model: model_name,
            seed,
        })
    }
}

impl LLMModel for OpenAICompatibleClient {
    fn generate(&self, prompt: &str, seed: Option<u64>, config: &ProbeConfig) -> Result<Response> {
        self.blocking_generate(prompt, seed, config)
    }
}

/// Probe runner — executes probes against an LLM model
pub struct ProbeRunner<M: LLMModel> {
    model: Arc<M>,
    config: ProbeConfig,
}

impl<M: LLMModel> ProbeRunner<M> {
    pub fn new(model: M, config: ProbeConfig) -> Self {
        Self {
            model: Arc::new(model),
            config,
        }
    }

    pub fn seed_fibulation(&self, prompt: &str, seeds: Vec<u64>) -> Result<ResponseBundle> {
        let mut responses = Vec::with_capacity(seeds.len());

        for seed in &seeds {
            let resp = self.model.generate(prompt, Some(*seed), &self.config)?;
            responses.push(resp);
        }

        let base = responses.remove(0);

        Ok(ResponseBundle {
            base,
            perturbations: responses,
            probe_type: ProbeType::SeedProbe,
        })
    }

    pub fn prompt_fibulation(&self, prompts: Vec<String>, seed: u64) -> Result<ResponseBundle> {
        let mut responses = Vec::with_capacity(prompts.len());

        for prompt in &prompts {
            let resp = self.model.generate(prompt, Some(seed), &self.config)?;
            responses.push(resp);
        }

        let base = responses.remove(0);

        Ok(ResponseBundle {
            base,
            perturbations: responses,
            probe_type: ProbeType::PromptProbe,
        })
    }

    pub fn joint_fibulation(
        &self,
        prompts: Vec<String>,
        seeds: Vec<u64>,
    ) -> Result<Vec<ResponseBundle>> {
        let mut bundles = Vec::new();

        for prompt in &prompts {
            for seed in &seeds {
                let resp = self.model.generate(prompt, Some(*seed), &self.config)?;
                bundles.push(ResponseBundle {
                    base: resp,
                    perturbations: vec![],
                    probe_type: ProbeType::SeedProbe,
                });
            }
        }

        Ok(bundles)
    }

    pub fn squeeze_tap(&self, probe: &Probe) -> Result<Response> {
        match probe {
            Probe::AttentionMaskProbe { prompt, seed, .. } => {
                self.model.generate(prompt, Some(*seed), &self.config)
            }
            Probe::TokenMaskProbe { prompt, seed, .. } => {
                self.model.generate(prompt, Some(*seed), &self.config)
            }
            Probe::SeedProbe { prompt, seeds } => {
                let first_seed = seeds.first().copied().unwrap_or(0);
                self.model.generate(prompt.as_str(), Some(first_seed), &self.config)
            }
            Probe::PromptProbe { prompts, seed } => {
                let first_prompt = prompts.first().map(|s| s.as_str()).unwrap_or("");
                self.model.generate(first_prompt, Some(*seed), &self.config)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockModel;

    impl LLMModel for MockModel {
        fn generate(&self, prompt: &str, seed: Option<u64>, _config: &ProbeConfig) -> Result<Response> {
            Ok(Response::synthetic(
                format!("Response to '{}' with seed {:?}", prompt, seed),
                seed,
            ))
        }
    }

    #[test]
    fn test_seed_fibulation() {
        let runner = ProbeRunner::new(MockModel, ProbeConfig::default());
        let bundle = runner.seed_fibulation("What is 2+2?", vec![1, 2, 3]).unwrap();
        assert_eq!(bundle.perturbations.len(), 2);
        assert_eq!(bundle.probe_type, ProbeType::SeedProbe);
    }

    #[test]
    fn test_prompt_fibulation() {
        let runner = ProbeRunner::new(MockModel, ProbeConfig::default());
        let bundle = runner
            .prompt_fibulation(
                vec!["What is 2+2?".to_string(), "What is 3+3?".to_string()],
                42,
            )
            .unwrap();
        assert_eq!(bundle.perturbations.len(), 1);
        assert_eq!(bundle.probe_type, ProbeType::PromptProbe);
    }

    #[test]
    fn test_synthetic_response() {
        let resp = Response::synthetic("Hello world".to_string(), Some(42));
        assert_eq!(resp.content, "Hello world");
        assert_eq!(resp.seed, Some(42));
        assert!(!resp.logprobs.is_empty());
    }
}