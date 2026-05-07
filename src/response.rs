//! response.rs — RING: Response recording
//!
//! The ring is what the model produces when excited. We record:
//! - Token distribution entropy (does response spread or concentrate?)
//! - Logprob variance (which tokens are confident vs uncertain?)
//! - Decay rate (how fast the response dissipates across tokens)

use super::probe::Response;

/// Token probability distribution
#[derive(Debug, Clone)]
pub struct TokenDistribution {
    pub tokens: Vec<u32>,
    pub probabilities: Vec<f64>,
    pub entropy: f64,
    pub max_prob: f64,
    pub concentration: f64, // How concentrated vs spread (1 = single token, 0 = uniform)
}

impl TokenDistribution {
    /// Create from response
    pub fn from_response(response: &Response) -> Self {
        let probs = if response.token_probs.is_empty() {
            // Synthetic uniform distribution if no real probs
            let n = response.content.len().max(1);
            vec![1.0 / n as f64; n]
        } else {
            response.token_probs.clone()
        };

        let tokens = response.tokens.clone();
        let entropy = Self::compute_entropy(&probs);
        let max_prob = probs.iter().cloned().fold(0.0_f64, f64::max);
        let concentration = if probs.is_empty() {
            0.0
        } else {
            // Normalize probs before computing concentration
            let sum: f64 = probs.iter().sum();
            let norm_probs: Vec<f64> = probs.iter().map(|p| p / sum).collect();
            let norm_entropy = Self::compute_entropy(&norm_probs);
            let max_entropy = (norm_probs.len() as f64).ln().max(1.0);
            (1.0 - norm_entropy / max_entropy).clamp(0.0, 1.0)
        };

        Self {
            tokens,
            probabilities: probs,
            entropy,
            max_prob,
            concentration,
        }
    }

    /// Compute Shannon entropy H = -Σ p * log(p)
    fn compute_entropy(probs: &[f64]) -> f64 {
        probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.log(std::f64::consts::E))
            .sum()
    }
}

/// Response recorder — extracts metrics from model responses
#[derive(Debug, Clone)]
pub struct ResponseRecorder;

impl ResponseRecorder {
    /// Record full token distribution from response
    pub fn record_distribution(response: &Response) -> TokenDistribution {
        TokenDistribution::from_response(response)
    }

    /// Extract confidence (inverse of logprob variance)
    /// High confidence = low variance = model is certain
    pub fn extract_confidence(response: &Response) -> f64 {
        if response.logprobs.is_empty() {
            return 0.5; // Default medium confidence
        }

        let mean = response.logprobs.iter().sum::<f64>() / response.logprobs.len() as f64;
        let variance = response
            .logprobs
            .iter()
            .map(|&lp| (lp - mean).powi(2))
            .sum::<f64>()
            / response.logprobs.len() as f64;

        // Map variance to confidence: high variance = low confidence
        // Use sigmoid-like mapping to keep in [0, 1]
        let std_dev = variance.sqrt();
        let confidence = 1.0 / (1.0 + std_dev.exp());

        confidence
    }

    /// Measure decay rate from multi-token response
    /// If response has N tokens, decay rate measures how quickly
    /// confidence drops from start to end
    pub fn measure_decay(responses: &[Response]) -> f64 {
        if responses.is_empty() {
            return 0.0;
        }

        // For a single response, look at token-by-token confidence
        if responses.len() == 1 {
            let resp = &responses[0];
            if resp.logprobs.len() < 2 {
                return 0.0;
            }

            // Fit exponential decay: logprob[i] ≈ -decay * i + const
            let n = resp.logprobs.len();
            let sum_i: f64 = (0..n).sum::<usize>() as f64;
            let sum_logp: f64 = resp.logprobs.iter().sum();
            let sum_i_logp: f64 = resp
                .logprobs
                .iter()
                .enumerate()
                .map(|(i, &lp)| i as f64 * lp)
                .sum::<f64>();
            let sum_i_sq: f64 = (0..n).map(|i| (i * i) as f64).sum::<f64>();

            let denom = n as f64 * sum_i_sq - sum_i * sum_i;
            if denom.abs() < 1e-10 {
                return 0.0;
            }

            let slope = (n as f64 * sum_i_logp - sum_i * sum_logp) / denom;
            return -slope; // Negate because logprobs are negative
        }

        // For multiple responses, measure consistency decay
        // High variance across responses = low "ring sustainability"
        let first_logprobs: Vec<f64> = responses
            .iter()
            .filter_map(|r| r.logprobs.first().copied())
            .collect();
        let last_logprobs: Vec<f64> = responses
            .iter()
            .filter_map(|r| r.logprobs.last().copied())
            .collect();

        if first_logprobs.is_empty() || last_logprobs.is_empty() {
            return 0.0;
        }

        let first_mean = first_logprobs.iter().sum::<f64>() / first_logprobs.len() as f64;
        let last_mean = last_logprobs.iter().sum::<f64>() / last_logprobs.len() as f64;

        // Decay = how much confidence is lost from start to end
        (first_mean - last_mean).max(0.0)
    }

    /// Extract activation trajectory (placeholder for when we have hidden states)
    /// Returns None if hidden state access not available (API-only probing)
    pub fn extract_trajectory(response: &Response) -> Option<Vec<f64>> {
        // When using API-only clients, we don't have hidden states
        // But we can approximate from logprob trajectory
        if response.logprobs.is_empty() {
            return None;
        }

        // Cumulative sum as proxy for "activation buildup"
        let mut trajectory = Vec::with_capacity(response.logprobs.len());
        let mut cumsum = 0.0;
        for &lp in &response.logprobs {
            cumsum += lp.abs();
            trajectory.push(cumsum);
        }

        // Normalize
        let max = trajectory.last().copied().unwrap_or(1.0);
        if max > 0.0 {
            for v in &mut trajectory {
                *v /= max;
            }
        }

        Some(trajectory)
    }

    /// Compute KL divergence between two distributions
    pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
        if p.len() != q.len() {
            return f64::INFINITY;
        }

        p.iter()
            .zip(q.iter())
            .map(|(&pi, &qi)| {
                if pi > 0.0 && qi > 0.0 {
                    pi * (pi / qi).ln()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Compute cosine similarity between two response signatures
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_distribution() {
        let resp = Response::synthetic("Hello world".to_string(), Some(42));
        let dist = TokenDistribution::from_response(&resp);

        assert!(dist.entropy > 0.0);
        assert!(dist.max_prob > 0.0);
        assert!(dist.concentration >= 0.0 && dist.concentration <= 1.0);
    }

    #[test]
    fn test_confidence_extraction() {
        let mut resp = Response::synthetic("test".to_string(), None);
        resp.logprobs = vec![-0.1, -0.1, -0.1, -0.1]; // Low variance = high confidence
        let conf = ResponseRecorder::extract_confidence(&resp);
        assert!(conf >= 0.5);

        resp.logprobs = vec![-5.0, 0.0, -10.0, 2.0]; // High variance = low confidence
        let conf_low = ResponseRecorder::extract_confidence(&resp);
        assert!(conf_low < conf);
    }

    #[test]
    fn test_decay_measurement() {
        let resp = Response::synthetic("test response".to_string(), None);
        let decay = ResponseRecorder::measure_decay(&[resp]);
        assert!(decay >= 0.0);
    }

    #[test]
    fn test_kl_divergence() {
        let p = vec![0.5, 0.3, 0.2];
        let q = vec![0.4, 0.4, 0.2];
        let kl = ResponseRecorder::kl_divergence(&p, &q);
        assert!(kl > 0.0);
        assert!(kl.is_finite());
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((ResponseRecorder::cosine_similarity(&a, &b) - 1.0).abs() < 1e-9);

        let c = vec![1.0, 1.0, 1.0];
        let d = vec![-1.0, -1.0, -1.0];
        assert!((ResponseRecorder::cosine_similarity(&c, &d) + 1.0).abs() < 1e-9);
    }
}