//! resonance.rs — Resonance signature extraction
//!
//! The resonance signature characterizes how a model "rings" when tapped:
//! - frequency_spectrum: which reasoning patterns dominate
//! - decay_rate: how fast the response dissipates
//! - harmonic_content: overtones — secondary reasoning paths
//! - impedance: how much the model resists perturbation (dead spots = high Z)
//! - entropy: token distribution spread
//! - logprob_variance: confidence variation across tokens

use super::probe::{Response, ResponseBundle};
use super::response::ResponseRecorder;

/// The resonance signature — a fingerprint of how the model responds to a tap
#[derive(Debug, Clone)]
pub struct ResonanceSignature {
    /// Which "frequencies" (reasoning patterns) dominate
    pub frequency_spectrum: Vec<f64>,

    /// How fast the response dissipates
    pub decay_rate: f64,

    /// Overtone intensities — secondary reasoning paths
    pub harmonic_content: Vec<f64>,

    /// Impedance: how much the model resists perturbation
    /// High Z = dead spot (absorbs perturbation, doesn't reradiate)
    /// Low Z = responsive pathway
    pub impedance: f64,

    /// Token distribution entropy
    pub entropy: f64,

    /// Logprob variance across tokens
    pub logprob_variance: f64,

    /// Overall resonance quality (composite score)
    pub quality_score: f64,
}

impl Default for ResonanceSignature {
    fn default() -> Self {
        Self {
            frequency_spectrum: vec![],
            decay_rate: 0.0,
            harmonic_content: vec![],
            impedance: 0.0,
            entropy: 0.0,
            logprob_variance: 0.0,
            quality_score: 0.0,
        }
    }
}

impl ResonanceSignature {
    /// Compute vector representation for comparison
    pub fn as_vector(&self) -> Vec<f64> {
        let mut v = vec![
            self.decay_rate,
            self.impedance,
            self.entropy,
            self.logprob_variance,
            self.quality_score,
        ];
        // Pad frequency spectrum to fixed length
        let freq_len = 8;
        v.extend(
            self.frequency_spectrum
                .iter()
                .take(freq_len)
                .chain(std::iter::repeat(&0.0).take(freq_len.saturating_sub(self.frequency_spectrum.len())))
                .copied(),
        );
        v.extend(
            self.harmonic_content
                .iter()
                .take(4)
                .chain(std::iter::repeat(&0.0).take(4usize.saturating_sub(self.harmonic_content.len())))
                .copied(),
        );
        v
    }
}

/// Resonance extractor — computes signatures from responses
#[derive(Debug, Clone)]
pub struct ResonanceExtractor;

impl ResonanceExtractor {
    /// Extract full resonance signature from a response bundle
    pub fn extract(bundle: &ResponseBundle) -> ResonanceSignature {
        // Extract base response metrics
        let base_entropy = bundle.base.logprobs.iter().map(|&lp| lp.abs()).sum::<f64>()
            / bundle.base.logprobs.len().max(1) as f64;

        // For synthetic/no-logprobs case, use content length as proxy
        let _base_entropy = if bundle.base.logprobs.is_empty() {
            let content_len = bundle.base.content.len().max(1) as f64;
            1.0 / content_len.sqrt()
        } else {
            base_entropy
        };

        // Compute decay from perturbations
        let decay_rate = ResponseRecorder::measure_decay(&bundle.perturbations);

        // Compute variance in responses (consistency across perturbations)
        let variance = Self::compute_response_variance(bundle);

        // Impedance = resistance to change (low variance = high impedance)
        let impedance = if variance > 0.0 {
            1.0 / (1.0 + variance.sqrt())
        } else {
            1.0 // Maximum impedance when no perturbation response
        };

        // Frequency spectrum from logprob sequence
        let spectrum = Self::frequency_spectrum(&bundle.base.logprobs);

        // Harmonic content (overtones)
        let harmonics = Self::harmonic_content(&spectrum);

        // Entropy
        let entropy = if bundle.base.logprobs.is_empty() {
            (bundle.base.content.len() as f64).ln().max(0.0)
        } else {
            -bundle.base.logprobs.iter().map(|&lp| lp.exp() * lp).sum::<f64>()
        };

        // Logprob variance
        let logprob_variance = if bundle.base.logprobs.is_empty() {
            0.5
        } else {
            bundle.base.logprobs.iter().map(|&lp| lp.powi(2)).sum::<f64>()
                / bundle.base.logprobs.len().max(1) as f64
        };

        // Quality score: combination of resonance properties
        // High quality = low impedance (responsive) + moderate entropy (not too spread)
        // + good decay (not too fast, not too slow)
        let quality_score = (1.0 - impedance) * (1.0 - (entropy / 10.0).min(1.0)) * (1.0 - (decay_rate.abs() / 5.0).min(1.0));

        ResonanceSignature {
            frequency_spectrum: spectrum,
            decay_rate,
            harmonic_content: harmonics,
            impedance,
            entropy,
            logprob_variance,
            quality_score,
        }
    }

    /// Extract signature from a single response (synthetic/probed)
    pub fn extract_single(response: &Response) -> ResonanceSignature {
        let spectrum = Self::frequency_spectrum(&response.logprobs);
        let harmonics = Self::harmonic_content(&spectrum);

        let entropy = if response.logprobs.is_empty() {
            (response.content.len() as f64).ln().max(0.0)
        } else {
            -response.logprobs.iter().map(|&lp| lp.exp() * lp).sum::<f64>()
        };

        let variance = if response.logprobs.is_empty() {
            0.5
        } else {
            let mean = response.logprobs.iter().sum::<f64>() / response.logprobs.len() as f64;
            response.logprobs.iter().map(|&lp| (lp - mean).powi(2)).sum::<f64>()
                / response.logprobs.len().max(1) as f64
        };

        ResonanceSignature {
            frequency_spectrum: spectrum,
            decay_rate: 0.0, // Can't compute from single response
            harmonic_content: harmonics,
            impedance: 1.0 / (1.0 + variance.sqrt()),
            entropy,
            logprob_variance: variance,
            quality_score: (1.0 - (entropy / 10.0).min(1.0)) * (1.0 - variance.min(1.0)),
        }
    }

    /// Compute frequency spectrum via FFT over logprob sequence
    pub fn frequency_spectrum(logprobs: &[f64]) -> Vec<f64> {
        if logprobs.is_empty() {
            return vec![0.0; 8];
        }

        // Use simple DFT (not optimized FFT) for clarity
        // In production, use fftw or rustfft
        let n = logprobs.len().max(1).next_power_of_two().min(64);
        let mut spectrum = vec![0.0; n / 2];

        for k in 0..n / 2 {
            let mut real = 0.0;
            let mut imag = 0.0;
            for i in 0..n {
                let angle = -2.0 * std::f64::consts::PI * (k * i) as f64 / n as f64;
                let x = logprobs.get(i).copied().unwrap_or(0.0);
                real += x * angle.cos();
                imag += x * angle.sin();
            }
            spectrum[k] = (real * real + imag * imag).sqrt();
        }

        // Normalize
        let max = spectrum.iter().cloned().fold(0.0_f64, f64::max);
        if max > 0.0 {
            for v in &mut spectrum {
                *v /= max;
            }
        }

        spectrum
    }

    /// Compute harmonic content (overtones = secondary reasoning paths)
    /// Harmonics are spectral peaks at integer multiples of fundamental frequency
    pub fn harmonic_content(spectrum: &[f64]) -> Vec<f64> {
        if spectrum.len() < 4 {
            return vec![0.0; 4];
        }

        // Find fundamental frequency (dominant peak)
        let fundamental_idx = spectrum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Measure energy at harmonic frequencies (2x, 3x, 4x fundamental)
        let harmonics: Vec<f64> = (1..=4)
            .map(|h| {
                let idx = (fundamental_idx * h) % spectrum.len();
                spectrum[idx]
            })
            .collect();

        harmonics
    }

    /// Compute impedance: how much model resists perturbation
    /// Z = ||R(tap) - R(base)|| / ||R(base)||
    /// High Z = dead spot (response barely changes)
    /// Low Z = responsive pathway
    pub fn impedance(base: &Response, tap: &Response) -> f64 {
        let base_vec: Vec<f64> = base.logprobs.iter().map(|&lp| lp.abs()).collect();
        let tap_vec: Vec<f64> = tap.logprobs.iter().map(|&lp| lp.abs()).collect();

        if base_vec.is_empty() || tap_vec.is_empty() {
            return 1.0; // Maximum impedance (no response change detectable)
        }

        let norm_base = base_vec.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-10);
        let norm_diff = base_vec
            .iter()
            .zip(tap_vec.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        // High impedance = small difference = dead spot
        // Low impedance = large difference = responsive
        1.0 - (norm_diff / norm_base).min(1.0)
    }

    /// Compute decay rate: how fast response dissipates
    pub fn decay_rate(responses: &[Response]) -> f64 {
        ResponseRecorder::measure_decay(responses)
    }

    /// Compute entropy of token distribution
    pub fn entropy(token_probs: &[f64]) -> f64 {
        if token_probs.is_empty() {
            return 0.0;
        }
        -token_probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.log(std::f64::consts::E))
            .sum::<f64>()
    }

    /// Compute variance across responses (for consistency measurement)
    fn compute_response_variance(bundle: &ResponseBundle) -> f64 {
        if bundle.perturbations.is_empty() {
            return 0.0;
        }

        // Compare each perturbation to base response
        let distances: Vec<f64> = bundle
            .perturbations
            .iter()
            .map(|pert| {
                ResponseRecorder::cosine_similarity(
                    &bundle.base.logprobs,
                    &pert.logprobs,
                )
            })
            .collect();

        // Variance in similarity = how much response changes across seeds
        if distances.is_empty() {
            return 0.0;
        }

        let mean = distances.iter().sum::<f64>() / distances.len() as f64;
        let variance = distances.iter().map(|&d| (d - mean).powi(2)).sum::<f64>()
            / distances.len() as f64;

        variance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_spectrum() {
        // Simple test: constant logprobs should give DC component only
        let logprobs = vec![-1.0, -1.0, -1.0, -1.0];
        let spectrum = ResonanceExtractor::frequency_spectrum(&logprobs);
        assert!(!spectrum.is_empty());
        // DC component (index 0) should be largest
        assert!(spectrum[0] >= spectrum[1]);
    }

    #[test]
    fn test_harmonic_content() {
        let spectrum = vec![0.1, 0.5, 0.2, 0.8, 0.3, 0.1, 0.05, 0.02];
        let harmonics = ResonanceExtractor::harmonic_content(&spectrum);
        assert_eq!(harmonics.len(), 4);
    }

    #[test]
    fn test_impedance() {
        // Two identical responses -> zero difference -> max impedance
        let base = Response::synthetic("Test".to_string(), Some(1));
        let same = Response::synthetic("Test".to_string(), Some(1));

        // Response with very different logprobs -> high difference -> low impedance
        let mut diff = Response::synthetic("Test".to_string(), Some(2));
        diff.logprobs = vec![-5.0, -5.0, -5.0, -5.0]; // Very different from base

        let z_same = ResonanceExtractor::impedance(&base, &same);
        let z_diff = ResonanceExtractor::impedance(&base, &diff);

        // Same = max impedance (1.0), Different = low impedance
        assert!(z_same > z_diff, "z_same={}, z_diff={}", z_same, z_diff);
    }

    #[test]
    fn test_single_response_signature() {
        let resp = Response::synthetic("Test response".to_string(), None);
        let sig = ResonanceExtractor::extract_single(&resp);

        assert!(sig.entropy > 0.0);
        assert!(sig.quality_score >= 0.0 && sig.quality_score <= 1.0);
    }

    #[test]
    fn test_signature_vector() {
        let sig = ResonanceSignature {
            frequency_spectrum: vec![0.5, 0.3, 0.2],
            decay_rate: 0.1,
            harmonic_content: vec![0.8, 0.2, 0.1, 0.05],
            impedance: 0.3,
            entropy: 2.5,
            logprob_variance: 0.4,
            quality_score: 0.7,
        };

        let v = sig.as_vector();
        assert!(v.len() >= 5); // At least the base fields
    }
}