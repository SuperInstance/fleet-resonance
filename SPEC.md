# SPEC.md — fleet-resonance Specification

## Overview

`fleet-resonance` is a Rust crate that implements perturbation-response probing for LLM decision graphs. It injects controlled perturbations (taps) into language models, records their responses (rings), extracts resonance signatures, and builds contrast images using the ΔR = R(tap) - R(base) framework.

## Core Data Structures

### Probe

```rust
pub enum Probe {
    PromptProbe { prompt: String, seed: u64 },
    SeedProbe { base_prompt: String, seeds: Vec<u64> },
    AttentionMaskProbe { prompt: String, seed: u64, mask: Vec<usize> },
    TokenMaskProbe { prompt: String, seed: u64, suppress_tokens: Vec<u32> },
}

pub struct ProbeConfig {
    pub temperature: f64,
    pub max_tokens: usize,
    pub top_p: f64,
    pub frequency_penalty: f64,
    pub presence_penalty: f64,
}
```

### Response

```rust
pub struct Response {
    pub content: String,
    pub tokens: Vec<u32>,
    pub logprobs: Vec<f64>,
    pub token_probs: Vec<f64>,
    pub finish_reason: String,
    pub model: String,
    pub seed: Option<u64>,
}

pub struct ResponseBundle {
    pub base: Response,
    pub perturbations: Vec<Response>,
    pub probe_type: ProbeType,
}
```

### ResonanceSignature

```rust
pub struct ResonanceSignature {
    pub frequency_spectrum: Vec<f64>,      // Dominant reasoning frequencies
    pub decay_rate: f64,                    // Response dissipation speed
    pub harmonic_content: Vec<f64>,         // Overtone intensities
    pub impedance: f64,                     // Z = resistance (dead spots = high Z)
    pub entropy: f64,                      // Token distribution spread
    pub logprob_variance: f64,              // Confidence variation
    pub quality_score: f64,                 // Overall resonance quality
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
```

### ContrastMap

```rust
pub struct ContrastMap {
    pub delta_r: Vec<f64>,                  // ΔR = R(tap) - R(base)
    pub hyperperfused: Vec<usize>,         // Amplified pathways
    pub hypoperfused: Vec<usize>,           // Suppressed pathways
    pub dead_spots: Vec<usize>,            // Absorbs, doesn't reradiate
    pub stable_frequencies: Vec<usize>,    // Persist across both
}

pub struct ContrastResult {
    pub base_signature: ResonanceSignature,
    pub tap_signature: ResonanceSignature,
    pub contrast_map: ContrastMap,
    pub similarity_score: f64,
}
```

## Core Functions

### probe.rs — TAP

```rust
/// Trait for LLM models that can be probed
pub trait LLMModel {
    fn generate(&self, prompt: &str, seed: Option<u64>, config: &ProbeConfig) -> Result<Response>;
    fn get_logprobs(&self, prompt: &str, response: &str) -> Result<Vec<f64>>;
}

pub struct ProbeRunner<M: LLMModel> {
    model: M,
    config: ProbeConfig,
}

impl<M: LLMModel> ProbeRunner<M> {
    pub fn new(model: M, config: ProbeConfig) -> Self;
    
    /// Seed-fibulation: hold prompt constant, vary seed
    pub fn seed_fibulation(&self, prompt: &str, seeds: Vec<u64>) -> Result<ResponseBundle>;
    
    /// Prompt-fibulation: hold seed constant, vary prompt
    pub fn prompt_fibulation(&self, prompts: Vec<String>, seed: u64) -> Result<ResponseBundle>;
    
    /// Joint fibulation: vary both seed and prompt
    pub fn joint_fibulation(&self, prompts: Vec<String>, seeds: Vec<u64>) -> Result<Vec<ResponseBundle>>;
    
    /// Squeeze: constrain attention or tokens, then tap
    pub fn squeeze_tap(&self, probe: &Probe) -> Result<Response>;
}
```

### response.rs — RING

```rust
pub struct ResponseRecorder;

impl ResponseRecorder {
    /// Record token distribution from response
    pub fn record_distribution(response: &Response) -> TokenDistribution;
    
    /// Extract logprob variance (confidence across tokens)
    pub fn extract_confidence(response: &Response) -> f64;
    
    /// Measure decay rate from multi-token response
    pub fn measure_decay(responses: &[Response]) -> f64;
    
    /// Extract hidden activation trajectory (requires model support)
    pub fn extract_trajectory(response: &Response) -> Option<Vec<f64>>;
}
```

### resonance.rs — RESONANCE SIGNATURE EXTRACTION

```rust
pub struct ResonanceExtractor;

impl ResonanceExtractor {
    /// Extract full resonance signature from a response bundle
    pub fn extract(bundle: &ResponseBundle) -> ResonanceSignature;
    
    /// Compute frequency spectrum (FFT over token positions)
    pub fn frequency_spectrum(logprobs: &[f64]) -> Vec<f64>;
    
    /// Compute harmonic content (overtones = secondary reasoning paths)
    pub fn harmonic_content(spectrum: &[f64]) -> Vec<f64>;
    
    /// Compute impedance (how much model resists perturbation)
    pub fn impedance(bundle: &ResponseBundle) -> f64;
    
    /// Compute decay rate (how fast response dissipates)
    pub fn decay_rate(responses: &[Response]) -> f64;
    
    /// Compute entropy (spread of token distribution)
    pub fn entropy(token_probs: &[f64]) -> f64;
}
```

### contrast.rs — CONTRAST

```rust
pub struct ContrastEngine;

impl ContrastEngine {
    /// Compute contrast map: ΔR = R(tap) - R(base)
    pub fn contrast(base: &ResonanceSignature, tap: &ResonanceSignature) -> ContrastResult;
    
    /// Identify hyperperfused pathways (amplified by tap)
    pub fn hyperperfused(delta_r: &[f64], threshold: f64) -> Vec<usize>;
    
    /// Identify hypoperfused pathways (suppressed by tap)
    pub fn hypoperfused(delta_r: &[f64], threshold: f64) -> Vec<usize>;
    
    /// Identify dead spots (absorb perturbation, don't reradiate)
    pub fn dead_spots(delta_r: &[f64]) -> Vec<usize>;
    
    /// Compute similarity between two signatures
    pub fn similarity(a: &ResonanceSignature, b: &ResonanceSignature) -> f64;
}
```

### imaging.rs — RESONANCE IMAGING

```rust
pub enum ImageType {
    Frequency,
    Impedance,
    Perfusion,
    Anisotropy,
}

pub struct ResonanceImager;

impl ResonanceImager {
    /// Build ASCII resonance image
    pub fn image(signature: &ResonanceSignature, image_type: ImageType, width: usize, height: usize) -> String;
    
    /// Build frequency map (which reasoning patterns dominate)
    fn frequency_map(signature: &ResonanceSignature, w: usize, h: usize) -> String;
    
    /// Build impedance map (dead spots across the graph)
    fn impedance_map(signature: &ResonanceSignature, w: usize, h: usize) -> String;
    
    /// Build perfusion map (time-series of activation)
    fn perfusion_map(responses: &[Response], w: usize, h: usize) -> String;
    
    /// Build anisotropy map (directional vs diffuse paths)
    fn anisotropy_map(signature: &ResonanceSignature, w: usize, h: usize) -> String;
}
```

### fleet_client.rs — FLEET INTEGRATION

```rust
pub struct FleetClient {
    keeper_url: String,
    http_client: reqwest::Client,
}

impl FleetClient {
    /// Fetch fleet data from keeper API
    pub async fn get_fleet_data(&self, room: &str) -> Result<FleetData>;
    
    /// Write resonance signature to PLATO room
    pub async fn write_signature(&self, room: &str, signature: &ResonanceSignature) -> Result<()>;
    
    /// Fetch contrast maps from PLATO room
    pub async fn get_contrast_maps(&self, room: &str) -> Result<Vec<ContrastResult>>;
}
```

## API Client

### OpenAI-Compatible Client

```rust
pub struct OpenAICompatibleClient {
    base_url: String,
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl LLMModel for OpenAICompatibleClient {
    fn generate(&self, prompt: &str, seed: Option<u64>, config: &ProbeConfig) -> Result<Response>;
}
```

### z.ai GLM Client

```rust
pub struct GLMClient {
    base_url: String,
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl LLMModel for GLMClient {
    fn generate(&self, prompt: &str, seed: Option<u64>, config: &ProbeConfig) -> Result<Response>;
}
```

## CLI Interface

```rust
// main.rs CLI structure
enum Command {
    Probe {
        prompt: String,
        model: String,
        seed: Option<u64>,
        temperature: f64,
    },
    Contrast {
        prompt: String,
        model_a: String,
        model_b: String,
        seeds: usize,
    },
    Image {
        prompt: String,
        model: String,
        seeds: usize,
        image_type: String,
    },
    Compare {
        prompt_a: String,
        prompt_b: String,
        model: String,
    },
}
```

## Tests

### test_probe.rs

- Test probe creation for all probe types
- Test seed-fibulation with known seeds
- Test prompt-fibulation with varying prompts
- Test attention mask application
- Test token suppression

### test_response.rs

- Test response recording (logprobs, tokens)
- Test entropy computation
- Test confidence extraction
- Test decay rate measurement

### test_contrast.rs

- Test contrast map computation
- Test hyperperfused pathway detection (threshold behavior)
- Test hypoperfused pathway detection
- Test dead spot identification (zero response = high Z)
- Test similarity scoring

## Mathematical Framework

### Resonance Frequency

The frequency spectrum is computed via FFT over logprob sequence:

```
F(k) = Σ[n=0..N-1] logprob[n] * exp(-2πi * k * n / N)
```

Dominant frequencies = peaks in |F(k)| — these are the "reasoning modes" that dominate.

### Impedance

```
Z = ||R(tap) - R(base)|| / ||R(base)||
```

High Z = dead spot (response barely changes). Low Z = responsive pathway.

### Contrast Map

```
ΔR[i] = R_tap[i] - R_base[i]
hyperperfused[i] = ΔR[i] > threshold
hypoperfused[i] = ΔR[i] < -threshold
dead_spot[i] = |ΔR[i]| < epsilon
```

### Information Content

Per the MRI/contrast imaging analogy:

```
I(A, B) = B - A = information neither A nor B contains alone
```

## Error Handling

All public functions return `Result<T, anyhow::Error>` for ergonomic error handling. Errors include:

- Network failures (API unavailable)
- Parse errors (malformed API responses)
- Validation errors (invalid probe parameters)
- Model errors (rate limits, auth failures)

## Platform Requirements

- Rust 1.70+
- `cargo` for building
- Network access for API calls
- Optional: PLATO keeper at localhost:8847 for fleet integration