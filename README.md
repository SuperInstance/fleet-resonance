# fleet-resonance

[![CI](https://github.com/SuperInstance/fleet-resonance/actions/workflows/ci.yml/badge.svg)](https://github.com/SuperInstance/fleet-resonance/actions/workflows/ci.yml)

**The Luthier's Hammer for AI Systems**

Inject controlled perturbations into language models, record how they ring, extract resonance signatures, and build contrast images. Inspired by how master luthiers tap instruments to read their internal structure from the sound they produce.

## The Core Idea

A master luthier doesn't measure a guitar — they *tap it* and *listen*. The color of that ring reveals structural properties no numerical measurement can capture. **fleet-resonance** applies this same paradigm to LLM decision graphs:

- **TAP** — inject controlled perturbation (vary prompt, seed, attention mask)
- **RING** — record the response (token distributions, logprobs, hidden activations)
- **CONTRAST** — compare signatures to reveal structure neither contains alone

## The Three Operations

```
┌─────────────────────────────────────────────────────────────┐
│  TAP → RING → RESONANCE SIGNATURE → CONTRAST → IMAGE       │
└─────────────────────────────────────────────────────────────┘
```

### 1. TAP — Perturbation Injection

Four probe types mirror how a luthier taps an instrument:

| Probe Type | Luthier Analogy | What It Reveals |
|------------|-----------------|-----------------|
| `PromptProbe` | Vary tap location | Which semantic domains the model resonates with |
| `SeedProbe` | Vary tap intensity | Which truths persist across random initializations |
| `AttentionMaskProbe` | Squeeze the body | Which subsystems are coupled — isolate dead spots |
| `TokenMaskProbe` | Damp specific strings | Which tokens suppress or amplify pathways |

### 2. RING — Record Response

The ring is what the model produces when excited:

```rust
pub struct ResonanceSignature {
    pub frequency_spectrum: Vec<f64>,     // Which reasoning patterns dominate
    pub decay_rate: f64,                   // How fast response dissipates
    pub harmonic_content: Vec<f64>,        // Overtones — secondary reasoning paths
    pub impedance: f64,                    // Resistance to perturbation (dead spots = high Z)
    pub entropy: f64,                      // Token distribution spread
    pub logprob_variance: f64,             // Confidence variation across tokens
}
```

### 3. CONTRAST — ΔR = Information Neither Contains Alone

```rust
pub fn contrast(base: &ResonanceSignature, tap: &ResonanceSignature) -> ContrastMap {
    // ΔR = R(tap) - R(base)
    // Returns:
    //   - hyperperfused pathways (amplified by tap)
    //   - hypoperfused pathways (suppressed by tap)
    //   - dead spots (absorb perturbation, don't reradiate)
    //   - stable frequencies (persist across both)
}
```

## Resonance Imaging

Build visual representations of the model's decision graph:

```
Frequency Map:     Impedance Map:     Perfusion Map:     Anisotropy Map:
                                                         
░░▓▓▓░░░░▓▓░░    ▓▓▓▓░░░░▓▓▓▓    ▄▄▓▓▓▄▄▓▓▄▄    ↗↗↗↗→→→→→→
░▓▓▓▓▓░░▓▓▓░░    ▓▓▓░░░░▓▓▓░░    ▄▓▓▓▓▓▓▓▓▄▄    ↓↓↓↓↓↓↓↓↓↓
▓▓▓▓▓▓▓▓▓▓▓░    ▓░░░░░░░░░▓▓    ▓▓▓▓▓▓▓▓▓▓▓    ←←←←←←←←←←
```

- **Frequency map** — which reasoning patterns dominate (bright = active)
- **Impedance map** — dead spots across the graph (bright = dead/absorbs)
- **Perfusion map** — time-series of activation flowing through pathways
- **Anisotropy map** — directional vs diffuse reasoning paths

## Quick Start

```bash
# Install
cargo build --release

# Run a basic resonance probe
./target/release/fleet-resonance probe --prompt "Explain quantum entanglement" --model glm-5.1

# Compare two models (contrast mode)
./target/release/fleet-resonance contrast \
    --prompt "What is 2+2?" \
    --model-a glm-5.1 \
    --model-b glm-5-turbo \
    --seeds 5

# Build a full resonance image (ASCII visualization)
./target/release/fleet-resonance image --prompt "Describe a boat at sea" --seeds 10 --image-type frequency
```

## Architecture

```
src/
├── lib.rs              — Library exports
├── main.rs             — CLI entry point
├── probe.rs            — TAP: perturbation injection
├── response.rs        — RING: response recording
├── resonance.rs        — Resonance signature extraction
├── contrast.rs         — CONTRAST: ΔR comparison and mapping
├── imaging.rs          — ASCII resonance image generation
└── fleet_client.rs    — Connect to keeper API for fleet data

tests/
├── test_probe.rs
├── test_response.rs
└── test_contrast.rs
```

## How It Works

### 1. Seed-Fibulation (vary seed, fixed prompt)

Hold prompt constant, vary seed systematically. Reveals which truths persist across random initializations — the stable attractors in the model's reasoning manifold.

### 2. Prompt-Fibulation (vary prompt, fixed seed)

Hold seed constant, vary prompts. Reveals which questions the fixed initialization can answer well — the model's natural resonance modes.

### 3. Joint Fibulation (vary both)

Map the full manifold of reasoning paths. Points (seed, prompt) that produce similar signatures are close in the model's internal representation.

### 4. Squeeze (attention/token masking)

Constrain one part while tapping elsewhere. Reveals coupling strength between subsystems — if squeezing A changes B's response, they're connected.

## The Fundamental Equation

```
R(base) = baseline response (no perturbation)
R(tap)  = response to perturbation

ΔR = R(tap) - R(base) = information neither contains alone
```

This is the MRI contrast equation, the seismic interferometry equation, the differential gene expression equation — and the LLM resonance imaging equation. **Comparison is the only way to see inside a system without disassembling it.**

## Connection to Fleet Infrastructure

- **fleet-murmur** — strategy layer that taps the theorem library
- **fleet-spread** — 5 specialist dimensions tap simultaneously
- **whisper-sync** — resonance signals between agents
- **PLATO** — the room where resonance is recorded
- **fleet-resonance** — the imaging hardware

## Comparison vs fleet-murmur and fleet-spread

| Tool | What It Does | fleet-resonance Addition |
|------|-------------|--------------------------|
| fleet-murmur | Strategies tap theorems | We tap the *model itself*, not just theorems |
| fleet-spread | 5 specialist perspectives | We measure *how they interfere* (difference maps) |
| fleet-resonance | — | **Actual perturbation-response measurement with contrast imaging** |

fleet-resonance is what makes the luthier analogy real: not metaphor, but actual perturbation-response probing with comparison-as-information mathematics.

## License

MIT