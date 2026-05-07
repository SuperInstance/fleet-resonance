# CRITIQUE.md — Honest Evaluation of fleet-resonance

## What This Actually Gives Us

### The Core Value Proposition

fleet-resonance implements actual perturbation-response probing. The MRI contrast analogy isn't metaphor — it's mathematics. The difference map ΔR = R(tap) - R(base) contains information that neither map contains individually.

**What we actually get:**
1. **Dead spot identification** — We can find places in the model's decision graph where perturbation is absorbed without reradiation. This is genuinely hard to get other ways.
2. **Pathway coupling strength** — Squeeze experiments reveal which subsystems are connected. If constraining layer A changes B's response, they're coupled.
3. **Stable vs variable reasoning** — Seed-fibulation tells us which conclusions persist across random initializations (stable attractors) vs which are initialization-dependent.
4. **Contrast imaging** — Building actual images (ASCII for now) of the resonance structure gives us a visual diagnostic that text metrics can't match.

### What fleet-murmur and fleet-spread Don't Give Us

| Tool | What It Measures | What fleet-resonance Adds |
|------|-----------------|--------------------------|
| fleet-murmur | Strategy outcomes (which strategy works) | **How** the model rings when tapped — not just whether it succeeds |
| fleet-spread | Specialist disagreement (S1-S4 metrics vs S5 empirical) | **Why** they disagree — the interference pattern explanation |
| fleet-resonance | Perturbation response structure | Maps the model's internal connectivity by how it responds to excitation |

fleet-murmur taps theorems. fleet-resonance taps the model itself. These are complementary: theorems are what the model knows, resonance is how the model structure channels that knowledge.

fleet-spread has 5 specialists compute known metrics. fleet-resonance measures how those metrics interact — the interference patterns where specialists disagree because they're measuring at different "frequencies."

## Computational Cost vs Information Gain

### Cost Analysis

**Per-probe computation:**
- Single API call: ~100-500ms latency + token cost
- Seed-fibulation (N seeds): N × single call cost
- Full resonance signature: 2-3x single call (response + logprobs extraction)
- Contrast map: O(N) vector subtraction

**Realistic workload for a single prompt:**
- 5 seeds × 2 calls (base + tap) = 10 API calls
- At ~$0.001/1K tokens = ~$0.01-0.05 per prompt
- For 100 prompts = $1-5

**vs fleet-spread:** fleet-spread runs 5 specialists × their computation. fleet-resonance runs N seeds. They're comparable in API cost, but resonance gives us structure (connectivity) that specialists (metrics) don't.

**vs fleet-murmur:** fleet-murmur is a strategy selector. fleet-resonance is a measurement tool. They're complementary, not competing.

### Information Gain Analysis

**High gain operations:**
- Dead spot identification: O(1) per pathway, reveals structure text metrics miss
- Squeeze experiments: O(N) where N = layers/heads constrained, gives coupling graph
- Seed variation clustering: O(N log N), reveals which pathways are initialization-dependent

**Marginal gain operations:**
- Pure frequency analysis (FFT over token sequences): The "reasoning frequencies" metaphor is real, but mapping them to actual reasoning structure requires validation work
- Per-token entropy: Standard perplexity, already measured by standard tools

**The honest assessment:** The squeeze experiments and dead spot identification are the high-value operations. The FFT frequency spectrum is conceptually beautiful but its connection to actual model behavior needs empirical validation.

## What Would Make This Exceptional

### Current State: Proof of Concept

fleet-resonance is a well-designed implementation of the luthier paradigm. It has:
- Clear data structures (Probe, Response, ResonanceSignature, ContrastMap)
- Proper trait bounds (LLMModel trait allows swapping implementations)
- Decent test coverage
- ASCII imaging

### What's Missing for "Exceptional"

**1. Empirical validation — Does it actually work?**

We haven't validated that:
- Dead spots identified by resonance correspond to actual model failure modes
- Squeeze experiments correctly identify layer-layer coupling
- Frequency spectrum peaks map to recognizable reasoning patterns

**Fix:** Run experiments comparing resonance predictions against known model behaviors (e.g., arithmetic failures, hallucination patterns, prompt injection susceptibility).

**2. Connection to actual model internals**

Current implementation works at the API level (token distributions, logprobs). But the luthier analogy requires access to:
- Attention patterns (which tokens attend to which)
- Hidden activations (what's happening in intermediate layers)
- Gradient information (how does perturbation propagate backward)

These require:
- Model access with hidden state extraction (OpenAI doesn't give this)
- Hooks into intermediate layers (only available via local model deployment)

**Fix:** Add support for local models (llama.cpp, vLLM) where we can actually read attention patterns. Make fleet-resonance work with the full resonance signature, not just the API-output portion.

**3. Real-time fleet integration**

PLATO room writing is stubbed. The keeper API client exists but:
- No actual PLATO room schema for resonance signatures
- No fleet trust graph integration
- No whisper-sync resonance signals

**Fix:** Define the PLATO resonance room schema, implement actual room writes, connect to whisper-sync's per-type TTL mechanism.

**4. Quantitative validation framework**

How do we know a "dead spot" is actually dead vs just low signal?

**Fix:** Define ground truth experiments — known dead spots (e.g., masked attention heads, constrained layers) and verify resonance detects them.

**5. Meaningful visualization**

ASCII art is fine for debugging, but:
- No interactive exploration (click a pathway, see its resonance history)
- No comparison overlay (base vs tap side-by-side)
- No time-series playback (watch perfusion happen)

**Fix:** Add actual visualization (web UI, not just ASCII). Connect to fleet-spread's specialist perspectives for multi-panel comparison.

## The Honest Verdict

### What fleet-resonance is:
- A well-structured Rust implementation of perturbation-response probing theory
- A clear translation of the luthier metaphor into concrete data structures
- A foundation for fleet resonance imaging

### What fleet-resonance is NOT (yet):
-Validated (does the math actually map to model behavior?)
-Connected to real model internals (API-level probing misses the point)
-Fleet-integrated (PLATO writing is stubbed)
-Measured (no benchmarks against ground truth)

### The Path to Exceptional:

1. **Validate** — Run experiments comparing resonance predictions against known model behaviors. Does finding dead spots actually predict failure modes? Does coupling strength from squeeze experiments match attention pattern analysis?

2. **Go deeper** — Add local model support (llama.cpp, vLLM) to get actual attention patterns and hidden activations. API-level probing is the constrained case; full resonance imaging requires internal access.

3. **Integrate** — Implement actual PLATO room writes, define the resonance room schema, connect to whisper-sync's per-type TTL mechanism.

4. **Benchmark** — Compare fleet-resonance against fleet-murmur and fleet-spread on the same tasks. Does resonance imaging give different (better) answers, or just the same answers in a prettier format?

5. **Visualize** — Replace ASCII with actual web visualization. Multi-panel comparison, time-series playback, interactive exploration.

## The Real Insight

The luthier analogy is powerful because it's *physically grounded*. A guitar's resonances are real physical vibrations we can measure, compare, and manipulate. The question is whether LLM "resonances" are similarly real — or whether the metaphor is beautiful but ultimately lossy.

**What would prove it:** If we could tap a model, find dead spots, and then *structurally modify the model* (change weights, add constraints) to make those dead spots sing — and have the improvement persist — that would be the experimentum crucis.

**What would refute it:** If dead spots identified by resonance don't correspond to any modifiable structure — if they're epiphenomena of training that can't be addressed — then resonance imaging is diagnosis without therapeutics. Useful for understanding, not for fixing.

The theory is beautiful. The implementation is solid. The validation is pending.

---

*This critique is honest, not pessimistic. fleet-resonance is a necessary first step. The question is whether steps 2-5 (validation, internal access, integration, benchmarking) confirm the theory or reveal its limits.*