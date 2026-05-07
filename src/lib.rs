//! fleet-resonance: Perturbation-response probing for LLM decision graphs
//!
//! The "luthier's hammer" — inject controlled perturbations into language models,
//! record how they ring, extract resonance signatures, and build contrast images.

pub mod probe;
pub mod response;
pub mod resonance;
pub mod contrast;
pub mod imaging;
pub mod fleet_client;

// Re-export main types
pub use probe::{Probe, ProbeConfig, ProbeRunner, ProbeType, Response, ResponseBundle, LLMModel};
pub use response::{ResponseRecorder, TokenDistribution};
pub use resonance::{ResonanceExtractor, ResonanceSignature};
pub use contrast::{ContrastEngine, ContrastMap, ContrastResult};
pub use imaging::{ImageType, ResonanceImager};
pub use fleet_client::FleetClient;

// OpenAI-compatible client
pub mod client {
    pub use super::probe::OpenAICompatibleClient;
}