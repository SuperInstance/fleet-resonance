//! main.rs — CLI entry point for fleet-resonance
//!
//! Usage:
//!   fleet-resonance probe --prompt "..." --model glm-5.1
//!   fleet-resonance contrast --prompt "..." --model-a glm-5.1 --model-b glm-5-turbo --seeds 5
//!   fleet-resonance image --prompt "..." --model glm-5.1 --seeds 10 --image-type frequency

use anyhow::Result;
use clap::{Parser, Subcommand};
use fleet_resonance::{
    contrast::ContrastEngine, imaging::ResonanceImager,
    probe::{LLMModel, OpenAICompatibleClient},
    ImageType, ProbeConfig, ResonanceExtractor, ResonanceSignature,
};

#[derive(Parser)]
#[command(name = "fleet-resonance")]
#[command(about = "Perturbation-response probing for LLM decision graphs — the luthier's hammer")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a probe and extract resonance signature
    Probe {
        /// The prompt to probe
        #[arg(short, long)]
        prompt: String,

        /// Model to use (e.g., glm-5.1, gpt-4)
        #[arg(short, long, default_value = "glm-5.1")]
        model: String,

        /// API base URL
        #[arg(short, long, default_value = "https://z.ai/api/v1")]
        base_url: String,

        /// Temperature
        #[arg(short, long, default_value_t = 0.7)]
        temperature: f64,

        /// Seed for reproducibility
        #[arg(short, long)]
        seed: Option<u64>,

        /// Image type to display
        #[arg(long, default_value = "frequency")]
        image_type: String,
    },

    /// Compare two models or two prompts
    Contrast {
        /// Base prompt
        #[arg(short, long)]
        prompt_a: String,

        /// Comparison prompt (or same prompt for seed variation)
        #[arg(short, long)]
        prompt_b: Option<String>,

        /// Model A
        #[arg(long)]
        model_a: String,

        /// Model B (for comparison)
        #[arg(long)]
        model_b: Option<String>,

        /// Base URL
        #[arg(long, default_value = "https://z.ai/api/v1")]
        base_url: String,

        /// Number of seeds to vary
        #[arg(short, long, default_value_t = 3)]
        seeds: usize,

        /// Temperature
        #[arg(short, long, default_value_t = 0.7)]
        temperature: f64,
    },

    /// Build a resonance image
    Image {
        /// Prompt to image
        #[arg(short, long)]
        prompt: String,

        /// Model to use
        #[arg(short, long, default_value = "glm-5.1")]
        model: String,

        /// Base URL
        #[arg(short, long, default_value = "https://z.ai/api/v1")]
        base_url: String,

        /// Number of seeds for averaging
        #[arg(short, long, default_value_t = 5)]
        seeds: usize,

        /// Image type (frequency, impedance, perfusion, anisotropy)
        #[arg(long, default_value = "frequency")]
        image_type: String,

        /// Width of image
        #[arg(long, default_value_t = 60)]
        width: usize,

        /// Height of image
        #[arg(long, default_value_t = 12)]
        height: usize,
    },

    /// Compare signatures and show difference image
    Compare {
        /// Signature file A
        #[arg(long)]
        sig_a: String,

        /// Signature file B
        #[arg(long)]
        sig_b: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Probe {
            prompt,
            model,
            base_url,
            temperature,
            seed,
            image_type,
        } => run_probe(&prompt, &model, &base_url, temperature, seed, &image_type),

        Commands::Contrast {
            prompt_a,
            prompt_b,
            model_a,
            model_b,
            base_url,
            seeds,
            temperature,
        } => run_contrast(
            &prompt_a,
            prompt_b.as_deref(),
            &model_a,
            model_b.as_deref(),
            &base_url,
            seeds,
            temperature,
        ),

        Commands::Image {
            prompt,
            model,
            base_url,
            seeds,
            image_type,
            width,
            height,
        } => run_image(&prompt, &model, &base_url, seeds, &image_type, width, height),

        Commands::Compare { sig_a, sig_b } => run_compare(&sig_a, &sig_b),
    }
}

fn run_probe(
    prompt: &str,
    model: &str,
    base_url: &str,
    temperature: f64,
    seed: Option<u64>,
    image_type: &str,
) -> Result<()> {
    println!("🔮 fleet-resonance probe");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Prompt: {}", prompt);
    println!("Model: {} @ {}", model, base_url);

    // Create client (would need real API key in practice)
    let api_key = std::env::var("Z_AI_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .unwrap_or_else(|_| "dummy".to_string());

    let client = OpenAICompatibleClient::new(base_url.to_string(), api_key, model.to_string());

    let config = ProbeConfig {
        temperature,
        max_tokens: 256,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
    };

    // Generate response
    println!("\n📡 Tapping model...");
    let seed_val = seed.unwrap_or(42);

    // For demo, use synthetic response if API fails
    let response = match client.generate(prompt, Some(seed_val), &config) {
        Ok(r) => r,
        Err(e) => {
            println!("⚠️  API call failed ({}), using synthetic response for demo", e);
            fleet_resonance::Response::synthetic(
                format!("Synthetic response to '{}' with seed {}", prompt, seed_val),
                Some(seed_val),
            )
        }
    };

    println!("✅ Response: {}", response.content.chars().take(80).collect::<String>());

    // Extract signature
    println!("\n🎸 Extracting resonance signature...");
    let signature = ResonanceExtractor::extract_single(&response);

    // Display imaging
    let img_type = match image_type.to_lowercase().as_str() {
        "impedance" => ImageType::Impedance,
        "perfusion" => ImageType::Perfusion,
        "anisotropy" => ImageType::Anisotropy,
        _ => ImageType::Frequency,
    };

    println!("\n{}", ResonanceImager::full_report(&signature));

    Ok(())
}

fn run_contrast(
    prompt_a: &str,
    prompt_b: Option<&str>,
    model_a: &str,
    model_b: Option<&str>,
    base_url: &str,
    n_seeds: usize,
    temperature: f64,
) -> Result<()> {
    println!("🔮 fleet-resonance contrast");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Create client
    let api_key = std::env::var("Z_AI_API_KEY")
        .or_else(|_| std::env::var("OPENAI_KEY"))
        .unwrap_or_else(|_| "dummy".to_string());

    let client_a = OpenAICompatibleClient::new(base_url.to_string(), api_key.clone(), model_a.to_string());
    let client_b = model_b.map(|m| {
        OpenAICompatibleClient::new(base_url.to_string(), api_key, m.to_string())
    });

    let config = ProbeConfig {
        temperature,
        max_tokens: 256,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
    };

    // Generate base response
    println!("Prompt A: {}", prompt_a);
    let seed_base = 42;

    let response_a = match client_a.generate(prompt_a, Some(seed_base), &config) {
        Ok(r) => r,
        Err(_) => fleet_resonance::Response::synthetic(
            format!("Synthetic A: response to '{}'", prompt_a),
            Some(seed_base),
        ),
    };

    // Generate tap response (different seed or different prompt)
    let tap_prompt = prompt_b.unwrap_or(prompt_a);
    let seed_tap = seed_base + 1;

    println!("Prompt B: {} (seed {})", tap_prompt, seed_tap);

    let response_b = match &client_b {
        Some(c) => c.generate(tap_prompt, Some(seed_tap), &config),
        None => client_a.generate(tap_prompt, Some(seed_tap), &config),
    };

    let response_b = match response_b {
        Ok(r) => r,
        Err(_) => fleet_resonance::Response::synthetic(
            format!("Synthetic B: response to '{}'", tap_prompt),
            Some(seed_tap),
        ),
    };

    // Extract signatures
    let sig_a = ResonanceExtractor::extract_single(&response_a);
    let sig_b = ResonanceExtractor::extract_single(&response_b);

    // Compute contrast
    let result = ContrastEngine::contrast(&sig_a, &sig_b);

    // Display
    println!("\n{}", ResonanceImager::compare_images(&sig_a, &sig_b, ImageType::Frequency, 70));

    println!("\n📊 Contrast Results:");
    println!("  Similarity: {:.3}", result.similarity_score);
    println!("  Hyperperfused: {} pathways", result.contrast_map.hyperperfused.len());
    println!("  Hypoperfused: {} pathways", result.contrast_map.hypoperfused.len());
    println!("  Dead spots: {} spots", result.contrast_map.dead_spots.len());
    println!("  Stable frequencies: {} frequencies", result.contrast_map.stable_frequencies.len());

    Ok(())
}

fn run_image(
    prompt: &str,
    model: &str,
    base_url: &str,
    n_seeds: usize,
    image_type: &str,
    width: usize,
    height: usize,
) -> Result<()> {
    println!("🔮 fleet-resonance image");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Prompt: {}", prompt);
    println!("Seeds: {}", n_seeds);

    let api_key = std::env::var("Z_AI_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .unwrap_or_else(|_| "dummy".to_string());

    let client = OpenAICompatibleClient::new(base_url.to_string(), api_key, model.to_string());

    let config = ProbeConfig {
        temperature: 0.7,
        max_tokens: 256,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
    };

    // Collect responses across seeds
    let mut signatures = Vec::new();

    for seed in 0..n_seeds {
        let resp = match client.generate(prompt, Some(seed as u64), &config) {
            Ok(r) => r,
            Err(_) => fleet_resonance::Response::synthetic(
                format!("Response {} to '{}'", seed, prompt),
                Some(seed as u64),
            ),
        };
        signatures.push(ResonanceExtractor::extract_single(&resp));
    }

    // Average signatures
    let avg_sig = average_signatures(&signatures);

    // Display
    let img_type = match image_type.to_lowercase().as_str() {
        "impedance" => ImageType::Impedance,
        "perfusion" => ImageType::Perfusion,
        "anisotropy" => ImageType::Anisotropy,
        _ => ImageType::Frequency,
    };

    println!("\n{}", ResonanceImager::image(&avg_sig, img_type, width, height));

    Ok(())
}

fn run_compare(sig_a_path: &str, _sig_b_path: &str) -> Result<()> {
    println!("🔮 fleet-resonance compare");

    // In practice, would load from files
    println!("Signature comparison not yet implemented (would load from {})", sig_a_path);
    println!("Compare is a placeholder for future file-based signature comparison.");

    Ok(())
}

fn average_signatures(sigs: &[ResonanceSignature]) -> ResonanceSignature {
    if sigs.is_empty() {
        return ResonanceSignature::default();
    }

    let n = sigs.len() as f64;

    let avg_freq: Vec<f64> = {
        let len = sigs.first().map(|s| s.frequency_spectrum.len()).unwrap_or(0);
        (0..len)
            .map(|i| sigs.iter().map(|s| s.frequency_spectrum.get(i).unwrap_or(&0.0)).sum::<f64>() / n)
            .collect()
    };

    let avg_harmonics: Vec<f64> = {
        let len = sigs.first().map(|s| s.harmonic_content.len()).unwrap_or(0);
        (0..len)
            .map(|i| sigs.iter().map(|s| s.harmonic_content.get(i).unwrap_or(&0.0)).sum::<f64>() / n)
            .collect()
    };

    ResonanceSignature {
        frequency_spectrum: avg_freq,
        decay_rate: sigs.iter().map(|s| s.decay_rate).sum::<f64>() / n,
        harmonic_content: avg_harmonics,
        impedance: sigs.iter().map(|s| s.impedance).sum::<f64>() / n,
        entropy: sigs.iter().map(|s| s.entropy).sum::<f64>() / n,
        logprob_variance: sigs.iter().map(|s| s.logprob_variance).sum::<f64>() / n,
        quality_score: sigs.iter().map(|s| s.quality_score).sum::<f64>() / n,
    }
}