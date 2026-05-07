//! imaging.rs — ASCII resonance image generation
//!
//! Build visual representations of the model's decision graph:
//! - Frequency map: which reasoning patterns dominate
//! - Impedance map: dead spots across the graph
//! - Perfusion map: time-series of activation flowing through pathways
//! - Anisotropy map: directional vs diffuse reasoning paths

use super::probe::Response;
use super::resonance::ResonanceSignature;

/// Image type for resonance visualization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImageType {
    Frequency,
    Impedance,
    Perfusion,
    Anisotropy,
}

impl std::fmt::Display for ImageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImageType::Frequency => write!(f, "frequency"),
            ImageType::Impedance => write!(f, "impedance"),
            ImageType::Perfusion => write!(f, "perfusion"),
            ImageType::Anisotropy => write!(f, "anisotropy"),
        }
    }
}

/// Resonance imager — generates ASCII visualizations
#[derive(Debug, Clone)]
pub struct ResonanceImager;

impl ResonanceImager {
    /// Build ASCII resonance image
    pub fn image(
        signature: &ResonanceSignature,
        image_type: ImageType,
        width: usize,
        height: usize,
    ) -> String {
        let w = width.max(20).min(80);
        let h = height.max(10).min(30);

        match image_type {
            ImageType::Frequency => Self::frequency_map(signature, w, h),
            ImageType::Impedance => Self::impedance_map(signature, w, h),
            ImageType::Perfusion => Self::perfusion_map(signature, w, h),
            ImageType::Anisotropy => Self::anisotropy_map(signature, w, h),
        }
    }

    /// Build frequency map — which reasoning patterns dominate
    /// Uses block characters: ░ (low) ▒ (mid) ▓ (high) ▓ (peak)
    fn frequency_map(signature: &ResonanceSignature, w: usize, h: usize) -> String {
        let spectrum = &signature.frequency_spectrum;
        if spectrum.is_empty() {
            return "No frequency data available.".to_string();
        }

        let mut lines = Vec::with_capacity(h + 2);
        lines.push(format!(
            "╔{}╗",
            "═".repeat(w)
        ));
        lines.push(format!(
            "║ {:^width$} ║",
            "FREQUENCY MAP (Reasoning Patterns)",
            width = w
        ));
        lines.push(format!("╠{}╣", "═".repeat(w)));

        // Create heatmap from spectrum
        let _rows_per_band = h / 4;
        for band in 0..4 {
            let mut row = String::from("║");
            for col in 0..w {
                let idx = (col * spectrum.len()) / w;
                let value = spectrum.get(idx).unwrap_or(&0.0);

                // Map to band
                let band_value = if band < 3 {
                    *spectrum.get(idx * 4 + band).unwrap_or(value)
                } else {
                    *value
                };

                let ch = match band {
                    0 if band_value > 0.75 => '▓',
                    0 if band_value > 0.5 => '▒',
                    0 => '░',
                    1 if band_value > 0.6 => '█',
                    1 if band_value > 0.3 => '▓',
                    1 => '░',
                    2 if band_value > 0.5 => '▒',
                    2 => '░',
                    _ if band_value > 0.3 => '░',
                    _ => ' ',
                };
                row.push(ch);
            }
            row.push_str(" ║");
            lines.push(row);
        }

        lines.push(format!("╚{}╝", "═".repeat(w)));
        lines.push(format!(
            "  Quality: {:.2} | Impedance: {:.2} | Entropy: {:.2}",
            signature.quality_score, signature.impedance, signature.entropy
        ));

        lines.join("\n")
    }

    /// Build impedance map — dead spots across the graph
    /// High impedance = bright (dead/absorbs), low impedance = dark (responsive)
    fn impedance_map(signature: &ResonanceSignature, w: usize, h: usize) -> String {
        let z = signature.impedance;

        let mut lines = Vec::with_capacity(h + 2);
        lines.push(format!("╔{}╗", "═".repeat(w)));
        lines.push(format!(
            "║ {:^width$} ║",
            "IMPEDANCE MAP (Dead Spots)",
            width = w
        ));
        lines.push(format!("╠{}╣", "═".repeat(w)));

        // Impedance as vertical gradient
        for row in 0..h {
            let mut line = String::from("║");
            let row_z = if row < h / 2 { z } else { 1.0 - z * 0.5 };

            for col in 0..w {
                // Add some pattern variation
                let variation = ((col as f64 * 0.1).sin() * 0.2) as f32;
                let cell_z = (row_z as f32 + variation).max(0.0).min(1.0) as f64;

                let ch = if cell_z > 0.8 {
                    '█'
                } else if cell_z > 0.6 {
                    '▓'
                } else if cell_z > 0.4 {
                    '▒'
                } else if cell_z > 0.2 {
                    '░'
                } else {
                    ' '
                };
                line.push(ch);
            }
            line.push_str(" ║");
            lines.push(line);
        }

        lines.push(format!("╚{}╝", "═".repeat(w)));
        lines.push(format!(
            "  Z = {:.3} ({})",
            z,
            if z > 0.7 {
                "DEAD SPOT"
            } else if z > 0.4 {
                "Moderate"
            } else {
                "Responsive"
            }
        ));

        lines.join("\n")
    }

    /// Build perfusion map — time-series of activation flowing through pathways
    /// Shows how response builds and decays over token positions
    fn perfusion_map(signature: &ResonanceSignature, w: usize, h: usize) -> String {
        // Use harmonics as "time series" proxy
        let harmonics = &signature.harmonic_content;

        let mut lines = Vec::with_capacity(h + 2);
        lines.push(format!("╔{}╗", "═".repeat(w)));
        lines.push(format!(
            "║ {:^width$} ║",
            "PERFUSION MAP (Activation Flow)",
            width = w
        ));
        lines.push(format!("╠{}╣", "═".repeat(w)));

        // Map harmonics to rows (bottom = early, top = late)
        let rows_per_harmonic = h / harmonics.len().max(1);
        for row in (0..h).rev() {
            let mut line = String::from("║");
            let harmonic_idx = (h - 1 - row) / rows_per_harmonic.max(1);
            let value = harmonics.get(harmonic_idx).copied().unwrap_or(0.0);

            for col in 0..w {
                // Create wave pattern based on column and harmonic
                let phase = col as f64 * 0.15 + harmonic_idx as f64 * 0.5;
                let wave = ((phase.sin() + 1.0) / 2.0) * value;
                let threshold = col as f64 / w as f64;

                let ch = if wave > threshold * 0.8 {
                    '▓'
                } else if wave > threshold * 0.5 {
                    '▒'
                } else if wave > threshold * 0.2 {
                    '░'
                } else {
                    ' '
                };
                line.push(ch);
            }
            line.push_str(" ║");
            lines.push(line);
        }

        lines.push(format!("╚{}╝", "═".repeat(w)));
        lines.push(format!(
            "  Decay: {:.3} | Flow direction: ↓",
            signature.decay_rate
        ));

        lines.join("\n")
    }

    /// Build anisotropy map — directional vs diffuse reasoning paths
    /// Shows whether reasoning is focused (anisotropic) or scattered (isotropic)
    fn anisotropy_map(signature: &ResonanceSignature, w: usize, h: usize) -> String {
        // Compute anisotropy from entropy and variance
        let entropy = signature.entropy;
        let variance = signature.logprob_variance;

        // High entropy + low variance = isotropic (scattered)
        // Low entropy + high variance = anisotropic (focused)
        let anisotropy = if entropy > 0.0 {
            (1.0 / (1.0 + entropy * 0.1)) * (1.0 + variance)
        } else {
            0.5
        }
        .clamp(0.0, 1.0);

        let mut lines = Vec::with_capacity(h + 2);
        lines.push(format!("╔{}╗", "═".repeat(w)));
        lines.push(format!(
            "║ {:^width$} ║",
            "ANISOTROPY MAP (Reasoning Focus)",
            width = w
        ));
        lines.push(format!("╠{}╣", "═".repeat(w)));

        // Draw directional arrows (anisotropic) or diffuse dots (isotropic)
        let center_y = h / 2;
        let center_x = w / 2;

        let chars: Vec<char> = if anisotropy > 0.6 {
            // Focused/directional - draw arrows pointing inward
            vec!['→', '⟶', '─', '⟵', '←']
        } else if anisotropy > 0.3 {
            // Mixed
            vec!['·', '○', '◌', '•', '·']
        } else {
            // Scattered/diffuse
            vec!['░', '▒', '▓', '░', '▒']
        };

        for row in 0..h {
            let mut line = String::from("║");
            for col in 0..w {
                let dx = (col as i32 - center_x as i32).abs() as f64 / w as f64;
                let dy = (row as i32 - center_y as i32).abs() as i32 as f64 / h as f64;
                let dist = (dx * dx + dy * dy).sqrt();

                let idx = ((dist * 4.0) as usize).min(chars.len() - 1);
                line.push(chars[idx]);
            }
            line.push_str(" ║");
            lines.push(line);
        }

        lines.push(format!("╚{}╝", "═".repeat(w)));
        lines.push(format!(
            "  Anisotropy: {:.3} ({})",
            anisotropy,
            if anisotropy > 0.6 {
                "Focused"
            } else if anisotropy > 0.3 {
                "Mixed"
            } else {
                "Scattered"
            }
        ));

        lines.join("\n")
    }

    /// Generate comparison image (base vs tap side by side)
    pub fn compare_images(
        base: &ResonanceSignature,
        tap: &ResonanceSignature,
        image_type: ImageType,
        width: usize,
    ) -> String {
        let half = (width / 2).saturating_sub(2);
        let base_img = Self::image(base, image_type, half, 15);
        let tap_img = Self::image(tap, image_type, half, 15);

        let base_lines: Vec<&str> = base_img.lines().collect();
        let tap_lines: Vec<&str> = tap_img.lines().collect();

        let mut combined = Vec::with_capacity(base_lines.len().max(tap_lines.len()));
        for i in 0..base_lines.len().max(tap_lines.len()) {
            let base_line = base_lines.get(i).unwrap_or(&"");
            let tap_line = tap_lines.get(i).unwrap_or(&"");
            combined.push(format!("{} │ {}", base_line, tap_line));
        }

        combined.join("\n")
    }

    /// Generate full report with all image types
    pub fn full_report(signature: &ResonanceSignature) -> String {
        let w = 60;
        let h = 12;

        format!(
            "╔════════════════════════════════════════════════════════════════════╗
║                    FLEET-RESONANCE IMAGING REPORT                     ║
╠════════════════════════════════════════════════════════════════════╣
║
║  FREQUENCY SPECTRUM
║  {}
║
║  IMPEDANCE MAP
║  {}
║
║  PERFUSION MAP
║  {}
║
║  ANISOTROPY MAP
║  {}
║
╠════════════════════════════════════════════════════════════════════╣
║  METRICS                                                           ║
║    Quality Score:    {:.3}                                         ║
║    Impedance (Z):    {:.3}  {}                                     ║
║    Entropy:          {:.3}                                         ║
║    Decay Rate:       {:.3}                                         ║
║    Logprob Variance: {:.3}                                         ║
╚════════════════════════════════════════════════════════════════════╝",
            Self::frequency_map(signature, w, h),
            Self::impedance_map(signature, w, h),
            Self::perfusion_map(signature, w, h),
            Self::anisotropy_map(signature, w, h),
            signature.quality_score,
            signature.impedance,
            if signature.impedance > 0.7 {
                "(DEAD SPOT)"
            } else if signature.impedance > 0.4 {
                "(Moderate)"
            } else {
                "(Responsive)"
            },
            signature.entropy,
            signature.decay_rate,
            signature.logprob_variance,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::resonance::ResonanceSignature;

    fn make_test_signature() -> ResonanceSignature {
        ResonanceSignature {
            frequency_spectrum: vec![0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02],
            decay_rate: 0.15,
            harmonic_content: vec![0.7, 0.4, 0.2, 0.1],
            impedance: 0.35,
            entropy: 2.5,
            logprob_variance: 0.25,
            quality_score: 0.72,
        }
    }

    #[test]
    fn test_frequency_map() {
        let sig = make_test_signature();
        let img = ResonanceImager::frequency_map(&sig, 40, 12);
        assert!(img.contains("FREQUENCY MAP"));
        assert!(img.contains("╔"));
    }

    #[test]
    fn test_impedance_map() {
        let sig = make_test_signature();
        let img = ResonanceImager::impedance_map(&sig, 40, 12);
        assert!(img.contains("IMPEDANCE MAP"));
        assert!(img.contains("Z ="));
    }

    #[test]
    fn test_perfusion_map() {
        let sig = make_test_signature();
        let img = ResonanceImager::perfusion_map(&sig, 40, 12);
        assert!(img.contains("PERFUSION MAP"));
    }

    #[test]
    fn test_anisotropy_map() {
        let sig = make_test_signature();
        let img = ResonanceImager::anisotropy_map(&sig, 40, 12);
        assert!(img.contains("ANISOTROPY MAP"));
    }

    #[test]
    fn test_compare_images() {
        let sig = make_test_signature();
        let img = ResonanceImager::compare_images(&sig, &sig, ImageType::Frequency, 80);
        assert!(img.contains("│"));
    }

    #[test]
    fn test_full_report() {
        let sig = make_test_signature();
        let report = ResonanceImager::full_report(&sig);
        assert!(report.contains("FLEET-RESONANCE IMAGING REPORT"));
        assert!(report.contains("Quality Score"));
    }
}