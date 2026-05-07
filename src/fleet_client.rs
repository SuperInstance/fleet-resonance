//! fleet_client.rs — Connect to keeper API for fleet data
//!
//! Integrates with the PLATO room server at localhost:8847 to:
//! - Fetch fleet trust graphs
//! - Write resonance signatures to rooms
//! - Read contrast maps from room history

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

const PLATO_HOST: &str = "http://127.0.0.1:8847";

/// Fleet data from keeper API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetData {
    pub agents: Vec<AgentInfo>,
    pub trust_scores: Vec<TrustEdge>,
    pub room_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub id: String,
    pub name: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustEdge {
    pub from: String,
    pub to: String,
    pub trust: f64,
}

/// Resonance signature for PLATO storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceRecord {
    pub room_name: String,
    pub probe_type: String,
    pub prompt_hash: String,
    pub signature: SerializedSignature,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedSignature {
    pub frequency_spectrum: Vec<f64>,
    pub decay_rate: f64,
    pub harmonic_content: Vec<f64>,
    pub impedance: f64,
    pub entropy: f64,
    pub logprob_variance: f64,
    pub quality_score: f64,
}

/// Contrast result for PLATO storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastRecord {
    pub room_name: String,
    pub base_prompt: String,
    pub tap_prompt: String,
    pub similarity: f64,
    pub hyperperfused_count: usize,
    pub hypoperfused_count: usize,
    pub dead_spot_count: usize,
    pub timestamp: i64,
}

/// Fleet client for PLATO integration
#[derive(Clone)]
pub struct FleetClient {
    keeper_url: String,
    http_client: Client,
}

impl FleetClient {
    pub fn new(keeper_url: Option<String>) -> Self {
        Self {
            keeper_url: keeper_url.unwrap_or_else(|| PLATO_HOST.to_string()),
            http_client: Client::new(),
        }
    }

    /// Check if keeper is available
    pub async fn is_available(&self) -> bool {
        self.http_client
            .get(format!("{}/health", self.keeper_url))
            .send()
            .await
            .is_ok()
    }

    /// Fetch fleet data from keeper API
    pub async fn get_fleet_data(&self, room: &str) -> Result<FleetData> {
        let url = format!("{}/room/{}/fleet_data", self.keeper_url, room);
        let resp = self
            .http_client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch fleet data")?
            .error_for_status()
            .context("Keeper API error")?;

        let data: FleetData = resp.json().await.context("Failed to parse fleet data")?;
        Ok(data)
    }

    /// Write resonance signature to PLATO room
    pub async fn write_signature(
        &self,
        room: &str,
        probe_type: &str,
        prompt_hash: &str,
        signature: &crate::resonance::ResonanceSignature,
    ) -> Result<()> {
        let record = ResonanceRecord {
            room_name: room.to_string(),
            probe_type: probe_type.to_string(),
            prompt_hash: prompt_hash.to_string(),
            signature: SerializedSignature {
                frequency_spectrum: signature.frequency_spectrum.clone(),
                decay_rate: signature.decay_rate,
                harmonic_content: signature.harmonic_content.clone(),
                impedance: signature.impedance,
                entropy: signature.entropy,
                logprob_variance: signature.logprob_variance,
                quality_score: signature.quality_score,
            },
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
        };

        let url = format!("{}/room/{}/resonance", self.keeper_url, room);
        self.http_client
            .post(&url)
            .json(&record)
            .send()
            .await
            .context("Failed to write resonance signature")?
            .error_for_status()
            .context("Keeper API error")?;

        Ok(())
    }

    /// Fetch resonance signatures from PLATO room
    pub async fn get_signatures(&self, room: &str) -> Result<Vec<ResonanceRecord>> {
        let url = format!("{}/room/{}/resonance", self.keeper_url, room);
        let resp = self
            .http_client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch signatures")?
            .error_for_status()
            .context("Keeper API error")?;

        let records: Vec<ResonanceRecord> = resp.json().await.context("Failed to parse signatures")?;
        Ok(records)
    }

    /// Write contrast result to PLATO room
    pub async fn write_contrast(
        &self,
        room: &str,
        base_prompt: &str,
        tap_prompt: &str,
        result: &crate::contrast::ContrastResult,
    ) -> Result<()> {
        let record = ContrastRecord {
            room_name: room.to_string(),
            base_prompt: base_prompt.to_string(),
            tap_prompt: tap_prompt.to_string(),
            similarity: result.similarity_score,
            hyperperfused_count: result.contrast_map.hyperperfused.len(),
            hypoperfused_count: result.contrast_map.hypoperfused.len(),
            dead_spot_count: result.contrast_map.dead_spots.len(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
        };

        let url = format!("{}/room/{}/contrast", self.keeper_url, room);
        self.http_client
            .post(&url)
            .json(&record)
            .send()
            .await
            .context("Failed to write contrast")?
            .error_for_status()
            .context("Keeper API error")?;

        Ok(())
    }

    /// Fetch contrast history from PLATO room
    pub async fn get_contrast_history(&self, room: &str) -> Result<Vec<ContrastRecord>> {
        let url = format!("{}/room/{}/contrast", self.keeper_url, room);
        let resp = self
            .http_client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch contrast history")?
            .error_for_status()
            .context("Keeper API error")?;

        let records: Vec<ContrastRecord> = resp.json().await.context("Failed to parse contrast history")?;
        Ok(records)
    }

    /// Get room history (generic)
    pub async fn get_room_history(&self, room: &str, limit: usize) -> Result<Vec<serde_json::Value>> {
        let url = format!("{}/room/{}_history?limit={}", self.keeper_url, room, limit);
        let resp = self
            .http_client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch room history")?
            .error_for_status()
            .context("Keeper API error")?;

        let history: Vec<serde_json::Value> = resp.json().await.context("Failed to parse history")?;
        Ok(history)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fleet_client_creation() {
        let client = FleetClient::new(None);
        assert_eq!(client.keeper_url, PLATO_HOST);
    }

    #[test]
    fn test_custom_url() {
        let client = FleetClient::new(Some("http://localhost:9000".to_string()));
        assert_eq!(client.keeper_url, "http://localhost:9000");
    }
}