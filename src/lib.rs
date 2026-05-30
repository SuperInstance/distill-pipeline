//! Distillation Pipeline for the Grand Pattern ecosystem.
//!
//! The pipeline models how rooms learn: raw signals accumulate during Wake,
//! get re-weighted during REM (re-distillation), and compressed during
//! DeepSleep. The distilled wisdom IS the recomposed intelligence.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// A raw observation entering the distillation pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub tick: u64,
    pub value: f64,
    pub confidence: f64,
}

/// A distilled record — compressed wisdom from many observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistilledRecord {
    pub start_tick: u64,
    pub end_tick: u64,
    pub compressed_value: f64,
    pub compressed_confidence: f64,
    pub observation_count: usize,
    pub variance: f64,
}

impl DistilledRecord {
    /// Merge two adjacent distilled records into one.
    pub fn merge(&self, other: &DistilledRecord) -> DistilledRecord {
        let total = self.observation_count + other.observation_count;
        let w1 = self.observation_count as f64 / total as f64;
        let w2 = other.observation_count as f64 / total as f64;
        let compressed_value = w1 * self.compressed_value + w2 * other.compressed_value;
        let compressed_confidence = w1 * self.compressed_confidence + w2 * other.compressed_confidence;
        let delta = other.compressed_value - self.compressed_value;
        let variance = w1 * (self.variance + delta * delta) + w2 * other.variance - w1 * w2 * delta * delta;
        DistilledRecord {
            start_tick: self.start_tick,
            end_tick: other.end_tick,
            compressed_value,
            compressed_confidence,
            observation_count: total,
            variance: variance.max(0.0),
        }
    }
}

/// Which phase of the wake/sleep cycle the pipeline is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    Wake,
    Rem,
    DeepSleep,
}

/// Configuration for the distillation pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillConfig {
    /// Maximum raw observations before forcing a distillation
    pub buffer_capacity: usize,
    /// Minimum observations required before distilling
    pub min_observations: usize,
    /// Variance threshold below which records get merged (deep sleep compression)
    pub merge_variance_threshold: f64,
    /// Maximum number of distilled records to retain
    pub max_distilled: usize,
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            buffer_capacity: 100,
            min_observations: 5,
            merge_variance_threshold: 0.01,
            max_distilled: 50,
        }
    }
}

/// Statistics about the distillation pipeline.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistillStats {
    pub total_observations_seen: usize,
    pub total_distillations: usize,
    pub total_merges: usize,
    pub current_buffer_size: usize,
    pub current_distilled_count: usize,
    pub compression_ratio: f64,
}

/// The distillation pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillPipeline {
    config: DistillConfig,
    buffer: VecDeque<Observation>,
    distilled: Vec<DistilledRecord>,
    phase: Phase,
    stats: DistillStats,
}

impl DistillPipeline {
    pub fn new(config: DistillConfig) -> Self {
        Self {
            config,
            buffer: VecDeque::new(),
            distilled: Vec::new(),
            phase: Phase::Wake,
            stats: DistillStats::default(),
        }
    }

    /// Set the current phase.
    pub fn set_phase(&mut self, phase: Phase) {
        self.phase = phase;
    }

    /// Current phase.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Ingest a raw observation.
    pub fn observe(&mut self, obs: Observation) {
        self.buffer.push_back(obs);
        self.stats.total_observations_seen += 1;
        self.stats.current_buffer_size = self.buffer.len();

        // Auto-distill if buffer is full
        if self.buffer.len() >= self.config.buffer_capacity {
            self.distill();
        }
    }

    /// Distill the current buffer into a compressed record.
    /// This is the REM phase operation.
    pub fn distill(&mut self) -> Option<DistilledRecord> {
        if self.buffer.len() < self.config.min_observations {
            return None;
        }

        let n = self.buffer.len() as f64;
        let values: Vec<f64> = self.buffer.iter().map(|o| o.value).collect();
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let conf_mean = self.buffer.iter().map(|o| o.confidence).sum::<f64>() / n;

        let record = DistilledRecord {
            start_tick: self.buffer.front().unwrap().tick,
            end_tick: self.buffer.back().unwrap().tick,
            compressed_value: mean,
            compressed_confidence: conf_mean,
            observation_count: self.buffer.len(),
            variance,
        };

        self.distilled.push(record.clone());
        self.buffer.clear();
        self.stats.total_distillations += 1;
        self.stats.current_buffer_size = 0;
        self.stats.current_distilled_count = self.distilled.len();

        if self.stats.total_observations_seen > 0 {
            self.stats.compression_ratio = self.distilled.len() as f64
                / self.stats.total_observations_seen as f64;
        }

        Some(record)
    }

    /// Compress distilled records during deep sleep.
    /// Merges adjacent records with low variance.
    pub fn deep_compress(&mut self) -> usize {
        if self.distilled.len() < 2 {
            return 0;
        }

        let mut merged = Vec::new();
        let mut i = 0;
        let mut merge_count = 0;

        while i < self.distilled.len() {
            if i + 1 < self.distilled.len() {
                let a = &self.distilled[i];
                let b = &self.distilled[i + 1];
                let combined_var = (a.variance + b.variance) / 2.0;
                if combined_var < self.config.merge_variance_threshold {
                    merged.push(a.merge(b));
                    merge_count += 1;
                    i += 2;
                    continue;
                }
            }
            merged.push(self.distilled[i].clone());
            i += 1;
        }

        self.distilled = merged;
        self.stats.total_merges += merge_count;
        self.stats.current_distilled_count = self.distilled.len();

        if self.stats.total_observations_seen > 0 {
            self.stats.compression_ratio = self.distilled.len() as f64
                / self.stats.total_observations_seen as f64;
        }

        merge_count
    }

    /// Get the distilled records.
    pub fn distilled(&self) -> &[DistilledRecord] {
        &self.distilled
    }

    /// Get current stats.
    pub fn stats(&self) -> &DistillStats {
        &self.stats
    }

    /// Predict the next value based on distilled history.
    /// Uses weighted average of recent distilled records.
    pub fn predict(&self) -> Option<f64> {
        if self.distilled.is_empty() && self.buffer.is_empty() {
            return None;
        }

        // Prefer distilled records, fall back to buffer
        if !self.distilled.is_empty() {
            let n = self.distilled.len().min(5);
            let recent = &self.distilled[self.distilled.len() - n..];
            let total_obs: f64 = recent.iter().map(|r| r.observation_count as f64).sum();
            let weighted: f64 = recent
                .iter()
                .map(|r| r.compressed_value * r.observation_count as f64)
                .sum();
            Some(weighted / total_obs)
        } else {
            let n = self.buffer.len() as f64;
            Some(self.buffer.iter().map(|o| o.value).sum::<f64>() / n)
        }
    }

    /// Run a full wake-REM-deep cycle.
    pub fn run_cycle(&mut self, observations: Vec<Observation>) -> DistillStats {
        // Wake: ingest
        self.phase = Phase::Wake;
        for obs in observations {
            self.observe(obs);
        }

        // REM: distill
        self.phase = Phase::Rem;
        self.distill();

        // Deep: compress
        self.phase = Phase::DeepSleep;
        self.deep_compress();

        self.stats.clone()
    }

    /// Reset the pipeline.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.distilled.clear();
        self.stats = DistillStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(tick: u64, value: f64) -> Observation {
        Observation { tick, value, confidence: 1.0 }
    }

    #[test]
    fn test_empty_predict() {
        let pipeline = DistillPipeline::new(DistillConfig::default());
        assert!(pipeline.predict().is_none());
    }

    #[test]
    fn test_single_observe() {
        let mut p = DistillPipeline::new(DistillConfig::default());
        p.observe(obs(0, 1.0));
        assert_eq!(p.buffer.len(), 1);
        assert_eq!(p.stats.total_observations_seen, 1);
    }

    #[test]
    fn test_distill_minimum() {
        let mut p = DistillPipeline::new(DistillConfig::default());
        for i in 0..4 {
            p.observe(obs(i, 1.0));
        }
        assert!(p.distill().is_none()); // Below minimum
    }

    #[test]
    fn test_distill_basic() {
        let mut p = DistillPipeline::new(DistillConfig::default());
        for i in 0..5 {
            p.observe(obs(i, 1.0));
        }
        let record = p.distill().unwrap();
        assert!((record.compressed_value - 1.0).abs() < 1e-10);
        assert_eq!(record.observation_count, 5);
        assert!(record.variance.abs() < 1e-10);
        assert_eq!(p.buffer.len(), 0);
    }

    #[test]
    fn test_distill_with_variance() {
        let mut p = DistillPipeline::new(DistillConfig::default());
        for i in 0..5 {
            p.observe(obs(i, i as f64));
        }
        let record = p.distill().unwrap();
        assert!((record.compressed_value - 2.0).abs() < 1e-10);
        assert!(record.variance > 0.0);
    }

    #[test]
    fn test_auto_distill_on_buffer_full() {
        let config = DistillConfig { buffer_capacity: 10, ..Default::default() };
        let mut p = DistillPipeline::new(config);
        for i in 0..10 {
            p.observe(obs(i, 1.0));
        }
        // Buffer should have been auto-distilled
        assert_eq!(p.buffer.len(), 0);
        assert_eq!(p.distilled.len(), 1);
    }

    #[test]
    fn test_predict_from_distilled() {
        let mut p = DistillPipeline::new(DistillConfig::default());
        for i in 0..5 {
            p.observe(obs(i, 3.0));
        }
        p.distill();
        assert!((p.predict().unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_predict_from_buffer() {
        let mut p = DistillPipeline::new(DistillConfig::default());
        for i in 0..3 {
            p.observe(obs(i, 5.0));
        }
        // Not enough to distill, but predict still works from buffer
        assert!((p.predict().unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_deep_compress_no_merge() {
        let mut p = DistillPipeline::new(DistillConfig {
            merge_variance_threshold: 0.001,
            ..Default::default()
        });
        for batch in 0..3 {
            for i in 0..5 {
                p.observe(obs(batch * 10 + i, batch as f64 * 10.0 + i as f64));
            }
            p.distill();
        }
        let merges = p.deep_compress();
        assert_eq!(merges, 0); // High variance between batches, no merging
        assert_eq!(p.distilled.len(), 3);
    }

    #[test]
    fn test_deep_compress_with_merge() {
        let mut p = DistillPipeline::new(DistillConfig {
            merge_variance_threshold: 1.0,
            ..Default::default()
        });
        // Two batches with similar values
        for i in 0..5 { p.observe(obs(i, 1.0)); }
        p.distill();
        for i in 5..10 { p.observe(obs(5 + i, 1.01)); }
        p.distill();
        let merges = p.deep_compress();
        assert!(merges >= 1);
        assert_eq!(p.distilled.len(), 1);
    }

    #[test]
    fn test_record_merge() {
        let a = DistilledRecord {
            start_tick: 0, end_tick: 4,
            compressed_value: 2.0, compressed_confidence: 1.0,
            observation_count: 5, variance: 0.0,
        };
        let b = DistilledRecord {
            start_tick: 5, end_tick: 9,
            compressed_value: 3.0, compressed_confidence: 0.9,
            observation_count: 5, variance: 0.0,
        };
        let merged = a.merge(&b);
        assert!((merged.compressed_value - 2.5).abs() < 1e-10);
        assert_eq!(merged.observation_count, 10);
        assert_eq!(merged.start_tick, 0);
        assert_eq!(merged.end_tick, 9);
    }

    #[test]
    fn test_phase_transitions() {
        let mut p = DistillPipeline::new(DistillConfig::default());
        assert_eq!(p.phase(), Phase::Wake);
        p.set_phase(Phase::Rem);
        assert_eq!(p.phase(), Phase::Rem);
        p.set_phase(Phase::DeepSleep);
        assert_eq!(p.phase(), Phase::DeepSleep);
    }

    #[test]
    fn test_stats_tracking() {
        let mut p = DistillPipeline::new(DistillConfig::default());
        for i in 0..5 { p.observe(obs(i, 1.0)); }
        p.distill();
        let stats = p.stats();
        assert_eq!(stats.total_observations_seen, 5);
        assert_eq!(stats.total_distillations, 1);
        assert_eq!(stats.current_distilled_count, 1);
    }

    #[test]
    fn test_compression_ratio() {
        let mut p = DistillPipeline::new(DistillConfig::default());
        for i in 0..15 { p.observe(obs(i, 1.0)); }
        p.distill(); // 15 observations → 1 record
        assert!((p.stats().compression_ratio - 1.0 / 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_full_cycle() {
        let mut p = DistillPipeline::new(DistillConfig::default());
        let observations: Vec<Observation> = (0..5).map(|i| obs(i, i as f64)).collect();
        let stats = p.run_cycle(observations);
        assert_eq!(stats.total_observations_seen, 5);
        assert_eq!(stats.total_distillations, 1);
    }

    #[test]
    fn test_reset() {
        let mut p = DistillPipeline::new(DistillConfig::default());
        for i in 0..5 { p.observe(obs(i, 1.0)); }
        p.distill();
        p.reset();
        assert_eq!(p.buffer.len(), 0);
        assert_eq!(p.distilled.len(), 0);
        assert_eq!(p.stats().total_observations_seen, 0);
    }

    #[test]
    fn test_multiple_distillations() {
        let mut p = DistillPipeline::new(DistillConfig::default());
        for batch in 0..3 {
            for i in 0..5 { p.observe(obs(batch * 5 + i, batch as f64)); }
            p.distill();
        }
        assert_eq!(p.distilled.len(), 3);
        // Predict should weight recent batches
        let pred = p.predict().unwrap();
        assert!(pred >= 0.0 && pred <= 2.0);
    }

    #[test]
    fn test_serialization() {
        let mut p = DistillPipeline::new(DistillConfig::default());
        for i in 0..5 { p.observe(obs(i, 1.0)); }
        p.distill();
        let json = serde_json::to_string(&p).unwrap();
        let restored: DistillPipeline = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.distilled.len(), 1);
        assert_eq!(restored.stats().total_observations_seen, 5);
    }

    #[test]
    fn test_empty_deep_compress() {
        let mut p = DistillPipeline::new(DistillConfig::default());
        assert_eq!(p.deep_compress(), 0);
    }
}
