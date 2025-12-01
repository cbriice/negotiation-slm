# Metrics Tracking System

## Overview
Modular system for tracking negotiation training metrics, calculating episode/session statistics, and persisting data to CSV files.

## Components

### 1. EpisodeMetrics (`episode_metrics.py`)
Calculates metrics for individual negotiation episodes:
- **Deal outcomes**: closure status, final price, terms matching
- **Protocol compliance**: violations detection, state machine validation
- **Negotiation dynamics**: proposals, counteroffers, price convergence
- **Communication quality**: verbosity, reasoning, empathy, professionalism

### 2. SessionMetrics (`session_metrics.py`)
Aggregates episode metrics into session-level statistics:
- **Performance metrics**: deal close rate, average turns/tokens
- **Throughput**: tokens/second (tps), conversations/second (ctps), agent tokens/second (ltps)
- **Side-specific analysis**: buyer vs seller performance
- **Violation patterns**: top violation types and frequencies

### 3. CSVLogger (`csv_logger.py`)
Handles metric persistence using pandas:
- **Stage-specific files**: separate CSVs per training stage
- **Automatic initialization**: creates files with proper headers
- **Data recovery**: read methods for analysis and reporting
- **Summary statistics**: quick aggregations from stored data

### 4. MetricsIntegration (`metrics_integration.py`)
Bridges metrics with ParlAI teachers:
- **Episode tracking**: individual or batch processing
- **Session management**: automatic aggregation and finalization
- **Export capabilities**: JSONL export for training data
- **Real-time monitoring**: current stats without finalization

## Usage

### Basic Integration
```python
from evaluation.metrics_integration import MetricsIntegration

# Initialize
metrics = MetricsIntegration(data_dir='data/metrics')
metrics.set_stage('selfchat')  # or 'baseline', 'dialog_partner', etc.

# Track episode
episode_data = {
    'meta': {'episode_id': 'ep_001', 'side': 'buyer'},
    'state': {...},  # Initial conditions
    'dialogue': [...]  # Turn exchanges
}
metrics.track_episode(episode_data)

# Get current stats
stats = metrics.get_current_stats()
metrics.log_summary()  # Logs to logger

# Finalize session
session_metrics = metrics.finalize_session()
```

### With ParlAI Teachers
```python
class NegotiationTeacher(FixedDialogTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.metrics = MetricsIntegration()
        
    def observe(self, observation):
        # After episode completes
        if observation.get('episode_done'):
            episode = self.current_episode  # Your episode data
            self.metrics.track_episode(episode)
```

## Data Flow

1. **Episode Generation**: ParlAI worlds generate negotiation episodes
2. **Metric Calculation**: EpisodeMetrics analyzes each episode
3. **Session Aggregation**: SessionMetrics computes batch statistics
4. **CSV Persistence**: CSVLogger writes to stage-specific files
5. **Analysis**: Read CSVs for training decisions and reporting

## File Structure

```
data/
├── metrics/
│   ├── episodes.csv           # All episodes
│   ├── sessions.csv           # All sessions
│   ├── episodes_baseline.csv  # Stage-specific
│   ├── sessions_baseline.csv
│   ├── episodes_selfchat.csv
│   └── sessions_selfchat.csv
└── exports/
    └── session_*.jsonl        # Exported episode data
```

## Key Metrics Tracked

### Per Episode
- `episode_id`, `dialogue_id`, `side`, `seed`
- `turns`, `total_tokens`, `agent_tokens`
- `deal_closed`, `final_price`, `terms_match_pct`
- `protocol_violations`, `violation_count`
- `num_proposals`, `price_convergence`
- `uses_reasoning`, `uses_empathy`, `professionalism_score`

### Per Session
- `total_episodes`, `deals_closed_pct`
- `avg_turns`, `avg_tokens`
- `tokens_per_second`, `conversations_per_second`
- `buyer/seller_avg_price_vs_target`
- `top_violations`

## Testing

Run `python test_metrics.py` to validate all components with sample negotiation data.