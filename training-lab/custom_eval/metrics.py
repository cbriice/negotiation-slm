"""
Metrics tracking and analysis for negotiation evaluations.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


class NegotiationMetrics:
    """Analyze and track negotiation performance metrics."""

    def __init__(self):
        self.results = []

    def add_result(self, result: Dict[str, Any]):
        """Add a negotiation result for analysis."""
        self.results.append(result)

    def load_results(self, result_file: str):
        """Load results from a JSON file."""
        with open(result_file, 'r') as f:
            data = json.load(f)
            if 'negotiations' in data:
                self.results.extend(data['negotiations'])
            else:
                self.results.append(data)

    def compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics across all results."""
        if not self.results:
            return {}

        # Extract metrics
        deals_made = [r['metrics']['deal_made'] for r in self.results]
        turns = [r['metrics']['turns'] for r in self.results]
        qualities = [r['metrics'].get('conversation_quality', 0) for r in self.results]

        # Deal prices (when deals were made)
        deal_prices = [r['metrics']['final_price'] for r in self.results
                      if r['metrics']['deal_made'] and r['metrics']['final_price']]

        # Token counts
        buyer_tokens = [r['metrics']['buyer_tokens'] for r in self.results]
        seller_tokens = [r['metrics']['seller_tokens'] for r in self.results]
        total_tokens = [r['metrics']['total_tokens'] for r in self.results]

        summary = {
            'total_negotiations': len(self.results),
            'deals_made': sum(deals_made),
            'deal_rate': np.mean(deals_made),
            'turns': {
                'mean': np.mean(turns),
                'std': np.std(turns),
                'min': min(turns),
                'max': max(turns),
            },
            'quality': {
                'mean': np.mean(qualities),
                'std': np.std(qualities),
                'min': min(qualities) if qualities else 0,
                'max': max(qualities) if qualities else 0,
            },
            'tokens': {
                'buyer_mean': np.mean(buyer_tokens),
                'seller_mean': np.mean(seller_tokens),
                'total_mean': np.mean(total_tokens),
            }
        }

        if deal_prices:
            summary['deal_prices'] = {
                'mean': np.mean(deal_prices),
                'std': np.std(deal_prices),
                'min': min(deal_prices),
                'max': max(deal_prices),
            }

        return summary

    def analyze_by_category(self) -> Dict[str, Dict[str, Any]]:
        """Analyze results grouped by item category."""
        categories = {}

        for result in self.results:
            category = result['seed_data'].get('category', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(result)

        # Compute metrics per category
        category_metrics = {}
        for cat, cat_results in categories.items():
            deals = [r['metrics']['deal_made'] for r in cat_results]
            qualities = [r['metrics'].get('conversation_quality', 0) for r in cat_results]

            category_metrics[cat] = {
                'count': len(cat_results),
                'deal_rate': np.mean(deals),
                'avg_quality': np.mean(qualities),
                'avg_turns': np.mean([r['metrics']['turns'] for r in cat_results]),
            }

        return category_metrics

    def find_issues(self, quality_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Find negotiations with quality issues."""
        issues = []

        for result in self.results:
            quality = result['metrics'].get('conversation_quality', 1.0)
            if quality < quality_threshold:
                issues.append({
                    'seed_id': result.get('seed_id'),
                    'quality': quality,
                    'turns': result['metrics']['turns'],
                    'item': result['seed_data']['item']['name'],
                    'sample_text': result['dialogue'][0]['text'][:100] if result['dialogue'] else ""
                })

        return sorted(issues, key=lambda x: x['quality'])

    def export_report(self, output_file: str):
        """Export a comprehensive metrics report."""
        report = {
            'summary': self.compute_summary(),
            'by_category': self.analyze_by_category(),
            'quality_issues': self.find_issues(),
            'num_results': len(self.results),
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        return report


def analyze_results_directory(results_dir: str) -> Dict[str, Any]:
    """Analyze all result files in a directory."""
    results_path = Path(results_dir)
    metrics = NegotiationMetrics()

    # Load all result files
    for result_file in results_path.glob("selfchat_results_*.json"):
        metrics.load_results(result_file)

    if not metrics.results:
        print(f"No results found in {results_dir}")
        return {}

    # Compute and return comprehensive metrics
    return {
        'total_files': len(list(results_path.glob("selfchat_results_*.json"))),
        'metrics': metrics.compute_summary(),
        'by_category': metrics.analyze_by_category(),
        'issues': metrics.find_issues(),
    }