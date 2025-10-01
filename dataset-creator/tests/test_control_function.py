"""
Comprehensive test suite for the control function implementation.
Tests various negotiation scenarios with diverse dummy data.
"""

import sys
import numpy as np
from pathlib import Path
import json
import pandas as pd
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.control_function import ControlFunction, NegotiationParams, NegotiationState


class ControlFunctionTester:
    """Test harness for control function with varied scenarios."""

    def __init__(self, change_note=None):
        self.scenarios = self._create_test_scenarios()
        self.results = []
        self.change_note = change_note
        # Create unique output directory with datetime
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"tests/outputs/run_{self.timestamp}"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

        # Open detailed log file for all console output
        self.log_file = open(Path(self.output_dir) / 'detailed_test_log.txt', 'w')
        self._log(f"Control Function Test Suite - Run {self.timestamp}\n")
        self._log(f"Output directory: {self.output_dir}\n")
        if self.change_note:
            self._log(f"Change Note: {self.change_note}\n")
            self._log("-" * 60 + "\n")

    def _log(self, message: str):
        """Write to both console and log file."""
        print(message, end='')
        self.log_file.write(message)
        self.log_file.flush()

    def __del__(self):
        """Clean up by closing log file."""
        if hasattr(self, 'log_file') and not self.log_file.closed:
            self.log_file.close()

    def _create_test_scenarios(self) -> List[Dict]:
        """Create diverse test scenarios covering different negotiation situations."""
        scenarios = []

        # Scenario 1: Balanced negotiation with reasonable gap
        scenarios.append({
            'name': 'Balanced_Negotiation',
            'description': 'Both parties have reasonable expectations with moderate gap',
            'params': NegotiationParams(
                beta_b=-0.1,  # Buyer willing to pay 10% less than max
                beta_s=0.1,   # Seller wants 10% more than min
                ALPHA=5.0,
                w_g=0.4,
                w_z=0.4,
                w_x=0.2,
                R=20
            ),
            'states': self._generate_negotiation_sequence(
                Ps=100, Pb=150, initial_price=130,
                n_extras_buyer=3, n_extras_seller=2,
                num_turns=20, price_adjustment='converging'
            )
        })

        # Scenario 2: Aggressive buyer vs firm seller
        scenarios.append({
            'name': 'Aggressive_Buyer',
            'description': 'Buyer pushes hard for low price, seller stands firm',
            'params': NegotiationParams(
                beta_b=-0.3,  # Buyer wants 30% discount
                beta_s=0.05,  # Seller slightly flexible
                ALPHA=7.0,    # Higher pressure
                w_g=0.3,
                w_z=0.5,      # Emphasize price range
                w_x=0.2,
                R=25
            ),
            'states': self._generate_negotiation_sequence(
                Ps=200, Pb=250, initial_price=240,
                n_extras_buyer=5, n_extras_seller=3,
                num_turns=25, price_adjustment='slow_descent'
            )
        })

        # Scenario 3: Wide price gap with extras negotiation
        scenarios.append({
            'name': 'Wide_Gap_Extra_Focus',
            'description': 'Large price gap, negotiation focuses on extra items',
            'params': NegotiationParams(
                beta_b=-0.15,
                beta_s=0.2,   # Seller wants significant premium
                ALPHA=4.0,
                w_g=0.2,
                w_z=0.3,
                w_x=0.5,      # Emphasize extras
                R=30
            ),
            'states': self._generate_negotiation_sequence(
                Ps=500, Pb=700, initial_price=650,
                n_extras_buyer=10, n_extras_seller=8,
                num_turns=30, price_adjustment='oscillating',
                extras_pattern='increasing'
            )
        })

        # Scenario 4: Near-deadlock situation
        scenarios.append({
            'name': 'Near_Deadlock',
            'description': 'Parties are far apart with little movement',
            'params': NegotiationParams(
                beta_b=0.0,   # Buyer at max limit
                beta_s=0.3,   # Seller wants high premium
                ALPHA=8.0,    # Very high pressure
                w_g=0.5,
                w_z=0.4,
                w_x=0.1,
                R=15
            ),
            'states': self._generate_negotiation_sequence(
                Ps=1000, Pb=1100, initial_price=1050,
                n_extras_buyer=2, n_extras_seller=1,
                num_turns=15, price_adjustment='stuck'
            )
        })

        # Scenario 5: Quick convergence scenario
        scenarios.append({
            'name': 'Quick_Convergence',
            'description': 'Both parties eager to close, rapid convergence',
            'params': NegotiationParams(
                beta_b=-0.05,
                beta_s=0.0,   # Seller at minimum
                ALPHA=3.0,    # Lower pressure
                w_g=0.3,
                w_z=0.3,
                w_x=0.4,
                hc2=0.5,      # Close action weight 2
                hc=0.8,       # Higher close multiplier
                R=10
            ),
            'states': self._generate_negotiation_sequence(
                Ps=50, Pb=80, initial_price=65,
                n_extras_buyer=1, n_extras_seller=1,
                num_turns=10, price_adjustment='fast_converge'
            )
        })

        # Scenario 6: Extreme personalities
        scenarios.append({
            'name': 'Extreme_Personalities',
            'description': 'Very aggressive buyer meets very greedy seller',
            'params': NegotiationParams(
                beta_b=-0.5,  # Buyer wants 50% off
                beta_s=0.5,   # Seller wants 50% premium
                ALPHA=10.0,   # Extreme pressure
                w_g=0.4,
                w_z=0.4,
                w_x=0.2,
                hl=1.0,       # High clarify weight
                eps=0.001,    # Tight clarify threshold
                R=40
            ),
            'states': self._generate_negotiation_sequence(
                Ps=300, Pb=400, initial_price=350,
                n_extras_buyer=7, n_extras_seller=5,
                num_turns=40, price_adjustment='volatile'
            )
        })

        # Scenario 7: No extras available
        scenarios.append({
            'name': 'No_Extras',
            'description': 'Pure price negotiation without extra items',
            'params': NegotiationParams(
                beta_b=-0.1,
                beta_s=0.1,
                ALPHA=5.0,
                w_g=0.5,      # More weight on price gap
                w_z=0.5,      # More weight on price range
                w_x=0.0,      # No weight on extras
                R=20
            ),
            'states': self._generate_negotiation_sequence(
                Ps=150, Pb=200, initial_price=180,
                n_extras_buyer=0, n_extras_seller=0,
                num_turns=20, price_adjustment='linear'
            )
        })

        # Scenario 8: Edge case - zero price gap
        scenarios.append({
            'name': 'Zero_Gap',
            'description': 'Adjusted prices are identical (edge case)',
            'params': NegotiationParams(
                beta_b=-0.2,
                beta_s=-0.2,  # Both want same adjusted price
                ALPHA=5.0,
                EPSILON=1e-6,
                w_g=0.4,
                w_z=0.4,
                w_x=0.2,
                R=10
            ),
            'states': self._generate_negotiation_sequence(
                Ps=100, Pb=100, initial_price=100,
                n_extras_buyer=2, n_extras_seller=2,
                num_turns=10, price_adjustment='static'
            )
        })

        return scenarios

    def _generate_negotiation_sequence(self, Ps: float, Pb: float, initial_price: float,
                                      n_extras_buyer: int, n_extras_seller: int,
                                      num_turns: int, price_adjustment: str,
                                      extras_pattern: str = 'static') -> List[NegotiationState]:
        """Generate a sequence of negotiation states for testing."""
        states = []
        prev_s = None

        for turn in range(num_turns):
            # Calculate price based on adjustment pattern
            if price_adjustment == 'converging':
                # Prices converge toward middle
                target = (Ps + Pb) / 2
                price = initial_price + (target - initial_price) * (turn / num_turns)
            elif price_adjustment == 'slow_descent':
                # Slow linear descent
                price = initial_price - (initial_price - Ps) * (turn / num_turns) * 0.5
            elif price_adjustment == 'oscillating':
                # Oscillate around middle
                middle = (Ps + Pb) / 2
                amplitude = (Pb - Ps) * 0.2
                price = middle + amplitude * np.sin(turn * np.pi / 5)
            elif price_adjustment == 'stuck':
                # Very little movement
                price = initial_price + np.random.normal(0, 2)
            elif price_adjustment == 'fast_converge':
                # Exponential convergence
                target = (Ps + Pb) / 2
                price = initial_price + (target - initial_price) * (1 - np.exp(-turn/3))
            elif price_adjustment == 'volatile':
                # High volatility
                price = initial_price + np.random.uniform(-20, 20)
            elif price_adjustment == 'linear':
                # Linear interpolation
                price = initial_price + (Ps - initial_price) * (turn / num_turns)
            elif price_adjustment == 'static':
                # No change
                price = initial_price
            else:
                price = initial_price

            # Ensure price stays in reasonable bounds
            price = max(Ps * 0.8, min(Pb * 1.2, price))

            # Calculate extras included based on pattern
            if extras_pattern == 'increasing':
                l_k = min(turn // 3, n_extras_buyer + n_extras_seller)
            elif extras_pattern == 'decreasing':
                l_k = max(0, (n_extras_buyer + n_extras_seller) - turn // 2)
            else:  # static
                l_k = min(2, n_extras_buyer + n_extras_seller) if turn > 5 else 0

            state = NegotiationState(
                r=turn,
                Ps=Ps,
                Pb=Pb,
                P_k=price,
                n_b_k=n_extras_buyer,
                n_s_k=n_extras_seller,
                l_k=l_k,
                prev_s=prev_s
            )
            states.append(state)

            # Update prev_s for next iteration (would normally come from control function)
            # Using a dummy value here
            prev_s = 0.5 + turn * 0.02

        return states

    def run_scenario_tests(self, scenario: Dict) -> Dict:
        """Run tests for a single scenario."""
        self._log(f"\n{'='*60}\n")
        self._log(f"Testing Scenario: {scenario['name']}\n")
        self._log(f"Description: {scenario['description']}\n")
        self._log(f"{'='*60}\n")

        cf = ControlFunction(scenario['params'])
        results = {
            'scenario_name': scenario['name'],
            'description': scenario['description'],
            'params': scenario['params'].__dict__,
            'turns': []
        }

        for state in scenario['states']:
            # Test for both buyer and seller
            for is_buyer in [True, False]:
                role = 'Buyer' if is_buyer else 'Seller'
                evaluation = cf.evaluate(state, is_buyer)

                turn_result = {
                    'turn': state.r,
                    'role': role,
                    'price': state.P_k,
                    'control_value': evaluation['control_value'],
                    'completion': evaluation['features']['s'],
                    'action_probs': evaluation['action_probs'],
                    'recommended_action': evaluation['recommended_action'],
                    'q_values': evaluation['q_values']
                }
                results['turns'].append(turn_result)

                # Print summary for key turns
                if state.r in [0, len(scenario['states'])//2, len(scenario['states'])-1]:
                    self._print_turn_summary(state.r, role, evaluation)

        return results

    def _print_turn_summary(self, turn: int, role: str, evaluation: Dict):
        """Print a summary of a single turn evaluation."""
        self._log(f"\nTurn {turn} - {role}:\n")
        self._log(f"  Control Value (V_k): {evaluation['control_value']:.4f}\n")
        self._log(f"  Completion (s): {evaluation['features']['s']:.2%}\n")
        self._log(f"  Recommended Action: {evaluation['recommended_action']}\n")
        self._log(f"  Action Probabilities:\n")
        for action, prob in evaluation['action_probs'].items():
            self._log(f"    {action:12s}: {prob:.3f}\n")

    def run_all_tests(self):
        """Run all test scenarios."""
        self._log("\n" + "="*60 + "\n")
        self._log("CONTROL FUNCTION TEST SUITE\n")
        self._log("="*60 + "\n")

        all_results = []
        for scenario in self.scenarios:
            results = self.run_scenario_tests(scenario)
            all_results.append(results)
            self.results = all_results

        # Generate summary statistics
        self._generate_summary_statistics(all_results)

        return all_results

    def _generate_summary_statistics(self, results: List[Dict]):
        """Generate and print summary statistics across all scenarios."""
        self._log("\n" + "="*60 + "\n")
        self._log("SUMMARY STATISTICS\n")
        self._log("="*60 + "\n")

        # Also write to summary file
        summary_file = Path(self.output_dir) / 'test_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("CONTROL FUNCTION TEST SUMMARY\n")
            f.write(f"Test Run: {self.timestamp}\n")
            if self.change_note:
                f.write("="*60 + "\n")
                f.write(f"CHANGE NOTE: {self.change_note}\n")
            f.write("="*60 + "\n\n")

            for result in results:
                scenario_name = result['scenario_name']
                turns_data = result['turns']

                # Calculate average completion by role
                buyer_completions = [t['completion'] for t in turns_data if t['role'] == 'Buyer']
                seller_completions = [t['completion'] for t in turns_data if t['role'] == 'Seller']

                # Count action recommendations
                buyer_actions = [t['recommended_action'] for t in turns_data if t['role'] == 'Buyer']
                seller_actions = [t['recommended_action'] for t in turns_data if t['role'] == 'Seller']

                summary = f"\n{scenario_name}:\n"
                summary += f"  Avg Buyer Completion: {np.mean(buyer_completions):.2%}\n"
                summary += f"  Avg Seller Completion: {np.mean(seller_completions):.2%}\n"
                summary += f"  Buyer Action Distribution:\n"
                for action in ['compensate', 'finetune', 'extra', 'close', 'clarify']:
                    count = buyer_actions.count(action)
                    summary += f"    {action}: {count}/{len(buyer_actions)}\n"
                summary += f"  Seller Action Distribution:\n"
                for action in ['compensate', 'finetune', 'extra', 'close', 'clarify']:
                    count = seller_actions.count(action)
                    summary += f"    {action}: {count}/{len(seller_actions)}\n"

                self._log(summary)
                f.write(summary)

            f.write("\n" + "="*60 + "\n")
            self._log(f"\nSummary written to {summary_file}\n")

    def visualize_results(self):
        """Create visualizations of test results."""
        for result in self.results:
            self._create_scenario_visualization(result, self.output_dir)

        self._log(f"\nVisualizations saved to {self.output_dir}/\n")

    def _create_scenario_visualization(self, result: Dict, output_dir: str):
        """Create visualization for a single scenario."""
        scenario_name = result['scenario_name']
        turns_data = result['turns']

        # Prepare data for plotting
        df = pd.DataFrame(turns_data)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Control Function Analysis: {scenario_name}", fontsize=16)

        # Plot 1: Completion over time
        ax1 = axes[0, 0]
        for role in ['Buyer', 'Seller']:
            role_data = df[df['role'] == role]
            ax1.plot(role_data['turn'], role_data['completion'],
                    label=role, marker='o', markersize=4)
        ax1.set_xlabel('Turn')
        ax1.set_ylabel('Completion (s)')
        ax1.set_title('Completion Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Control value over time
        ax2 = axes[0, 1]
        for role in ['Buyer', 'Seller']:
            role_data = df[df['role'] == role]
            ax2.plot(role_data['turn'], role_data['control_value'],
                    label=role, marker='s', markersize=4)
        ax2.set_xlabel('Turn')
        ax2.set_ylabel('Control Value (V_k)')
        ax2.set_title('Control Value Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Action distribution heatmap
        ax3 = axes[1, 0]
        actions = ['compensate', 'finetune', 'extra', 'close', 'clarify']

        # Extract action probabilities
        buyer_probs = []
        for turn in df[df['role'] == 'Buyer']['action_probs']:
            buyer_probs.append([turn[action] for action in actions])

        if buyer_probs:
            buyer_probs_array = np.array(buyer_probs).T
            im = ax3.imshow(buyer_probs_array, aspect='auto', cmap='YlOrRd')
            ax3.set_yticks(range(len(actions)))
            ax3.set_yticklabels(actions)
            ax3.set_xlabel('Turn')
            ax3.set_title('Buyer Action Probabilities')
            plt.colorbar(im, ax=ax3)

        # Plot 4: Price evolution
        ax4 = axes[1, 1]
        unique_turns = df['turn'].unique()
        prices = [df[df['turn'] == t]['price'].iloc[0] for t in unique_turns]
        ax4.plot(unique_turns, prices, 'g-', linewidth=2, label='Negotiated Price')

        # Add reference lines for Ps and Pb
        Ps = result['turns'][0]['price']  # Approximation
        Pb = result['turns'][0]['price']  # Approximation
        ax4.axhline(y=Ps, color='r', linestyle='--', alpha=0.5, label='Seller Min')
        ax4.axhline(y=Pb, color='b', linestyle='--', alpha=0.5, label='Buyer Max')

        ax4.set_xlabel('Turn')
        ax4.set_ylabel('Price')
        ax4.set_title('Price Evolution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{scenario_name.lower().replace(' ', '_')}_analysis.png", dpi=150)
        plt.close()

    def export_results(self):
        """Export test results to JSON file in the timestamped directory."""
        output_file = Path(self.output_dir) / 'control_function_test_results.json'

        # Convert numpy values to Python native types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj

        serializable_results = convert_to_native(self.results)

        # Also add metadata about the test run
        test_metadata = {
            'timestamp': self.timestamp,
            'num_scenarios': len(self.scenarios),
            'scenario_names': [s['name'] for s in self.scenarios],
            'results': serializable_results
        }

        #with open(output_file, 'w') as f:
            #json.dump(test_metadata, f, indent=2)

        print(f"\nResults exported to {output_file}")


def run_edge_case_tests():
    """Test edge cases and boundary conditions."""
    print("\n" + "="*60)
    print("EDGE CASE TESTS")
    print("="*60)

    # Test 1: Division by zero protection
    print("\nTest 1: Zero price denominators")
    params = NegotiationParams(beta_b=-0.1, beta_s=0.1)
    cf = ControlFunction(params)
    state = NegotiationState(r=0, Ps=0, Pb=100, P_k=50)  # Zero Ps
    try:
        result = cf.evaluate(state, is_buyer=True)
        print("  ✓ Handled zero Ps successfully")
    except Exception as e:
        print(f"  ✗ Failed with: {e}")

    # Test 2: Negative prices
    print("\nTest 2: Negative prices")
    state = NegotiationState(r=5, Ps=-100, Pb=100, P_k=0)
    try:
        result = cf.evaluate(state, is_buyer=True)
        print("  ✓ Handled negative prices")
    except Exception as e:
        print(f"  ✗ Failed with: {e}")

    # Test 3: Turn beyond maximum
    print("\nTest 3: Turn beyond maximum")
    state = NegotiationState(r=30, Ps=100, Pb=200, P_k=150)
    params.R = 20  # Max is 20, but turn is 30
    cf = ControlFunction(params)
    try:
        result = cf.evaluate(state, is_buyer=True)
        print(f"  ✓ Handled turn overflow (t_k = {result['base_values']['t_k']:.2f})")
    except Exception as e:
        print(f"  ✗ Failed with: {e}")

    # Test 4: Extreme parameter values
    print("\nTest 4: Extreme parameter values")
    extreme_params = NegotiationParams(
        beta_b=-0.99,  # Extreme discount
        beta_s=10.0,   # Extreme premium
        ALPHA=100.0,   # Very high pressure
        w_g=0.0,
        w_z=0.0,
        w_x=1.0        # Only extras matter
    )
    cf = ControlFunction(extreme_params)
    state = NegotiationState(r=10, Ps=100, Pb=200, P_k=150,
                            n_b_k=10, n_s_k=10, l_k=5)
    try:
        result = cf.evaluate(state, is_buyer=True)
        print("  ✓ Handled extreme parameters")
    except Exception as e:
        print(f"  ✗ Failed with: {e}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Control Function Test Suite')
    parser.add_argument('--note', type=str, help='Change note to document what was modified', default=None)
    parser.add_argument('--skip-viz', action='store_true', help='Skip visualization generation')
    args = parser.parse_args()

    # Run main test suite with change note
    tester = ControlFunctionTester(change_note=args.note)
    results = tester.run_all_tests()

    # Run edge case tests
    run_edge_case_tests()

    # Generate visualizations unless skipped
    if not args.skip_viz:
        try:
            tester.visualize_results()
        except ImportError:
            print("\n[Note: Visualizations skipped - matplotlib/seaborn not available]")

    # Export results
    tester.export_results()

    print("\n" + "="*60)
    print("TEST SUITE COMPLETE")
    print("="*60)

    if args.note:
        print(f"Change Note: {args.note}")
    print(f"All outputs saved to: {tester.output_dir}")