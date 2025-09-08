"""
Test script for negotiation teachers without worlds.
Teachers can be tested independently by iterating through their data.
"""

import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from parlai.core.params import ParlaiParser
from tasks.negotiation_teacher import NegotiationJSONTeacher
from tasks.selfplay_teacher import SelfPlayTeacher
from tasks.rule_opponent_teacher import RuleOpponentTeacher
from tasks.human_eval_teacher import HumanEvalTeacher


def test_teacher_basic(teacher_class, teacher_name, custom_opts=None):
    """Test basic functionality of a teacher."""
    print(f"\n{'='*60}")
    print(f"Testing {teacher_name}")
    print('='*60)
    
    # Create parser and add teacher args
    parser = ParlaiParser(add_model_args=False)
    
    # Parse first to get partial_opt
    opt = parser.parse_args([], print_args=False)
    opt['datatype'] = 'train'
    
    # Now add teacher args with partial_opt
    teacher_class.add_cmdline_args(parser, partial_opt=opt)
    
    # Add any custom options
    if custom_opts:
        opt.update(custom_opts)
    
    # Initialize teacher
    try:
        teacher = teacher_class(opt)
        print(f"✓ {teacher_name} initialized successfully")
        
        # Test data iteration
        print(f"\nIterating through {teacher_name} data:")
        num_episodes = 0
        num_examples = 0
        sample_obs = None
        
        for example in teacher.setup_data():
            if example:
                obs, episode_done = example
                num_examples += 1
                if episode_done:
                    num_episodes += 1
                
                # Store first observation as sample
                if num_examples == 1:
                    sample_obs = obs
                
                # Print first few examples
                if num_examples <= 3:
                    print(f"\n  Example {num_examples}:")
                    print(f"    Episode ID: {obs.get('episode_id', 'N/A')}")
                    print(f"    Side: {obs.get('side', 'N/A')}")
                    print(f"    Episode Done: {episode_done}")
                    if 'text' in obs:
                        text_preview = obs['text'][:100] + '...' if len(obs['text']) > 100 else obs['text']
                        print(f"    Text Preview: {text_preview}")
                    if 'labels' in obs and obs['labels']:
                        label_preview = obs['labels'][0][:50] + '...' if len(obs['labels'][0]) > 50 else obs['labels'][0]
                        print(f"    Label Preview: {label_preview}")
        
        print(f"\n  Total Examples: {num_examples}")
        print(f"  Total Episodes: {num_episodes}")
        
        # Test metric functions if available
        if hasattr(teacher, 'get_aggregate_metrics'):
            metrics = teacher.get_aggregate_metrics()
            if metrics:
                print(f"\n  Aggregate Metrics: {metrics}")
        
        return True, sample_obs
        
    except Exception as e:
        print(f"✗ Failed to initialize {teacher_name}: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_selfplay_specific():
    """Test SelfPlay teacher specific features."""
    print("\n" + "="*60)
    print("Testing SelfPlay Teacher Specific Features")
    print("="*60)
    
    parser = ParlaiParser(add_model_args=False)
    
    # Parse first to get partial_opt
    opt = parser.parse_args([], print_args=False)
    opt['datatype'] = 'train'
    
    # Now add teacher args with partial_opt
    SelfPlayTeacher.add_cmdline_args(parser, partial_opt=opt)
    opt['seed_mutation_rate'] = 0.5
    opt['opening_prompt_style'] = 'structured'
    opt['balanced_sides'] = True
    
    teacher = SelfPlayTeacher(opt)
    
    # Test state seed generation
    if not teacher.state_seeds:
        print("Generating default state seeds...")
        teacher.state_seeds = teacher._generate_default_seeds()
        print(f"✓ Generated {len(teacher.state_seeds)} state seeds")
    
    # Test seed mutation
    if teacher.state_seeds:
        original_seed = teacher.state_seeds[0]
        mutated_seed = teacher._mutate_seed(original_seed)
        print("\n✓ Seed mutation test:")
        print(f"  Original min_price: {original_seed.get('seller_state', {}).get('min_price')}")
        print(f"  Mutated min_price: {mutated_seed.get('seller_state', {}).get('min_price')}")
        if 'mutations' in mutated_seed.get('seed_metadata', {}):
            print(f"  Applied mutations: {mutated_seed['seed_metadata']['mutations']}")
    
    # Test regression seeds
    teacher._select_regression_seeds()
    print(f"\n✓ Selected {len(teacher.regression_seeds)} regression seeds")
    
    # Test opening prompt generation
    if teacher.state_seeds:
        for style in ['minimal', 'guided', 'structured']:
            teacher.opening_style = style
            prompt = teacher._generate_opening_prompt(teacher.state_seeds[0], 'buyer')
            print(f"\n  {style.capitalize()} opening: {prompt[:80]}...")


def test_rule_opponent_specific():
    """Test RuleOpponent teacher specific features."""
    print("\n" + "="*60)
    print("Testing RuleOpponent Teacher Specific Features")
    print("="*60)
    
    parser = ParlaiParser(add_model_args=False)
    
    # Parse first to get partial_opt
    opt = parser.parse_args([], print_args=False)
    opt['datatype'] = 'train'
    
    # Now add teacher args with partial_opt
    RuleOpponentTeacher.add_cmdline_args(parser, partial_opt=opt)
    opt['persona_selection'] = 'blended'
    opt['persona_intensity'] = 0.8
    opt['conversation_complexity'] = 'complex'
    
    teacher = RuleOpponentTeacher(opt)
    
    # Test persona selection
    print("\nTesting persona selection:")
    for selection_type in ['random', 'sequential', 'difficulty']:
        teacher.persona_selection = selection_type
        teacher.persona_idx = 0
        personas_selected = []
        for i in range(5):
            persona = teacher._select_next_persona()
            personas_selected.append(persona['type'])
        print(f"  {selection_type}: {personas_selected[:3]}...")
    
    # Test persona blending
    from schema.rule_agent.personas import PersonaType
    print("\n✓ Testing persona blending:")
    blended = teacher.persona_manager.blend_personas(
        PersonaType.AGGRESSIVE, 
        PersonaType.SYMPATHETIC,
        blend_ratio=0.7
    )
    print(f"  Blended type: {blended['type']}")
    print(f"  Traits: {blended['template']['traits']}")
    
    # Test conversation structure generation
    print("\n✓ Testing conversation structure generation:")
    for complexity in ['simple', 'medium', 'complex']:
        teacher.complexity = complexity
        structure = teacher._generate_conversation_structure(blended)
        print(f"  {complexity}: {len(structure['tactics'])} tactics, {len(structure['language_patterns'])} patterns")


def test_human_eval_specific():
    """Test HumanEval teacher specific features."""
    print("\n" + "="*60)
    print("Testing HumanEval Teacher Specific Features")
    print("="*60)
    
    parser = ParlaiParser(add_model_args=False)
    
    # Parse first to get partial_opt
    opt = parser.parse_args([], print_args=False)
    opt['datatype'] = 'train'
    
    # Now add teacher args with partial_opt
    HumanEvalTeacher.add_cmdline_args(parser, partial_opt=opt)
    opt['eval_scenario_type'] = 'edge_cases'
    opt['include_edge_cases'] = True
    opt['separate_eval_logs'] = True
    
    teacher = HumanEvalTeacher(opt)
    
    # Generate evaluation scenarios
    if not teacher.eval_scenarios:
        teacher.eval_scenarios = teacher._generate_eval_scenarios()
    
    print(f"\n✓ Generated {len(teacher.eval_scenarios)} evaluation scenarios")
    
    # Show scenario types
    scenario_types = {}
    for scenario in teacher.eval_scenarios:
        stype = scenario.get('type', 'unknown')
        scenario_types[stype] = scenario_types.get(stype, 0) + 1
    print(f"  Scenario breakdown: {scenario_types}")
    
    # Test feedback prompts
    feedback_prompts = teacher.get_human_feedback_prompt()
    print(f"\n✓ Human feedback prompts ({len(feedback_prompts)} questions):")
    for i, prompt in enumerate(feedback_prompts[:3], 1):
        print(f"  {i}. {prompt}")
    
    # Test session logging
    print(f"\n✓ Session ID: {teacher.session_id}")
    print(f"  Separate logs: {teacher.separate_logs}")
    if teacher.separate_logs:
        print(f"  Log directory: {teacher.eval_log_dir}")


def test_data_persistence():
    """Test saving and loading of episodes."""
    print("\n" + "="*60)
    print("Testing Data Persistence")
    print("="*60)
    
    # Create a teacher with some dummy episodes
    parser = ParlaiParser(add_model_args=False)
    
    # Parse first to get partial_opt
    opt = parser.parse_args([], print_args=False)
    opt['datatype'] = 'train'
    
    # Now add teacher args with partial_opt
    NegotiationJSONTeacher.add_cmdline_args(parser, partial_opt=opt)
    
    teacher = NegotiationJSONTeacher(opt)
    
    # Add dummy episodes
    dummy_episodes = [
        {
            'state': {'item': {'name': 'Test Item 1'}},
            'dialogue': [
                {'text': 'Hello', 'response': 'Hi there', 'side': 'buyer'},
                {'text': 'Want to buy?', 'response': 'Yes', 'side': 'seller'}
            ],
            'meta': {'episode_id': 'test_1', 'dialogue_id': 'test_d1'}
        },
        {
            'state': {'item': {'name': 'Test Item 2'}},
            'dialogue': [
                {'text': 'Another test', 'response': 'Response', 'side': 'buyer'}
            ],
            'meta': {'episode_id': 'test_2', 'dialogue_id': 'test_d2'}
        }
    ]
    
    teacher.episodes = dummy_episodes
    
    # Test saving
    test_file = Path('data/test_episodes.jsonl')
    test_file.parent.mkdir(parents=True, exist_ok=True)
    teacher.save_episodes(test_file)
    print(f"✓ Saved {len(dummy_episodes)} episodes to {test_file}")
    
    # Test loading
    loaded_episodes = []
    with open(test_file, 'r') as f:
        for line in f:
            if line.strip():
                loaded_episodes.append(json.loads(line))
    
    print(f"✓ Loaded {len(loaded_episodes)} episodes from file")
    assert len(loaded_episodes) == len(dummy_episodes), "Episode count mismatch!"
    
    # Clean up
    test_file.unlink()
    print("✓ Data persistence test passed")


def main():
    """Run all teacher tests."""
    print("="*60)
    print("NEGOTIATION TEACHERS TEST SUITE")
    print("="*60)
    
    # Test each teacher's basic functionality
    teachers_to_test = [
        (NegotiationJSONTeacher, "NegotiationJSONTeacher (Base)"),
        (SelfPlayTeacher, "SelfPlayTeacher"),
        (RuleOpponentTeacher, "RuleOpponentTeacher"),
        (HumanEvalTeacher, "HumanEvalTeacher")
    ]
    
    results = {}
    for teacher_class, name in teachers_to_test:
        success, sample = test_teacher_basic(teacher_class, name)
        results[name] = success
    
    # Test specific features
    test_selfplay_specific()
    test_rule_opponent_specific()
    test_human_eval_specific()
    test_data_persistence()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check output above for details.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)