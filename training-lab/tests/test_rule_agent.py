#!/usr/bin/env python3
"""Test script for RuleAgent functionality."""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.rule_agent import RuleAgent
from agents.negotiator_base import NegotiationAction
from schema.rule_agent.personas import PersonaType

def test_basic_initialization():
    """Test basic agent initialization."""
    print("Testing basic initialization...")
    
    opt = {
        'persona_type': 'aggressive',
        'strict_rules': True,
        'max_turns': 20
    }
    
    agent = RuleAgent(opt)
    print(f"✓ Agent initialized with persona: {agent.persona['type']}")
    print(f"  Traits: {agent.traits}")
    print(f"  Preferred tactics: {agent.preferred_tactics[:3]}...")
    return agent

def test_state_initialization():
    """Test initialization from state."""
    print("\nTesting state initialization...")
    
    # Sample state from state_template
    state = {
        "dialogue_id": "test_001",
        "item": {
            "name": "Used Laptop",
            "condition": "good",
            "list_price": 1000,
            "description": "2-year old laptop in good condition"
        },
        "buyer_state": {
            "max_price": 900,
            "target_price": 750,
            "urgency": 0.6
        },
        "seller_state": {
            "min_price": 600,
            "target_price": 850,
            "flexibility": 0.4
        }
    }
    
    # Test as buyer
    opt = {'persona_type': 'cooperative', 'max_turns': 20}
    buyer_agent = RuleAgent(opt)
    buyer_agent.initialize_from_state(state, 'buyer')
    
    print(f"✓ Buyer agent initialized")
    print(f"  Side: {buyer_agent.side}")
    print(f"  Max price: ${buyer_agent.role_state.get('max_price')}")
    print(f"  Target price: ${buyer_agent.role_state.get('target_price')}")
    
    # Test as seller
    seller_agent = RuleAgent(opt)
    seller_agent.initialize_from_state(state, 'seller')
    
    print(f"✓ Seller agent initialized")
    print(f"  Side: {seller_agent.side}")
    print(f"  Min price: ${seller_agent.role_state.get('min_price')}")
    print(f"  Target price: ${seller_agent.role_state.get('target_price')}")
    
    return buyer_agent, seller_agent

def test_negotiation_actions():
    """Test parsing and generating negotiation actions."""
    print("\nTesting negotiation actions...")
    
    opt = {'persona_type': 'analytical', 'max_turns': 20}
    agent = RuleAgent(opt)
    
    # Test parsing actions
    test_messages = [
        "[PROPOSE] I'll offer $500 for the laptop",
        "[COMMIT] Final offer: $750",
        "[ACK] Deal accepted!",
        "[NACK] Can't accept that price",
        "[NO_DEAL] Walking away from this negotiation"
    ]
    
    for msg in test_messages:
        action, cleaned = agent.parse_negotiation_action(msg)
        if action:
            print(f"✓ Parsed {action.value} from: '{msg[:30]}...'")
    
    return agent

def test_price_generation():
    """Test price generation for different personas and sides."""
    print("\nTesting price generation...")
    
    state = {
        "item": {"name": "Test Item", "list_price": 1000},
        "buyer_state": {"max_price": 900, "target_price": 750},
        "seller_state": {"min_price": 600, "target_price": 850}
    }
    
    # Test different personas as buyer
    for persona in ['aggressive', 'cooperative', 'sympathetic']:
        opt = {'persona_type': persona, 'max_turns': 20}
        agent = RuleAgent(opt)
        agent.initialize_from_state(state, 'buyer')
        
        price = agent._generate_price()
        print(f"✓ {persona} buyer opening price: ${price:.0f}")
    
    # Test different personas as seller
    for persona in ['aggressive', 'analytical', 'impatient']:
        opt = {'persona_type': persona, 'max_turns': 20}
        agent = RuleAgent(opt)
        agent.initialize_from_state(state, 'seller')
        
        price = agent._generate_price()
        print(f"✓ {persona} seller opening price: ${price:.0f}")

def test_response_generation():
    """Test full response generation."""
    print("\nTesting response generation...")
    
    state = {
        "dialogue_id": "test_002",
        "item": {
            "name": "Gaming Console",
            "condition": "excellent",
            "list_price": 500
        },
        "buyer_state": {"max_price": 450, "target_price": 400},
        "seller_state": {"min_price": 350, "target_price": 450}
    }
    
    # Create buyer agent
    opt = {'persona_type': 'diplomatic', 'max_turns': 20}
    agent = RuleAgent(opt)
    agent.initialize_from_state(state, 'buyer')
    
    # Generate opening response
    response = agent.act()
    print(f"✓ Generated opening response:")
    print(f"  Text: {response['text'][:100]}...")
    print(f"  Episode done: {response['episode_done']}")
    
    # Simulate receiving a seller response
    seller_msg = {
        'text': '[PROPOSE] I\'m asking for $480, which is already below list price',
        'episode_done': False
    }
    agent.observe(seller_msg)
    
    # Generate counter-response
    response = agent.act()
    print(f"✓ Generated counter-response:")
    print(f"  Text: {response['text'][:100]}...")
    
    return agent

def test_rule_enforcement():
    """Test rule enforcement and violation detection."""
    print("\nTesting rule enforcement...")
    
    opt = {
        'persona_type': 'aggressive',
        'strict_rules': True,
        'terminate_on_violation': True,
        'max_turns': 20
    }
    agent = RuleAgent(opt)
    agent.initialize_from_state({
        "item": {"name": "Test", "list_price": 1000},
        "buyer_state": {"max_price": 900, "target_price": 750}
    }, 'buyer')
    
    # Test violation detection
    violation_messages = [
        "I want to buy this",  # Missing action tag
        "[INVALID_TAG] This is wrong",  # Invalid action
        "[PROPOSE] " + "x" * 600,  # Too long
        "[PROPOSE] You're an idiot if you don't take this",  # Personal attack
    ]
    
    for msg in violation_messages:
        observation = {'text': msg, 'episode_done': False}
        agent.observe(observation)
        
        if agent.partner_violations:
            print(f"✓ Detected violations in: '{msg[:50]}...'")
            print(f"  Violations: {list(agent.partner_violations.keys())}")
        
        # Reset for next test
        agent.partner_violations = {}
        agent.terminated = False
    
    return agent

def test_persona_blending():
    """Test persona blending functionality."""
    print("\nTesting persona blending...")
    
    opt = {
        'persona_blend': 'aggressive,cooperative',
        'persona_intensity': 0.8,
        'max_turns': 20
    }
    agent = RuleAgent(opt)
    
    print(f"✓ Blended persona created")
    print(f"  Type: {agent.persona['type']}")
    print(f"  Traits: {agent.traits}")
    print(f"  Combines aggressive and cooperative behaviors")

def test_negotiation_simulation():
    """Simulate a simple negotiation between two RuleAgents."""
    print("\nTesting negotiation simulation...")
    
    state = {
        "dialogue_id": "test_sim",
        "item": {
            "name": "Vintage Watch",
            "condition": "excellent",
            "list_price": 2000
        },
        "buyer_state": {"max_price": 1800, "target_price": 1500},
        "seller_state": {"min_price": 1400, "target_price": 1700}
    }
    
    # Create agents
    buyer_opt = {'persona_type': 'analytical', 'max_turns': 15}
    seller_opt = {'persona_type': 'diplomatic', 'max_turns': 15}
    
    buyer = RuleAgent(buyer_opt)
    seller = RuleAgent(seller_opt)
    
    buyer.initialize_from_state(state, 'buyer')
    seller.initialize_from_state(state, 'seller')
    
    print("Starting negotiation simulation...")
    print("-" * 50)
    
    # Run negotiation for a few turns
    for turn in range(5):
        # Buyer's turn
        buyer_response = buyer.act()
        print(f"Buyer (turn {turn}): {buyer_response['text'][:80]}...")
        
        if buyer_response['episode_done']:
            print("Negotiation ended by buyer")
            break
        
        # Seller observes and responds
        seller.observe(buyer_response)
        seller_response = seller.act()
        print(f"Seller (turn {turn}): {seller_response['text'][:80]}...")
        
        if seller_response['episode_done']:
            print("Negotiation ended by seller")
            break
        
        # Buyer observes seller's response
        buyer.observe(seller_response)
    
    print("-" * 50)
    print("✓ Simulation completed")
    
    # Print summaries
    buyer_summary = buyer.get_agent_summary()
    seller_summary = seller.get_agent_summary()
    
    print(f"\nBuyer Summary:")
    print(f"  Turns: {buyer_summary['turns']}")
    print(f"  Tactics used: {buyer_summary['tactics_used'][:3]}")
    print(f"  Deal closed: {buyer_summary['deal_closed']}")
    
    print(f"\nSeller Summary:")
    print(f"  Turns: {seller_summary['turns']}")
    print(f"  Tactics used: {seller_summary['tactics_used'][:3]}")
    print(f"  Deal closed: {seller_summary['deal_closed']}")

def main():
    """Run all tests."""
    print("=" * 60)
    print("RULEAGENT FUNCTIONALITY TEST SUITE")
    print("=" * 60)
    
    try:
        # Run tests
        test_basic_initialization()
        test_state_initialization()
        test_negotiation_actions()
        test_price_generation()
        test_response_generation()
        test_rule_enforcement()
        test_persona_blending()
        test_negotiation_simulation()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()