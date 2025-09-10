# Key Files for Two-Part Response Implementation

  ## Core Agent Files (Primary modifications)

  1. agents/ollama_wrapper.py - Main implementation site
    - Add generate_strategy() and generate_response() methods
    - Modify act() to orchestrate 2-part flow
    - Add _build_strategy_prompt() and _build_response_prompt()
    - Update token tracking
  2. agents/negotiator_base.py - Base class updates
    - Add strategy history tracking
    - Enhance metrics structure for dual-phase tracking
    - Add strategy validation methods

  ## Schema Files (Define structures)

  3. schema/strategy_template.py
  4. schema/sft_packet_template.jsonl - Update for phase separation
    - Add phase-specific template variations

  ## World/Simulation Files

  5. worlds/selfchat_world.py - Episode logging updates
    - Modify run_episode() to capture both phases (lines 231-390)
    - Update log_episode() to include strategy data (line 469)
    - Enhance metrics calculation (lines 376-388)
  6. tasks/selfplay_teacher.py - May need strategy seed support
    - Check if opening prompts need strategy awareness

  ## Training Data Converters

  7. training/converters/sft_converter.py - Add dual-phase conversion
    - Add episode_to_sft_separated() method
    - Update output directory structure
  8. training/converters/dpo_converter.py - Strategy-aware DPO pairs
    - Create strategy comparison pairs
    - Response quality pairs given same strategy

  ## Configuration Files

  9. training_parameters.py - Add new parameters
    - USE_TWO_PART_RESPONSE = True
    - STRATEGY_MAX_TOKENS = 150
    - RESPONSE_MAX_TOKENS = 256
  10. config/parlai_config.py - Update model configs
    - Add strategy generation temperature settings

  ## Evaluation/Metrics

  11. evaluation/episode_metrics.py - Update metrics calculation
    - Add strategy token counting (lines 87-96)
    - Separate phase performance metrics
  12. evaluation/session_metrics.py - Aggregate phase metrics
    - Add strategy vs response throughput (lines 54-62)

  ## Testing Files

  13. tests/test_selfchat.py - Update/add tests for 2-part flow
  14. run_selfchat.py - May need CLI arg for enabling/disabling 2-part