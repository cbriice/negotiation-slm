"""
Simple prompt builder for negotiation scenarios.
Creates clear, natural prompts without complex state serialization.
"""

from typing import Dict, Any, List, Optional, Tuple


class PromptBuilder:
    """Build clear prompts for negotiation scenarios."""

    def __init__(self, include_strategy_hints: bool = False):
        """
        Initialize prompt builder.

        Args:
            include_strategy_hints: Whether to include negotiation strategy hints
        """
        self.include_strategy_hints = include_strategy_hints

    def build_initial_prompt(self,
                            role: str,
                            item: Dict[str, Any],
                            role_params: Dict[str, Any],
                            conversation_history: Optional[List[Tuple[str, str]]] = None) -> str:
        """
        Build a prompt for a negotiation agent.

        Args:
            role: 'buyer' or 'seller'
            item: Item details (name, price, condition, etc.)
            role_params: Role-specific parameters (target_price, min_price, etc.)
            conversation_history: List of (speaker, message) tuples

        Returns:
            Complete prompt string
        """
        # Build role-specific context
        if role == 'buyer':
            prompt = self._build_buyer_prompt(item, role_params)
        elif role == 'seller':
            prompt = self._build_seller_prompt(item, role_params)
        else:
            raise ValueError(f"Unknown role: {role}")

        # Add conversation history if present
        if conversation_history:
            prompt += "\n\n### Conversation so far:\n"
            for speaker, message in conversation_history:
                # Capitalize the speaker role for clarity
                prompt += f"{speaker.capitalize()}: {message}\n"

            # Add turn indicator
            prompt += f"\n{role.capitalize()}: "
        else:
            # Starting the conversation
            prompt += f"\n\n### Start the negotiation:\n{role.capitalize()}: "

        return prompt

    def _build_buyer_prompt(self, item: Dict[str, Any], params: Dict[str, Any]) -> str:
        """Build buyer-specific prompt."""
        prompt = f"""You are a buyer interested in purchasing a {item.get('name', 'product')}.

### Item Details:
- Product: {item.get('name', 'Unknown')}"""

        if item.get('brand'):
            prompt += f"\n- Brand: {item['brand']}"
        if item.get('model'):
            prompt += f"\n- Model: {item['model']}"
        if item.get('condition'):
            prompt += f"\n- Condition: {item['condition']}"

        prompt += f"\n- Seller's asking price: ${item.get('list_price', 'Unknown')}"

        prompt += f"""

### Your Budget:
- Maximum you can pay: ${params.get('max_price', 'flexible')}
- Target price (ideal): ${params.get('target_price', 'negotiable')}"""

        # Add optional strategy hints
        if self.include_strategy_hints:
            prompt += """

### Negotiation Tips:
- Start with a reasonable offer below your target
- Be willing to negotiate but stay within budget
- Ask about the item's condition and features
- Consider walking away if price exceeds your maximum"""

        return prompt

    def _build_seller_prompt(self, item: Dict[str, Any], params: Dict[str, Any]) -> str:
        """Build seller-specific prompt."""
        prompt = f"""You are selling your {item.get('name', 'item')}.

### Your Item:
- Product: {item.get('name', 'Unknown')}"""

        if item.get('brand'):
            prompt += f"\n- Brand: {item['brand']}"
        if item.get('model'):
            prompt += f"\n- Model: {item['model']}"
        if item.get('condition'):
            prompt += f"\n- Condition: {item['condition']}"

        prompt += f"\n- Your asking price: ${item.get('list_price', 'Unknown')}"

        prompt += f"""

### Your Limits:
- Minimum acceptable price: ${params.get('min_price', 'negotiable')}"""

        # Add any issues if present
        if params.get('item_issues'):
            prompt += f"\n- Known issues: {', '.join(params['item_issues'])}"

        # Add optional strategy hints
        if self.include_strategy_hints:
            prompt += """

### Negotiation Tips:
- Highlight the item's positive features
- Be firm but fair in your pricing
- Consider reasonable offers above your minimum
- Be prepared to explain your pricing"""

        return prompt

    def build_continuation_prompt(self,
                                role: str,
                                conversation_history: List[Tuple[str, str]]) -> str:
        """
        Build a simple continuation prompt with just the conversation history.
        This is for ongoing turns after the initial context has been established.

        Args:
            role: Current speaker role
            conversation_history: Full conversation history

        Returns:
            Prompt for continuation
        """
        prompt = "### Negotiation in progress:\n"

        for speaker, message in conversation_history:
            prompt += f"{speaker.capitalize()}: {message}\n"

        prompt += f"\n{role.capitalize()}: "
        return prompt

    def extract_response(self, generated_text: str, role: str) -> str:
        """
        Extract the actual response from generated text.
        Handles cases where the model might repeat the role label.

        Args:
            generated_text: Raw generated text
            role: The role that was generating

        Returns:
            Clean response text
        """
        # Remove role prefix if model repeated it
        role_prefixes = [
            f"{role.capitalize()}: ",
            f"{role}: ",
            f"{role.upper()}: "
        ]

        text = generated_text.strip()
        for prefix in role_prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        return text

    def format_for_logging(self,
                          conversation_history: List[Tuple[str, str]],
                          deal_outcome: Optional[Dict[str, Any]] = None) -> str:
        """
        Format a conversation for logging/display.

        Args:
            conversation_history: Complete conversation
            deal_outcome: Optional deal details (price, terms, etc.)

        Returns:
            Formatted string for display
        """
        output = "=" * 60 + "\n"
        output += "NEGOTIATION TRANSCRIPT\n"
        output += "=" * 60 + "\n\n"

        for i, (speaker, message) in enumerate(conversation_history, 1):
            output += f"[Turn {i}] {speaker.upper()}:\n{message}\n\n"

        if deal_outcome:
            output += "-" * 60 + "\n"
            output += "OUTCOME:\n"
            if deal_outcome.get('deal_made'):
                output += f"✓ Deal reached at ${deal_outcome.get('final_price', 'Unknown')}\n"
                if deal_outcome.get('terms'):
                    output += f"  Terms: {deal_outcome['terms']}\n"
            else:
                output += "✗ No deal reached\n"
                if deal_outcome.get('reason'):
                    output += f"  Reason: {deal_outcome['reason']}\n"

        output += "=" * 60 + "\n"
        return output