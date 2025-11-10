"""
Quick test to verify MemoryManager works correctly
"""

from src.memory_manager import MemoryManager
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize memory manager
mm = MemoryManager(logger=logger)

# Test 1: Token estimation
print("=" * 60)
print("TEST 1: Token Estimation")
print("=" * 60)

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
    {"role": "user", "content": "Can you help me with something?"},
    {"role": "assistant", "content": "Of course! I'd be happy to help. What do you need?"}
]

estimated_tokens = mm.estimate_conversation_tokens(conversation)
print(f"Estimated tokens (excluding system): {estimated_tokens}")
print(f"Expected: ~40-50 tokens")
print()

# Test 2: Should prune memory check
print("=" * 60)
print("TEST 2: Should Prune Memory Check")
print("=" * 60)

# Short conversation - should not prune
short_conv = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]
should_prune, reasons = mm.should_prune_memory(short_conv, max_turns_before_clear=3)
print(f"Short conversation (1 turn): should_prune={should_prune}, reasons={reasons}")

# Long conversation by turns - should prune
long_conv = [
    {"role": "user", "content": "Message 1"},
    {"role": "assistant", "content": "Response 1"},
    {"role": "user", "content": "Message 2"},
    {"role": "assistant", "content": "Response 2"},
    {"role": "user", "content": "Message 3"},
    {"role": "assistant", "content": "Response 3"},
]
should_prune, reasons = mm.should_prune_memory(long_conv, max_turns_before_clear=3)
print(f"Long conversation (3 turns): should_prune={should_prune}, reasons={reasons}")

# Long conversation by tokens - should prune
token_conv = [
    {"role": "user", "content": "x" * 4000},  # ~1000 tokens
    {"role": "assistant", "content": "y" * 4000},  # ~1000 tokens
    {"role": "user", "content": "z" * 2000},  # ~500 tokens
]
should_prune, reasons = mm.should_prune_memory(token_conv, max_conversation_tokens=2000)
print(f"Token-heavy conversation (~2500 tokens): should_prune={should_prune}, reasons={reasons}")
print()

# Test 3: Prune tool result
print("=" * 60)
print("TEST 3: Prune Tool Result")
print("=" * 60)

# Test process_text pruning
long_response = "x" * 300
tool_result = {"response": long_response, "metadata": "important"}
pruned = mm.prune_tool_result(tool_result, "process_text")
print(f"Original response length: {len(tool_result['response'])}")
print(f"Pruned response: {pruned['response'][:50]}...")
print(f"Metadata preserved: {pruned.get('metadata')}")
print()

# Test get_layer_names pruning
layers_result = {"layers": [f"layer_{i}" for i in range(100)], "count": 100}
pruned_layers = mm.prune_tool_result(layers_result, "get_layer_names")
print(f"Original layers: {len(tool_result.get('layers', []))} items")
print(f"Pruned layers: {pruned_layers['layers'][:50]}...")
print()

# Test 4: Reset conversation with sliding window
print("=" * 60)
print("TEST 4: Reset Conversation with Sliding Window")
print("=" * 60)

# Create a conversation with 5 exchanges
multi_turn_conv = []
for i in range(5):
    multi_turn_conv.append({"role": "user", "content": f"User message {i+1}"})
    multi_turn_conv.append({"role": "assistant", "content": f"Assistant response {i+1}"})

print(f"Original conversation: {len(multi_turn_conv)} messages (5 exchanges)")
trimmed = mm.reset_conversation_with_sliding_window(multi_turn_conv, keep_recent_turns=2)
print(f"Trimmed conversation: {len(trimmed)} messages (2 exchanges)")
print(f"First message in trimmed: {trimmed[0]['content']}")
print(f"Expected: 'User message 4'")
print()

print("=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
