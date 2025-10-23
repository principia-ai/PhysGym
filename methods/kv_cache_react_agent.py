"""
KV Cache Optimized ReAct Agent for Physics Discovery

This implementation optimizes multi-turn conversations by reusing KV cache,
only adding new tokens for each turn instead of regenerating the full context.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from physgym.interface import ResearchInterface

# For KV cache support
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# For vLLM backend (more efficient KV caching)
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


class KVCacheReActAgent:
    """
    ReAct agent optimized for multi-turn conversations with KV cache reuse.
    
    Key optimizations:
    1. Maintains KV cache across turns
    2. Only adds new tokens for each turn
    3. Supports both local models (transformers) and vLLM
    """
    
    def __init__(self,
                 model_name: str = "microsoft/DialoGPT-medium",
                 backend: str = "transformers",  # "transformers" or "vllm"
                 max_steps: int = 20,
                 temperature: float = 0.7,
                 max_new_tokens: int = 512,
                 device: str = "auto"):
        """
        Initialize KV Cache optimized ReAct agent.
        
        Args:
            model_name: HuggingFace model name or path
            backend: "transformers" for HF models, "vllm" for optimized inference
            max_steps: Maximum ReAct steps
            temperature: Sampling temperature
            max_new_tokens: Max tokens to generate per turn
            device: Device to run model on
        """
        self.model_name = model_name
        self.backend = backend
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        # Initialize model and tokenizer
        if backend == "vllm" and HAS_VLLM:
            self._init_vllm()
        elif backend == "transformers" and HAS_TORCH:
            self._init_transformers(device)
        else:
            raise ValueError(f"Backend {backend} not available or dependencies missing")
        
        # Conversation state
        self.conversation_history = []
        self.tokenized_history = []
        self.kv_cache = None
        self.step_count = 0
        
    def _init_vllm(self):
        """Initialize vLLM backend for efficient inference."""
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            stop=["Observation:", "\n\n"]  # Stop tokens for ReAct
        )
        
    def _init_transformers(self, device: str):
        """Initialize transformers backend with KV cache support."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device if device != "auto" else None
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.generation_config = GenerationConfig(
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True  # Enable KV caching
        )
        
    def _build_system_prompt(self, interface: ResearchInterface) -> str:
        """Build the system prompt that stays constant across turns."""
        return f"""You are a physics researcher discovering equations through experiments.

Environment Information:
- Problem ID: {interface.env.id}
- Parameters: {', '.join(interface.all_params)}
- Sample quota: {interface.sample_quota}
- Test quota: {interface.test_quota}
- Problem: {interface.problem_content[:200]}...

Available Actions:
1. run_experiment: {{'type': 'run_experiment', 'params': {{'param1': value1}}}}
2. test_hypothesis: {{'type': 'test_hypothesis', 'hypothesis': 'math_expression'}}
3. get_status: {{'type': 'get_status'}}

Instructions:
- Follow ReAct: Thought -> Action -> Observation
- Be strategic with limited quotas
- Find the underlying equation

"""

    def _add_turn_prompt(self, observation: str = "", quotas: Dict = None) -> str:
        """Build prompt for new turn (only new information)."""
        prompt = ""
        
        if observation:
            prompt += f"Observation: {observation}\n\n"
            
        if quotas:
            prompt += f"Remaining - Samples: {quotas['sample']}, Tests: {quotas['test']}\n"
            
        prompt += "Your turn:\nThought:"
        return prompt
        
    def _generate_with_kv_cache(self, new_prompt: str, is_first_turn: bool = False) -> str:
        """Generate response using KV cache optimization."""
        
        if self.backend == "vllm":
            return self._generate_vllm(new_prompt, is_first_turn)
        else:
            return self._generate_transformers(new_prompt, is_first_turn)
    
    def _generate_vllm(self, new_prompt: str, is_first_turn: bool) -> str:
        """Generate using vLLM (automatically handles KV caching)."""
        if is_first_turn:
            full_prompt = new_prompt
        else:
            # vLLM automatically handles continuation
            full_prompt = "".join(self.conversation_history) + new_prompt
            
        outputs = self.llm.generate([full_prompt], self.sampling_params)
        return outputs[0].outputs[0].text.strip()
    
    def _generate_transformers(self, new_prompt: str, is_first_turn: bool) -> str:
        """Generate using transformers with manual KV cache management."""
        
        if is_first_turn:
            # First turn: tokenize full prompt, no cache
            full_prompt = new_prompt
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            past_key_values = None
        else:
            # Subsequent turns: only tokenize new content, reuse cache
            new_tokens = self.tokenizer(new_prompt, return_tensors="pt").to(self.model.device)
            input_ids = new_tokens["input_ids"]
            attention_mask = torch.cat([
                torch.ones((1, len(self.tokenized_history)), device=self.model.device),
                new_tokens["attention_mask"]
            ], dim=1)
            past_key_values = self.kv_cache
        
        # Generate with KV cache
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True
            )
        
        # Extract generated text (excluding input)
        generated_ids = outputs.sequences[0][input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Update KV cache for next turn
        self.kv_cache = outputs.past_key_values
        
        # Update tokenized history
        if is_first_turn:
            self.tokenized_history = input_ids[0].tolist()
        else:
            self.tokenized_history.extend(input_ids[0].tolist())
        self.tokenized_history.extend(generated_ids.tolist())
        
        return generated_text.strip()
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse ReAct response to extract thought and action."""
        thought = ""
        action = ""
        
        # Extract thought
        thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|$)", response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # Extract action
        action_match = re.search(r"Action:\s*(\{.*?\})", response, re.DOTALL | re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()
        
        return thought, action
    
    def reset_conversation(self):
        """Reset conversation state and KV cache."""
        self.conversation_history = []
        self.tokenized_history = []
        self.kv_cache = None
        self.step_count = 0
    
    def run_episode(self, env_id: int, sample_quota: int = 50, test_quota: int = 5) -> Dict[str, Any]:
        """
        Run ReAct episode with KV cache optimization.
        
        Args:
            env_id: Physics problem ID
            sample_quota: Experiment quota
            test_quota: Hypothesis test quota
            
        Returns:
            Episode results dictionary
        """
        # Initialize environment and reset conversation
        interface = ResearchInterface(env_id, sample_quota=sample_quota, test_quota=test_quota)
        self.reset_conversation()
        
        print(f"Starting KV-cached ReAct episode on problem {env_id}")
        print("="*60)
        
        # Build system prompt (stays constant)
        system_prompt = self._build_system_prompt(interface)
        
        total_reward = 0.0
        done = False
        observation = "Starting physics discovery task."
        
        # ReAct loop with KV cache optimization
        while self.step_count < self.max_steps and not done:
            self.step_count += 1
            print(f"\n--- Step {self.step_count} ---")
            
            # Build turn-specific prompt (only new info)
            if self.step_count == 1:
                # First turn: include system prompt
                turn_prompt = system_prompt + self._add_turn_prompt(
                    observation, 
                    {'sample': interface.get_remaining_quota(), 'test': interface.test_quota}
                )
                is_first_turn = True
            else:
                # Subsequent turns: only new information
                turn_prompt = self._add_turn_prompt(
                    observation,
                    {'sample': interface.get_remaining_quota(), 'test': interface.test_quota}
                )
                is_first_turn = False
            
            # Generate with KV cache
            response = self._generate_with_kv_cache(turn_prompt, is_first_turn)
            
            # Parse response
            thought, action_str = self._parse_response(response)
            
            print(f"Thought: {thought}")
            print(f"Action: {action_str}")
            
            # Store in conversation history
            self.conversation_history.append(turn_prompt + f" {thought}\nAction: {action_str}\n")
            
            # Execute action
            if action_str:
                try:
                    observation, reward, done, info = interface.step(action_str)
                    total_reward += reward
                    
                    print(f"Observation: {observation}")
                    print(f"Reward: {reward:.3f}, Done: {done}")
                    
                    if info.get('hypothesis_correct', False):
                        print("🎉 CORRECT HYPOTHESIS FOUND!")
                        break
                        
                except Exception as e:
                    observation = f"Error: {str(e)}"
                    print(f"Observation: {observation}")
            else:
                observation = "Error: Could not parse action"
                print(f"Observation: {observation}")
            
            # Check termination
            if interface.get_remaining_quota() <= 0 and interface.test_quota <= 0:
                done = True
        
        # Results summary
        results = {
            'env_id': env_id,
            'steps_taken': self.step_count,
            'total_reward': total_reward,
            'experiments_conducted': len(interface.observations),
            'hypotheses_tested': len(interface.tested_hypothesis),
            'episode_completed': done,
            'conversation_length': len("".join(self.conversation_history)),
            'tokens_processed': len(self.tokenized_history) if hasattr(self, 'tokenized_history') else 0
        }
        
        print(f"\n{'='*60}")
        print("EPISODE SUMMARY")
        print(f"{'='*60}")
        print(f"Steps: {self.step_count}")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Conversation tokens: {results['tokens_processed']}")
        
        return results


# Example with conversation chunking for very long episodes
class ChunkedKVCacheAgent(KVCacheReActAgent):
    """Extended version that handles very long conversations by chunking."""
    
    def __init__(self, max_context_length: int = 4096, **kwargs):
        super().__init__(**kwargs)
        self.max_context_length = max_context_length
        
    def _maybe_truncate_context(self):
        """Truncate context if it gets too long, keeping recent turns."""
        if len(self.tokenized_history) > self.max_context_length:
            # Keep last 75% of context
            keep_length = int(self.max_context_length * 0.75)
            self.tokenized_history = self.tokenized_history[-keep_length:]
            self.conversation_history = self.conversation_history[-5:]  # Keep last 5 turns
            self.kv_cache = None  # Reset cache after truncation
            print(f"Context truncated to {keep_length} tokens")


# Example usage
if __name__ == "__main__":
    # Test with different backends
    
    if HAS_VLLM:
        print("Testing with vLLM backend...")
        agent = KVCacheReActAgent(
            model_name="microsoft/DialoGPT-medium",
            backend="vllm",
            max_steps=10
        )
        results = agent.run_episode(env_id=285, sample_quota=15, test_quota=3)
        
    elif HAS_TORCH:
        print("Testing with transformers backend...")
        agent = KVCacheReActAgent(
            model_name="microsoft/DialoGPT-medium", 
            backend="transformers",
            max_steps=10,
            device="cpu"
        )
        results = agent.run_episode(env_id=285, sample_quota=15, test_quota=3)
        
    else:
        print("No supported backends available. Install torch or vllm.")