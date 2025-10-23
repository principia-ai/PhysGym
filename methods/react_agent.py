"""
ReAct Agent for Physics Discovery using ResearchInterface

This module implements a ReAct (Reasoning and Acting) agent that can interact with
physics environments through the ResearchInterface step method. The agent follows
the ReAct pattern: Thought -> Action -> Observation cycles.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from physgym.interface import ResearchInterface
from physgym.utils.llm_providers import generate_with_provider, get_recommended_provider, load_api_key


class ReActPhysicsAgent:
    """
    ReAct agent for physics discovery using ResearchInterface.
    
    The agent follows the ReAct pattern:
    1. Thought: Reason about the current situation
    2. Action: Take an action in the environment  
    3. Observation: Process the environment response
    4. Repeat until goal is achieved or resources exhausted
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "google/gemini-2.5-flash",
                 env_file: str = "api_keys.env",
                 provider: Optional[str] = None,
                 max_steps: int = 20,
                 temperature: float = 0.7):
        """
        Initialize the ReAct Physics Agent.
        
        Args:
            api_key: API key for LLM provider (None for local providers)
            model: Model name to use
            env_file: Path to environment file with API key
            provider: LLM provider (ollama, vllm, openrouter, etc.). Auto-detected if None.
            max_steps: Maximum number of ReAct steps
            temperature: LLM sampling temperature
        """
        self.model = model
        self.max_steps = max_steps
        self.temperature = temperature
        self.env_file = env_file
        
        # Initialize LLM provider
        self.provider = provider or get_recommended_provider()
        self.api_key = api_key
        print(f"Using LLM provider: {self.provider} with model: {self.model}")
        
        # ReAct state
        self.step_count = 0
        self.thought_history = []
        self.action_history = []
        self.observation_history = []
        self.current_hypothesis = ""
        
    def _generate_text(self, prompt: str) -> str:
        """Generate text using the configured LLM provider."""
        try:
            response, _ = generate_with_provider(
                prompt=prompt,
                provider=self.provider,
                model=self.model,
                api_key=self.api_key,
                env_file=self.env_file,
                temperature=self.temperature
            )
            return response
        except Exception as e:
            print(f"Error generating text: {e}")
            return "Error: Could not generate response from LLM"
    
    def _build_react_prompt(self, interface: ResearchInterface, observation: str = "") -> str:
        """Build the ReAct prompt with current context."""
        
        # Get environment information
        env_info = f"""
Environment Information:
- Problem ID: {interface.env.id}
- Available parameters: {', '.join(interface.all_params)}
- Sample quota: {interface.get_remaining_quota()}/{interface.sample_quota}
- Test quota: {interface.test_quota}
- Problem description: {interface.problem_content[:200]}...
- True equation (hidden): {interface.equation}
"""
        
        # Build history context
        history_context = ""
        if self.thought_history or self.action_history or self.observation_history:
            history_context = "\nPrevious Steps:\n"
            max_history = min(5, len(self.thought_history))  # Show last 5 steps
            for i in range(max_history):
                if i < len(self.thought_history):
                    history_context += f"Thought {len(self.thought_history) - max_history + i + 1}: {self.thought_history[-(max_history-i)]}\n"
                if i < len(self.action_history):
                    history_context += f"Action {len(self.action_history) - max_history + i + 1}: {self.action_history[-(max_history-i)]}\n"
                if i < len(self.observation_history):
                    history_context += f"Observation {len(self.observation_history) - max_history + i + 1}: {self.observation_history[-(max_history-i)]}\n"
                history_context += "\n"
        
        # Current observation
        current_obs = f"Current Observation: {observation}\n" if observation else ""
        
        prompt = f"""You are a physics researcher trying to discover the underlying equation of a physical system through experiments and hypothesis testing.

{env_info}

{history_context}

{current_obs}

Available Actions:
1. run_experiment: Test parameter values to collect data
   Format: {{'type': 'run_experiment', 'params': {{'param1': value1, 'param2': value2}}}}

2. test_hypothesis: Test a mathematical hypothesis
   Format: {{'type': 'test_hypothesis', 'hypothesis': 'mathematical_expression'}}

3. get_status: Check current experiment status
   Format: {{'type': 'get_status'}}

Instructions:
- Follow the ReAct pattern: Think step by step, then take an action
- Start your response with "Thought:" followed by your reasoning
- Then provide "Action:" with a JSON-formatted action
- Use experiments to gather data and understand parameter relationships
- Propose hypotheses based on observed patterns
- You have limited sample and test quotas, so be strategic
- Try to find the true underlying equation

Your response format:
Thought: [Your reasoning about what to do next, be concise]
Action: [JSON action to take]
"""
        
        return prompt
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse LLM response to extract thought and action."""
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
        else:
            # Try to find any JSON-like structure
            json_match = re.search(r"(\{[^}]*\})", response)
            if json_match:
                action = json_match.group(1).strip()
        
        return thought, action
    
    def run_episode(self, env_id: int, sample_quota: int = 50, test_quota: int = 5) -> Dict[str, Any]:
        """
        Run a complete ReAct episode on a physics problem.
        
        Args:
            env_id: Physics problem ID
            sample_quota: Maximum experiments allowed
            test_quota: Maximum hypothesis tests allowed
            
        Returns:
            Dictionary with episode results
        """
        # Initialize environment
        interface = ResearchInterface(env_id, sample_quota=sample_quota, test_quota=test_quota)
        
        # Reset agent state
        self.step_count = 0
        self.thought_history = []
        self.action_history = []
        self.observation_history = []
        self.current_hypothesis = ""
        
        print(f"Starting ReAct episode on problem {env_id}")
        print(f"Sample quota: {sample_quota}, Test quota: {test_quota}")
        print("="*60)
        
        # Initial observation
        observation = f"Starting physics discovery task. Problem: {interface.problem_content[:100]}..."
        
        total_reward = 0.0
        done = False
        
        # ReAct loop
        while self.step_count < self.max_steps and not done:
            self.step_count += 1
            print(f"\n--- Step {self.step_count} ---")
            
            # Generate thought and action
            prompt = self._build_react_prompt(interface, observation)
            response = self._generate_text(prompt)
            
            thought, action_str = self._parse_response(response)
            
            print(f"Thought: {thought}")
            print(f"Action: {action_str}")
            
            # Store thought and action
            self.thought_history.append(thought)
            self.action_history.append(action_str)
            
            # Execute action
            if action_str:
                try:
                    observation, reward, done, info = interface.step(action_str)
                    total_reward += reward
                    
                    print(f"Observation: {observation}")
                    print(f"Reward: {reward:.3f}, Done: {done}")
                    
                    # Store observation
                    self.observation_history.append(observation)
                    
                    # Check if hypothesis was tested and correct
                    if info.get('hypothesis_correct', False):
                        print("🎉 CORRECT HYPOTHESIS FOUND!")
                        break
                        
                except Exception as e:
                    observation = f"Error executing action: {str(e)}"
                    print(f"Observation: {observation}")
                    self.observation_history.append(observation)
            else:
                observation = "Error: Could not parse action from LLM response"
                print(f"Observation: {observation}")
                self.observation_history.append(observation)
            
            # Check termination conditions
            if interface.get_remaining_quota() <= 0 and interface.test_quota <= 0:
                print("All quotas exhausted - episode ended")
                done = True
        
        # Generate final summary
        print("\n" + "="*60)
        print("EPISODE SUMMARY")
        print("="*60)
        
        results = {
            'env_id': env_id,
            'steps_taken': self.step_count,
            'total_reward': total_reward,
            'experiments_conducted': len(interface.observations),
            'hypotheses_tested': len(interface.tested_hypothesis),
            'episode_completed': done,
            'remaining_sample_quota': interface.get_remaining_quota(),
            'remaining_test_quota': interface.test_quota,
            'thought_history': self.thought_history.copy(),
            'action_history': self.action_history.copy(),
            'observation_history': self.observation_history.copy()
        }
        
        # Check if correct hypothesis was found
        if interface.tested_hypothesis:
            best_hypothesis = max(
                interface.tested_hypothesis,
                key=lambda h: h["evaluation"].get("overall_score", 0)
            )
            results['best_hypothesis'] = {
                'hypothesis': best_hypothesis.get('function_code', ''),
                'score': best_hypothesis["evaluation"].get("overall_score", 0),
                'is_correct': best_hypothesis["evaluation"].get("is_correct", False)
            }
            
            if best_hypothesis["evaluation"].get("is_correct", False):
                print("✅ SUCCESS: Found the correct hypothesis!")
            else:
                print(f"❌ Best hypothesis score: {best_hypothesis['evaluation'].get('overall_score', 0):.4f}")
        else:
            print("❌ No hypotheses were tested")
            results['best_hypothesis'] = None
        
        print(f"Total steps: {self.step_count}")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Experiments: {len(interface.observations)}")
        print(f"Hypotheses tested: {len(interface.tested_hypothesis)}")
        
        return results
    
    def run_batch(self, env_ids: List[int], sample_quota: int = 50, test_quota: int = 5) -> List[Dict[str, Any]]:
        """
        Run ReAct agent on multiple physics problems.
        
        Args:
            env_ids: List of physics problem IDs
            sample_quota: Sample quota per problem
            test_quota: Test quota per problem
            
        Returns:
            List of episode results
        """
        all_results = []
        
        for env_id in env_ids:
            print(f"\n{'='*80}")
            print(f"STARTING PROBLEM {env_id}")
            print(f"{'='*80}")
            
            try:
                results = self.run_episode(env_id, sample_quota, test_quota)
                all_results.append(results)
            except Exception as e:
                print(f"Error running episode on problem {env_id}: {e}")
                all_results.append({
                    'env_id': env_id,
                    'error': str(e),
                    'episode_completed': False
                })
        
        # Print batch summary
        print(f"\n{'='*80}")
        print("BATCH SUMMARY")
        print(f"{'='*80}")
        
        success_count = sum(1 for r in all_results 
                          if r.get('best_hypothesis', {}).get('is_correct', False))
        print(f"Problems solved: {success_count}/{len(env_ids)}")
        
        if success_count > 0:
            avg_steps = sum(r.get('steps_taken', 0) for r in all_results 
                           if r.get('best_hypothesis', {}).get('is_correct', False)) / success_count
            print(f"Average steps for successful episodes: {avg_steps:.1f}")
        
        return all_results


# Example usage and testing
if __name__ == "__main__":
    # Create ReAct agent
    agent = ReActPhysicsAgent(
        model="google/gemini-2.5-flash",
        max_steps=15,
        temperature=0.7
    )
    
    # Test on a single problem
    results = agent.run_episode(env_id=285, sample_quota=20, test_quota=3)
    
    # Save results
    with open("react_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to react_results.json")