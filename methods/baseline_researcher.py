import json
import os
from typing import Dict, List, Any, Union, Optional, Tuple
from physgym.utils.llm_providers import generate_with_provider, get_recommended_provider, load_api_key


class BaselineResearcher:
    """
    Baseline implementation of an LLM-based physics researcher.
    
    This researcher uses a prompt-based approach to analyze experimental data,
    propose hypotheses, and design new experiments using the OpenRouter API.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "google/gemini-2.5-flash-preview", 
                 env_file: str = "api_keys.env",
                 provider: Optional[str] = None):
        """
        Initialize the BaselineResearcher.
        
        Args:
            api_key: API key. If None, it will be loaded from env_file for remote providers.
            model: Model name to use.
            env_file: Path to the environment file containing the API key.
            provider: LLM provider to use (ollama, vllm, openrouter, etc.). If None, auto-detected.
        """
        self.env_file = env_file
        self.provider = provider or get_recommended_provider()
        self.model = model
        self.api_key = api_key  # May be None for local providers
        print(f"Using LLM provider: {self.provider} with model: {self.model}")
        
        # Load the prompt template - find it relative to this file
        from pathlib import Path
        methods_dir = Path(__file__).parent
        
        # Try prompts in methods directory first
        prompt_path = methods_dir / "prompts" / "baseline_researcher.txt"
        if not prompt_path.exists():
            # Fallback to project root prompts for backwards compatibility
            project_root = methods_dir.parent
            prompt_path = project_root / "prompts" / "baseline_researcher.txt"
        
        try:
            with open(prompt_path, "r") as f:
                self.prompt_template = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt template not found at {prompt_path}")
        
        # Initialize state
        self.current_hypothesis = ""
        self.confidence = 0.0
    
    def analyze_and_propose(
        self,
        problem_description: str,
        controllable_variables: Dict[str, str],
        observable_variable: Dict[str, str],
        historical_experiments: List[Dict[str, Any]],
        quota: Dict[str, int] = None,
        capture_thinking: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze experimental data and propose next experiments.
        
        Args:
            problem_description: A brief description of the physical phenomenon.
            controllable_variables: List of dictionaries containing controllable variable names and descriptions.
            observable_variable: Dictionary containing the observable variable name and description.
            historical_experiments: List of dictionaries containing previous experimental results.
            quota: Dictionary with experiment quotas.
            capture_thinking: Whether to capture and return model thinking/reasoning.
            
        Returns:
            Dictionary containing next experiments to run, flag to test the hypothesis,
            and the current hypothesis formula. If capture_thinking is True, also includes
            model thinking data under the 'thinking' key.
        """
        # Prepare the input for the LLM
        llm_input = {
            "problem_description": problem_description,
            "controllable_variables": controllable_variables,
            "observable_variable": observable_variable,
            "historical_experiments": historical_experiments,
            "quota": quota
        }
        
        # Convert to JSON string for the prompt
        input_json = json.dumps(llm_input, indent=2)
        
        # Create the full prompt for the LLM
        prompt = f"{self.prompt_template}\n\n**Input:**\n```json\n{input_json}\n```\n\nProvide the **Output:**\n"
        
        # Generate response using the LLM
        system_message = "You are a top-tier AI Physicist and Experimental Design Researcher. Your mission is to analyze experimental data, propose hypotheses, and design new experiments to discover and validate the mathematical relationships between physical quantities."
        
        # Call LLM using the unified provider system
        response, thinking_data = generate_with_provider(
            prompt=prompt,
            provider=self.provider,
            model=self.model,
            system_message=system_message,
            temperature=0.2,
            api_key=self.api_key,
            env_file=self.env_file,
            capture_thinking=capture_thinking
        )
        
        # Extract JSON output from the response
        try:
            # Find JSON in the response (might be surrounded by text)
            if "```json" in response:
                json_start = response.rfind("```json")
                json_end = response.find("```", json_start + 6)
                if json_end > 0 and json_end > json_start:
                    response = response[json_start + 7:json_end]
            output_start = response.find("{")
            output_end = response.rfind("}") + 1
            if output_start >= 0 and output_end > output_start:
                json_str = response[output_start:output_end]
                output = json.loads(json_str)
            else:
                # If no JSON object found, try to parse as JSON directly
                output = json.loads(response)
        except json.JSONDecodeError:
            print("Warning: Could not parse LLM response as JSON. Using fallback parsing.")
            # Fallback to extracting key fields using string matching
            output = self._extract_output_from_text(response)
        
        # Update the internal state
        self.current_hypothesis = output.get("current_hypothesis_formula", "")
        self.confidence = 1.0 if output.get("test_hypothesis_flag", False) else 0.5
        
        # Include thinking data if requested
        if capture_thinking:
            output["thinking"] = thinking_data
            # Save model's raw response for debugging
            output["raw_response"] = response[:10000] + ("..." if len(response) > 10000 else "")
        
        # Include the input data in the output for reference
        output['input'] = llm_input
        
        return output
    
    def _extract_output_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract structured output from the LLM's text response when JSON parsing fails.
        
        Args:
            text: The text response from the LLM.
            
        Returns:
            Dictionary with extracted information.
        """
        output = {
            "next_experiments": [],
            "test_hypothesis_flag": False,
            "current_hypothesis_formula": ""
        }
        
        # Extract next experiments
        if "next_experiments" in text.lower():
            try:
                experiments_section = text.split("next_experiments")[1].split("test_hypothesis_flag")[0]
                # This is a very simplified parser - in practice you would need more robust extraction
                experiments_start = experiments_section.find("[")
                experiments_end = experiments_section.find("]") + 1
                if experiments_start >= 0 and experiments_end > experiments_start:
                    experiments_json = experiments_section[experiments_start:experiments_end]
                    output["next_experiments"] = json.loads(experiments_json)
            except (IndexError, json.JSONDecodeError):
                # Fallback to empty list if parsing fails
                pass
        
        # Extract test hypothesis flag
        if "test_hypothesis_flag" in text.lower():
            output["test_hypothesis_flag"] = "true" in text.lower().split("test_hypothesis_flag")[1].split("\n")[0]
        
        # Extract current hypothesis formula
        if "current_hypothesis_formula" in text.lower():
            formula_section = text.split("current_hypothesis_formula")[1].split("\n")[0]
            formula = formula_section.split(":")[1].strip().strip('"').strip("'").strip(",")
            output["current_hypothesis_formula"] = formula
        
        return output
    
    def get_current_hypothesis(self) -> str:
        """
        Get the current hypothesis formula.
        
        Returns:
            The current hypothesis formula as a string.
        """
        return self.current_hypothesis
    
    def get_confidence(self) -> float:
        """
        Get the confidence level in the current hypothesis.
        
        Returns:
            Confidence level as a float between 0 and 1.
        """
        return self.confidence


# Create a simpler function-based interface
def run_baseline_researcher(
    problem_description: str,
    controllable_variables: Dict[str, str],
    observable_variable: Dict[str, str],
    historical_experiments: List[Dict[str, Any]],
    quota: Dict[str, int] = None,
    api_key: Optional[str] = None,
    env_file: str = "api_keys.env",
    model: str = "google/gemini-2.5-flash-preview",
    provider: Optional[str] = None,
    capture_thinking: bool = False
) -> Dict[str, Any]:
    """
    Run the baseline researcher to analyze experimental data and propose next experiments.
    
    Args:
        problem_description: A brief description of the physical phenomenon.
        controllable_variables: List of dictionaries containing controllable variable names and descriptions.
        observable_variable: Dictionary containing the observable variable name and description.
        historical_experiments: List of dictionaries containing previous experimental results.
        quota: Dictionary with experiment quotas.
        api_key: API key. If None, it will be loaded from env_file for remote providers.
        env_file: Path to the environment file containing the API key.
        model: The LLM model to use.
        provider: LLM provider (ollama, vllm, openrouter, etc.). Auto-detected if None.
        capture_thinking: Whether to capture and return model thinking.
        
    Returns:
        Dictionary containing next experiments to run, flag to test the hypothesis,
        and the current hypothesis formula. If capture_thinking is True, also includes
        model thinking data.
    """
    researcher = BaselineResearcher(
        api_key=api_key, 
        model=model, 
        env_file=env_file,
        provider=provider
    )
    return researcher.analyze_and_propose(
        problem_description,
        controllable_variables,
        observable_variable,
        historical_experiments,
        quota=quota,
        capture_thinking=capture_thinking
    )