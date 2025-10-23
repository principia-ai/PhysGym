"""
LLM Provider Abstractions for Local and Remote LLMs

Unified implementation supporting:
- Ollama (local, user-friendly)
- vLLM (local, high-performance)
- Remote APIs (OpenRouter, OpenAI, Anthropic, DeepSeek)
"""

import os
import json
import time
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str                           # Provider name (ollama, vllm, openrouter, etc.)
    model: str                             # Model identifier
    api_key: Optional[str] = None          # API key (if required)
    base_url: Optional[str] = None         # Base URL for API
    temperature: float = 0.3               # Temperature for generation
    max_tokens: int = 4000                 # Maximum tokens to generate
    timeout: int = 60                      # Request timeout in seconds
    extra_params: Optional[Dict] = None    # Additional provider-specific parameters


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider_name = config.provider
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        system_message: str = "You are a helpful assistant.",
        capture_thinking: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """Generate text using the LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and ready to use."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "provider": self.provider_name,
            "model": self.config.model,
            "base_url": self.config.base_url
        }


class OllamaProvider(BaseLLMProvider):
    """Provider for Ollama local LLM server."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
        
    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """Get list of available models in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except:
            return []
    
    def generate(
        self, 
        prompt: str, 
        system_message: str = "You are a helpful assistant.",
        capture_thinking: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """Generate text using Ollama API."""
        url = f"{self.base_url}/api/generate"
        # url = self.base_url
        
        # Construct the prompt with system message
        full_prompt = f"{system_message}\n\nUser: {prompt}\n\nAssistant:"
        
        data = {
            "model": self.config.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            }
        }
        
        # Add extra parameters if provided
        if self.config.extra_params:
            data["options"].update(self.config.extra_params)
        
        try:
            response = requests.post(
                url, 
                json=data, 
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get("response", "")
            
            if capture_thinking:
                thinking_data = {
                    "raw_thinking": "",  # Ollama doesn't provide thinking by default
                    "api_response": result,
                    "model": self.config.model,
                    "timestamp": time.time(),
                    "provider": "ollama"
                }
                return content, thinking_data
            
            return content, None
            
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")


class vLLMProvider(BaseLLMProvider):
    """Provider for vLLM OpenAI-compatible server."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:8000"
        
    def is_available(self) -> bool:
        """Check if vLLM server is running."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """Get list of available models in vLLM."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        except:
            return []
    
    def generate(
        self, 
        prompt: str, 
        system_message: str = "You are a helpful assistant.",
        capture_thinking: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """Generate text using vLLM OpenAI-compatible API."""
        url = f"{self.base_url}/v1/chat/completions"
        
        headers = {"Content-Type": "application/json"}
        
        data = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": False
        }
        
        # Add extra parameters if provided
        if self.config.extra_params:
            data.update(self.config.extra_params)
        
        try:
            response = requests.post(
                url, 
                headers=headers,
                json=data, 
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            if capture_thinking:
                thinking_data = {
                    "raw_thinking": "",  # vLLM doesn't provide thinking by default
                    "api_response": result,
                    "model": self.config.model,
                    "timestamp": result.get("created", time.time()),
                    "provider": "vllm"
                }
                return content, thinking_data
            
            return content, None
            
        except Exception as e:
            raise RuntimeError(f"vLLM generation failed: {e}")


class RemoteAPIProvider(BaseLLMProvider):
    """Provider for remote API services (OpenRouter, OpenAI, Anthropic, DeepSeek)."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.max_retries = 3
        self.retry_delay = 2.0
        
    def is_available(self) -> bool:
        """Check if API key is configured."""
        return self.config.api_key is not None and len(self.config.api_key) > 0
    
    def generate(
        self, 
        prompt: str, 
        system_message: str = "You are a helpful assistant.",
        capture_thinking: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """Generate text using remote API."""
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        model = self.config.model
        api_key = self.config.api_key
        provider = self.config.provider.lower()
        
        # Handle Anthropic (Claude) models
        if model == 'claude-3-7-sonnet-20250219' or provider == 'anthropic':
            return self._generate_anthropic(prompt, system_message, temperature, 
                                           max_tokens, model, api_key, capture_thinking)
        
        # Handle DeepSeek models
        if provider == 'deepseek' or model == "deepseek-reasoner":
            return self._generate_deepseek(prompt, system_message, temperature,
                                          model, api_key, capture_thinking)
        
        # Handle DeepSeek R1 via Volcano Engine
        if model == "deepseek-r1-250120":
            return self._generate_deepseek_r1_volcano(prompt, system_message, temperature,
                                                      model, api_key, capture_thinking)
        
        # Default: OpenRouter or OpenAI-compatible API
        return self._generate_openrouter(prompt, system_message, temperature,
                                        max_tokens, model, api_key, capture_thinking)
    
    def _generate_anthropic(self, prompt, system_message, temperature, max_tokens, 
                           model, api_key, capture_thinking):
        """Generate using Anthropic API."""
        client = Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens or 5000,
            temperature=temperature,
            system=system_message,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        
        content = response.content[0].text
        retry_count = 0
        
        # Retry mechanism for empty content
        while (not content or content == "\n" or "error" in content.lower()) and retry_count < self.max_retries:
            retry_count += 1
            print(f"Received empty or error content. Retrying ({retry_count}/{self.max_retries})...")
            time.sleep(self.retry_delay)
            
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens or 5000,
                temperature=temperature,
                system=system_message,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            content = response.content[0].text
            
            if content and "error" not in content.lower():
                break
        
        if capture_thinking:
            return content, {
                "raw_thinking": "",
                "api_response": content,
                "model": model,
                "timestamp": time.time(),
                "provider": "anthropic"
            }
        
        return content, None
    
    def _generate_deepseek(self, prompt, system_message, temperature, model, api_key, capture_thinking):
        """Generate using DeepSeek API."""
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        
        reasoning = response.choices[0].message.reasoning_content
        answer = response.choices[0].message.content
        
        if capture_thinking:
            return answer, {
                "raw_thinking": reasoning,
                "api_response": answer,
                "model": model,
                "timestamp": response.created,
                "provider": "deepseek"
            }
        
        return answer, None
    
    def _generate_deepseek_r1_volcano(self, prompt, system_message, temperature, model, api_key, capture_thinking):
        """Generate using DeepSeek R1 via Volcano Engine."""
        client = OpenAI(
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        
        reasoning = response.choices[0].message.reasoning_content
        answer = response.choices[0].message.content
        
        if capture_thinking:
            return answer, {
                "raw_thinking": reasoning,
                "api_response": answer,
                "model": model,
                "timestamp": response.created,
                "provider": "deepseek_volcano"
            }
        
        return answer, None
    
    def _generate_openrouter(self, prompt, system_message, temperature, max_tokens, 
                            model, api_key, capture_thinking):
        """Generate using OpenRouter or OpenAI-compatible API."""
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "reasoning": {
                "effort": "high",
                "exclude": False,
                "enabled": True
            },
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Configure thinking mode for specific models
        if capture_thinking:
            if "claude" in model.lower():
                data["claude_thinking"] = True
            elif "mistral" in model.lower() and "instruct" in model.lower():
                data["extra_tools"] = {"thinking": True}
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            print(f"Error response: {err.response.text}")
            raise
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        retry_count = 0
        
        # Retry mechanism
        while (not content or content == "\n" or "error" in content.lower()) and retry_count < self.max_retries:
            retry_count += 1
            print(f"Received empty or error content. Retrying ({retry_count}/{self.max_retries})...")
            time.sleep(self.retry_delay)
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            if content and "error" not in content.lower():
                break
        
        if not content:
            content = "Failed to generate content after multiple attempts."
        
        # Extract thinking if requested
        if capture_thinking:
            thinking_data = {}
            
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice:
                    message = choice["message"]
                    
                    # DeepSeek format (via tools)
                    if "tool_calls" in message:
                        for tool_call in message.get("tool_calls", []):
                            if tool_call.get("type") == "thinking":
                                thinking_data["raw_thinking"] = tool_call.get("thinking", {}).get("content", "")
                    
                    # Extract model-specific thinking
                    if "thinking" in message:
                        thinking_data["raw_thinking"] = message["thinking"]
                    
                    # Anthropic format
                    if "claude_thinking" in message:
                        thinking_data["raw_thinking"] = message["claude_thinking"]
            
            thinking_data["api_response"] = result
            thinking_data["timestamp"] = result.get("created")
            thinking_data["model"] = result.get("model", model)
            thinking_data["provider"] = "openrouter"
            
            return content, thinking_data
        
        return content, None


class LLMProviderFactory:
    """Factory class for creating LLM providers."""
    
    _providers = {
        # Local providers
        "ollama": OllamaProvider,
        "vllm": vLLMProvider,
        # Remote providers
        "openrouter": RemoteAPIProvider,
        "openai": RemoteAPIProvider,
        "anthropic": RemoteAPIProvider,
        "deepseek": RemoteAPIProvider,
        "volcano": RemoteAPIProvider,
    }
    
    @classmethod
    def create_provider(cls, config: LLMConfig) -> BaseLLMProvider:
        """Create a provider instance based on the configuration."""
        provider_class = cls._providers.get(config.provider.lower())
        if not provider_class:
            raise ValueError(f"Unsupported provider: {config.provider}. Supported: {list(cls._providers.keys())}")
        
        return provider_class(config)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of all supported provider names."""
        return list(cls._providers.keys())
    
    @classmethod
    def detect_local_providers(cls) -> Dict[str, Dict[str, Any]]:
        """Detect which local providers are currently available and their details."""
        local_providers = ["ollama", "vllm"]
        available = {}
        
        for provider_name in local_providers:
            try:
                # Create a minimal config to test availability
                config = LLMConfig(provider=provider_name, model="test")
                provider = cls.create_provider(config)
                if provider.is_available():
                    info = {"status": "available", "base_url": provider.config.base_url}
                    
                    # Get available models if provider supports it
                    if hasattr(provider, 'list_models'):
                        try:
                            models = provider.list_models()
                            info["models"] = models[:5]  # Show first 5 models
                            info["total_models"] = len(models)
                        except:
                            info["models"] = []
                    
                    available[provider_name] = info
            except Exception as e:
                available[provider_name] = {"status": "error", "error": str(e)}
                
        return available


def load_llm_config_from_env(
    provider: str = None, 
    model: str = None,
    env_file: str = "api_keys.env"
) -> LLMConfig:
    """
    Load LLM configuration from environment variables.
    
    Args:
        provider: Override provider from environment
        model: Override model from environment  
        env_file: Path to environment file
        
    Returns:
        LLMConfig instance
    """
    # Load environment variables
    if os.path.exists(env_file):
        load_dotenv(env_file, override=True)
    
    # Get configuration from environment or use defaults
    provider = provider or os.getenv("LLM_PROVIDER", "openrouter")
    model = model or os.getenv("LLM_MODEL", "google/gemini-2.5-flash-preview")
    
    # Create base configuration
    config = LLMConfig(
        provider=provider,
        model=model,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4000")),
        timeout=int(os.getenv("LLM_TIMEOUT", "60"))
    )
    
    # Set API key and base URL based on provider
    provider_lower = provider.lower()
    
    if provider_lower == "ollama":
        config.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        config.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
    elif provider_lower == "vllm":
        config.base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
        config.model = model or os.getenv("VLLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
    elif provider_lower == "openrouter":
        config.api_key = os.getenv("OPENROUTER_API_KEY")
    elif provider_lower == "openai":
        config.api_key = os.getenv("OPENAI_API_KEY")
    elif provider_lower == "anthropic":
        config.api_key = os.getenv("ANTHROPIC_API_KEY")
    elif provider_lower == "deepseek":
        config.api_key = os.getenv("DEEPSEEK_API_KEY")
    
    return config


def generate_with_provider(
    prompt: str,
    provider: str = None,
    model: str = None,
    system_message: str = "You are a helpful assistant.",
    temperature: float = 0.3,
    max_tokens: int = 4000,
    capture_thinking: bool = False,
    api_key: str = None,
    base_url: str = None,
    env_file: str = "api_keys.env",
    **kwargs
) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """
    Generate text using any supported LLM provider.
    
    This is a high-level convenience function that automatically:
    1. Loads configuration from environment variables
    2. Creates the appropriate provider
    3. Generates text using the provider
    
    Args:
        prompt: The prompt to send to the LLM
        provider: Provider name (auto-detected if None)
        model: Model name (from env if None)
        system_message: System message
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        capture_thinking: Whether to capture thinking/reasoning
        api_key: Override API key
        base_url: Override base URL for local providers
        env_file: Environment file path
        **kwargs: Additional parameters
        
    Returns:
        Generated text, optionally with thinking data
    """
    # Load base configuration
    config = load_llm_config_from_env(provider, model, env_file)
    
    # Apply overrides
    if api_key:
        config.api_key = api_key
    if base_url:
        config.base_url = base_url
    if temperature != 0.3:
        config.temperature = temperature
    if max_tokens != 4000:
        config.max_tokens = max_tokens
    
    # Create provider and generate
    provider_instance = LLMProviderFactory.create_provider(config)
    
    return provider_instance.generate(
        prompt=prompt,
        system_message=system_message,
        capture_thinking=capture_thinking,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )


def get_recommended_provider() -> str:
    """
    Automatically detect and recommend the best available LLM provider.
    
    Priority order:
    1. Local providers (if available)
    2. Remote providers (if API keys available)
    
    Returns:
        Recommended provider name
    """
    # Check local providers first
    available_local = LLMProviderFactory.detect_local_providers()
    
    # Ollama is preferred for local due to ease of use
    if "ollama" in available_local and available_local["ollama"]["status"] == "available":
        return "ollama"
    
    # vLLM is good for performance
    if "vllm" in available_local and available_local["vllm"]["status"] == "available":
        return "vllm"
    
    # Fall back to remote providers
    # Check for API keys
    load_dotenv("api_keys.env", override=True)
    
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    elif os.getenv("OPENAI_API_KEY"):
        return "openai"
    elif os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    elif os.getenv("DEEPSEEK_API_KEY"):
        return "deepseek"
    
    # Default fallback
    return "openrouter"


def show_provider_status():
    """Print a status report of all available LLM providers."""
    print("LLM Provider Status Report")
    print("=" * 50)
    
    # Check local providers
    print("\n🖥️  Local Providers:")
    local_providers = LLMProviderFactory.detect_local_providers()
    
    for provider, info in local_providers.items():
        status = info.get("status", "unknown")
        if status == "available":
            print(f"  ✅ {provider.upper()}: Available")
            print(f"     URL: {info.get('base_url', 'N/A')}")
            if "models" in info:
                models_info = f"{info.get('total_models', 0)} models"
                if info["models"]:
                    models_info += f" (e.g., {', '.join(info['models'][:3])})"
                print(f"     Models: {models_info}")
        else:
            print(f"  ❌ {provider.upper()}: Not available")
            if "error" in info:
                print(f"     Error: {info['error']}")
    
    # Check remote providers
    print("\n🌐 Remote Providers:")
    load_dotenv("api_keys.env", override=True)
    
    remote_providers = {
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY", 
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY"
    }
    
    for provider, key_name in remote_providers.items():
        api_key = os.getenv(key_name)
        if api_key:
            print(f"  ✅ {provider.upper()}: API key configured")
        else:
            print(f"  ❌ {provider.upper()}: No API key found ({key_name})")
    
    # Show recommendation
    recommended = get_recommended_provider()
    print(f"\n💡 Recommended provider: {recommended.upper()}")
    print("=" * 50)


def load_api_key(env_file="api_keys.env", provider="openrouter"):
    """
    Load API key from a .env file or environment variables.

    Args:
        env_file (str): Path to the environment file.
        provider (str): API provider to use ('openrouter', 'openai', 'deepseek', 'anthropic', 'volcano').

    Returns:
        str: The API key or None if not found.
    """
    provider_lower = provider.lower()
    if provider_lower == "openrouter":
        key_name = "OPENROUTER_API_KEY"
    elif provider_lower == "openai":
        key_name = "OPENAI_API_KEY"
    elif provider_lower == "deepseek":
        key_name = "DEEPSEEK"
    elif provider_lower == "volcano":
        key_name = "VOLCANO"
    elif provider_lower == "anthropic":
        key_name = "ANTHROPIC_API_KEY"
    else:
        print(f"Unsupported provider: {provider}")
        return None

    try:
        # Try to load from environment variables first
        api_key = os.getenv(key_name)
        if api_key:
            return api_key

        # Try to load from the specified .env file
        if os.path.exists(env_file):
            load_dotenv(env_file, override=True)
            api_key = os.getenv(key_name)
            if api_key:
                return api_key
        
        print(f"Could not find {key_name} in environment variables or .env files (including {env_file}).")
        return None
    except Exception as e:
        print(f"Error loading API key for {provider}: {e}")
        return None


def generate_with_api(
    prompt: str, 
    api_key: str, 
    system_message: str = "You are a helpful assistant.", 
    temperature: float = 0.3, 
    model: str = "google/gemini-2.5-flash",
    max_tokens: int = 50000,
    capture_thinking: bool = False,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    provider: str = "openrouter"
) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """
    Generate content using any supported API provider.
    
    This is a backward-compatible wrapper for the old generate_with_api function.
    It now uses the unified provider system under the hood.
    
    Args:
        prompt (str): The prompt to send to the API
        api_key (str): API key for the provider
        system_message (str): System message to set the assistant's behavior
        temperature (float): Controls randomness (0-1)
        model (str): Model ID to use
        max_tokens (int): Maximum number of tokens to generate
        capture_thinking (bool): Whether to capture thinking output in models that support it
        max_retries (int): Maximum number of retries for failed requests (deprecated, handled by provider)
        retry_delay (float): Delay between retries (deprecated, handled by provider)
        provider (str): Provider name ('openrouter', 'openai', 'anthropic', 'deepseek', 'volcano')
        
    Returns:
        If capture_thinking is False: str - The generated content
        If capture_thinking is True: Tuple[str, Dict] - The generated content and thinking metadata
    """
    # Create configuration for the provider
    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Create provider instance
    provider_instance = LLMProviderFactory.create_provider(config)
    
    # Generate response
    return provider_instance.generate(
        prompt=prompt,
        system_message=system_message,
        capture_thinking=capture_thinking,
        temperature=temperature,
        max_tokens=max_tokens
    )


if __name__ == "__main__":
    # Demo usage
    show_provider_status()