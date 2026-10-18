#!/usr/bin/env python3
"""
Test script to verify the model change works
"""

import sys
sys.path.insert(0, '/mnt/c/Users/jaafo/3a/arxiv_research_companion')

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os

# Load environment variables
load_dotenv()

def test_model_change():
    """Test that the model change works"""
    print("Testing model change to openai/gpt-oss-20b...")

    # Get configuration from environment
    model_name = os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-120b")
    api_token = os.getenv("HF_API_TOKEN", "")

    print(f"Model: {model_name}")
    print(f"Token set: {bool(api_token)}")

    if not api_token:
        print("ERROR: HF_API_TOKEN not set in environment")
        return False

    try:
        # Initialize client
        client_options = {'model': model_name, 'token': api_token}
        client = InferenceClient(**client_options)

        # Test with a simple prompt
        test_prompt = "What is the capital of France?"
        print(f"Sending test prompt: {test_prompt}")

        response = client.chat_completion(
            messages=[{'role': 'user', 'content': test_prompt}],
            max_tokens=50,
            temperature=0.7
        )

        print(f"Raw response: {response}")

        # Extract content
        choices = response.get('choices') if isinstance(response, dict) else getattr(response, 'choices', None)
        if choices:
            message = choices[0].get('message') if isinstance(choices[0], dict) else getattr(choices[0], 'message', None)
            content = message.get('content') if isinstance(message, dict) else getattr(message, 'content', None)
            print(f"Response content: {content}")
            if content and isinstance(content, str) and content.strip():
                print("✅ Model change test PASSED - got non-empty response")
                return True
            else:
                print("❌ Model change test FAILED - empty response")
                return False
        else:
            print("❌ No choices in response")
            return False

    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_change()
    if not success:
        sys.exit(1)