import openai

class MultiPromptGenerator:
    def __init__(self, api_key: str, api_base: str = "https://openai.vocareum.com/v1", model: str = "gpt-3.5-turbo"):
        """
        Initialize the OpenAI API and model configuration.
        """
        openai.api_key = api_key
        openai.api_base = api_base
        self.model = model

    def generate_response(self, prompt_pairs, temperature=1, max_tokens=256, top_p=1, frequency_penalty=0, presence_penalty=0):
        """
        Generate a response based on multiple (system, user) prompt pairs.

        Args:
            prompt_pairs (list): List of tuples [("system prompt", "user prompt"), ...]
        """
        messages = []

        # Build messages list dynamically
        for system_msg, user_msg in prompt_pairs:
            messages.extend([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ])

        # Send to OpenAI API
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )

        return response.choices[0].message.content