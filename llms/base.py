from typing import Optional

class LLM:
    """
    A class representing a Language Model (LLM) for text generation tasks,
    utilizing transformer pipelines for advanced natural language processing.

    Attributes:
        model_id (str): Identifier for the pre-trained model to be used. Default is "meta-llama/Meta-Llama-3-8B-Instruct".
        access_token (str): Access token required for authenticating with the model hosting service.
        pipeline (transformers.Pipeline): The loaded transformer pipeline for text generation.

    Methods:
        load_pipeline(): Configures and loads the transformer pipeline with the specified model.
        invoke(messages, cfg): Generates text based on the input messages using the pre-loaded pipeline with an optional configuration.

    Example:
        >>> llm = LLM(access_token="your_access_token_here")
        >>> llm.load_pipeline()
        >>> response = await llm.invoke([LLMMessage(prompt="Tell me a story about a wizard")])
        >>> print(response)
    """

    model_id: str = ""
    access_token: str = ""

    def __init__(self, model_id: Optional[str] = None, access_token: str = "", system_prompt: str = "You are an assistant."):
        """
        Initializes the LLM instance with the given model ID and access token.

        Parameters:
            model_id (str): Identifier for the pre-trained model to be used for text generation. Default is "meta-llama/Meta-Llama-3-8B-Instruct".
            access_token (str): Access token for using the model hosting service. Must be provided by the user.
        """
        self.access_token = access_token
        if model_id:
            self.model_id = model_id
        self.system_prompt = system_prompt