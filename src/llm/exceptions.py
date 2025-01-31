class InvalidLLMResponseException(Exception):
    """Raised when the response of the LLM is not as expected"""
    def __init__(self, message: str, llm_result: str):
        self.message: str = message
        self.llm_result: str = llm_result

    def __str__(self):
        message: str = self.message
        llm_result: str  = self.llm_result
        return f"Error parsing LLM response {llm_result}: {message}"
