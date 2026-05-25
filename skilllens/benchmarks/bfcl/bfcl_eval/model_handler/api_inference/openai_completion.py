import json
import os
import time
from typing import Any

from skilllens.benchmarks.bfcl.bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from skilllens.benchmarks.bfcl.bfcl_eval.model_handler.base_handler import BaseHandler
from skilllens.benchmarks.bfcl.bfcl_eval.constants.enums import ModelStyle
from skilllens.benchmarks.bfcl.bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    format_execution_results_prompting,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
from openai import OpenAI, RateLimitError, APITimeoutError

from skilllens.benchmarks.bfcl.bfcl_eval.model_handler.api_inference._openai_client_factory import build_openai_client


class OpenAICompletionsHandler(BaseHandler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        self.client, self.api_model_name = build_openai_client(model_name)

    def decode_ast(self, result, language, has_tool_call_tag):
        if self.is_fc_model:
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = json.loads(invoked_function[name])
                decoded_output.append({name: params})
            return decoded_output
        else:
            return default_decode_ast_prompting(result, language, has_tool_call_tag)

    def decode_execute(self, result, has_tool_call_tag):
        if self.is_fc_model:
            return convert_to_function_call(result)
        else:
            return default_decode_execute_prompting(result)

    @retry_with_backoff(
        error_type=[RateLimitError, APITimeoutError],
        error_message_pattern=r"content_filter|content management",
    )
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        api_response = self.client.chat.completions.create(**kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        kwargs = {
            "messages": message,
            "model": self.api_model_name,
            "temperature": self.temperature,
            "store": False,
        }

        if len(tools) > 0:
            kwargs["tools"] = tools

        # Enable thinking/reasoning for Qwen3.5 models served via sglang
        if "qwen" in self.api_model_name.lower():
            kwargs["extra_body"] = {"enable_thinking": True}

        # Reasoning effort support
        effort = os.getenv("REASONING_EFFORT")
        if "gemini" in self.model_name.lower():
            # Gemini models via OpenAI-compatible proxy: pass reasoning_effort, skip temperature
            if effort:
                kwargs["reasoning_effort"] = effort.lower()
            del kwargs["temperature"]
        elif effort and "gpt-5" in self.model_name.lower():
            # GPT-5.x models support reasoning_effort
            kwargs["reasoning_effort"] = effort.lower()

        return self.generate_with_backoff(**kwargs)

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]

        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)

        inference_data["tools"] = tools

        return inference_data

    def _parse_query_response_FC(self, api_response: Any) -> dict:
        try:
            model_responses = [
                {func_call.function.name: func_call.function.arguments}
                for func_call in api_response.choices[0].message.tool_calls
            ]
            tool_call_ids = [
                func_call.id for func_call in api_response.choices[0].message.tool_calls
            ]
        except:
            model_responses = api_response.choices[0].message.content
            tool_call_ids = []

        model_responses_message_for_chat_history = api_response.choices[0].message

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "tool_call_ids": tool_call_ids,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
            "cached_token": getattr(getattr(api_response.usage, 'prompt_tokens_details', None), 'cached_tokens', 0) or 0,
        }

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        # Inject prompt and skill into the first system message (merged)
        # to avoid multiple system messages which vLLM/Qwen rejects.
        import os as _os
        _inject_parts = []
        _prompt_file = _os.environ.get("PROMPT_INJECT_FILE")
        if _prompt_file and _os.path.isfile(_prompt_file):
            with open(_prompt_file, "r", encoding="utf-8") as _f:
                _prompt_text = _f.read().strip()
            if _prompt_text:
                _inject_parts.append(_prompt_text)
        _skill_file = _os.environ.get("SKILL_INJECT_FILE")
        if _skill_file and _os.path.isfile(_skill_file):
            with open(_skill_file, "r", encoding="utf-8") as _f:
                _skill_text = _f.read().strip()
            if _skill_text:
                _inject_parts.append(_skill_text)
        if _inject_parts:
            # Merge into existing system message if present, otherwise prepend one
            if inference_data["message"] and inference_data["message"][0]["role"] == "system":
                inference_data["message"][0]["content"] += "\n\n" + "\n\n".join(_inject_parts)
            else:
                inference_data["message"].insert(0, {"role": "system", "content": "\n\n".join(_inject_parts)})
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        # Add the execution results to the current round result, one at a time
        for execution_result, tool_call_id in zip(
            execution_results, model_response_data["tool_call_ids"]
        ):
            tool_message = {
                "role": "tool",
                "content": execution_result,
                "tool_call_id": tool_call_id,
            }
            inference_data["message"].append(tool_message)

        return inference_data

    def _add_reasoning_content_if_available_FC(
        self, api_response: Any, response_data: dict
    ) -> None:
        """
        OpenAI models don't show reasoning content in the api response,
        but many other models that use the OpenAI interface do, such as DeepSeek and Grok.
        This method is included here to avoid code duplication.

        These models often don't take reasoning content in the chat history for next turn.
        Thus, this method saves reasoning content to response_data (for local result file) if present in the response,
        but does not include it in the chat history.
        """
        # Original assistant message object (contains `reasoning_content` on DeepSeek).
        message = api_response.choices[0].message

        # Preserve tool_call information but strip the unsupported `reasoning_content` field before inserting into chat history.
        if getattr(message, "tool_calls", None):
            assistant_message = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ],
            }
            response_data["model_responses_message_for_chat_history"] = assistant_message

        # If no tool_calls, we still need to strip reasoning_content.
        elif hasattr(message, "reasoning_content"):
            response_data["model_responses_message_for_chat_history"] = {
                "role": "assistant",
                "content": message.content,
            }

        # Capture the reasoning trace so it can be logged to the local result file.
        if hasattr(message, "reasoning_content"):
            response_data["reasoning_content"] = message.reasoning_content

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        inference_data["inference_input_log"] = {"message": repr(inference_data["message"])}

        kwargs = {
            "messages": inference_data["message"],
            "model": self.api_model_name,
            "temperature": self.temperature,
            "store": False,
        }

        # Enable thinking/reasoning for Qwen3.5 models served via sglang
        if "qwen" in self.api_model_name.lower():
            kwargs["extra_body"] = {"enable_thinking": True}

        # Reasoning effort support
        effort = os.getenv("REASONING_EFFORT")
        if "gemini" in self.model_name.lower():
            # Gemini models via OpenAI-compatible proxy: pass reasoning_effort, skip temperature
            if effort:
                kwargs["reasoning_effort"] = effort.lower()
            del kwargs["temperature"]
        elif effort and "gpt-5" in self.model_name.lower():
            # GPT-5.x models support reasoning_effort
            kwargs["reasoning_effort"] = effort.lower()

        return self.generate_with_backoff(**kwargs)

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )

        return {"message": []}

    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        return {
            "model_responses": api_response.choices[0].message.content,
            "model_responses_message_for_chat_history": api_response.choices[0].message,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
            "cached_token": getattr(getattr(api_response.usage, 'prompt_tokens_details', None), 'cached_tokens', 0) or 0,
        }

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        formatted_results_message = format_execution_results_prompting(
            inference_data, execution_results, model_response_data
        )
        inference_data["message"].append(
            {"role": "user", "content": formatted_results_message}
        )

        return inference_data

    def _add_reasoning_content_if_available_prompting(
        self, api_response: Any, response_data: dict
    ) -> None:
        """
        OpenAI models don't show reasoning content in the api response,
        but many other models that use the OpenAI interface do, such as DeepSeek and Grok.
        This method is included here to avoid code duplication.

        These models often don't take reasoning content in the chat history for next turn.
        Thus, this method saves reasoning content to response_data (for local result file) if present in the response,
        but does not include it in the chat history.
        """
        message = api_response.choices[0].message
        if hasattr(message, "reasoning_content"):
            response_data["reasoning_content"] = message.reasoning_content
            # Reasoning content should not be included in the chat history
            response_data["model_responses_message_for_chat_history"] = {
                "role": "assistant",
                "content": str(response_data["model_responses"]),
            }
