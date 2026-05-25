import json
import os
import time

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
from openai import OpenAI, RateLimitError, BadRequestError, APITimeoutError
from openai.types.responses import Response

from skilllens.benchmarks.bfcl.bfcl_eval.model_handler.api_inference._openai_client_factory import build_openai_client


class OpenAIResponsesHandler(BaseHandler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_RESPONSES
        self.client, self.api_model_name = build_openai_client(model_name)

    def _build_reasoning_param(self) -> dict:
        """Build the reasoning parameter dict, respecting REASONING_EFFORT env var."""
        reasoning = {"summary": "auto"}
        effort = os.getenv("REASONING_EFFORT")
        if effort and effort.lower() in ("low", "medium", "high"):
            reasoning["effort"] = effort.lower()
        return reasoning

    @staticmethod
    def _substitute_prompt_role(prompts: list[dict]) -> list[dict]:
        # OpenAI allows `system` role in the prompt, but it is meant for "messages added by OpenAI"
        # For our use case, it is recommended to use `developer` role instead.
        # See https://model-spec.openai.com/2025-04-11.html#definitions
        for prompt in prompts:
            if prompt["role"] == "system":
                prompt["role"] = "developer"

        return prompts

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
            return default_decode_execute_prompting(result, has_tool_call_tag)

    @retry_with_backoff(
        error_type=[RateLimitError, APITimeoutError],
        error_message_pattern=r"content_filter|content management",
    )
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        api_response = self.client.responses.create(**kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]

        inference_data["inference_input_log"] = {
            "message": repr(message),
            "tools": tools,
        }

        kwargs = {
            "input": message,
            "model": self.api_model_name,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "reasoning": self._build_reasoning_param(),
            "temperature": self.temperature,
        }

        # OpenAI reasoning models don't support temperature parameter
        if (
            "o3" in self.model_name
            or "o4-mini" in self.model_name
            or "gpt-5" in self.model_name
        ):
            del kwargs["temperature"]

        # Non-reasoning models don't support reasoning parameter
        else:
            del kwargs["reasoning"]
            del kwargs["include"]

        if len(tools) > 0:
            kwargs["tools"] = tools

        return self.generate_with_backoff(**kwargs)

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = self._substitute_prompt_role(
                test_entry["question"][round_idx]
            )

        inference_data["message"] = []

        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]

        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)

        inference_data["tools"] = tools

        return inference_data

    def _parse_query_response_FC(self, api_response: Response) -> dict:
        model_responses = []
        tool_call_ids = []

        for func_call in api_response.output:
            if func_call.type == "function_call":
                model_responses.append({func_call.name: func_call.arguments})
                tool_call_ids.append(func_call.call_id)

        if not model_responses:  # If there are no function calls
            model_responses = api_response.output_text

        # OpenAI reasoning models don't show full reasoning content in the api response,
        # but only a summary of the reasoning content.
        reasoning_content = ""
        for item in api_response.output:
            if item.type == "reasoning":
                for summary in item.summary:
                    reasoning_content += summary.text + "\n"

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": api_response.output,
            "tool_call_ids": tool_call_ids,
            "reasoning_content": reasoning_content,
            "input_token": api_response.usage.input_tokens,
            "output_token": api_response.usage.output_tokens,
            "cached_token": getattr(api_response.usage.input_tokens_details, 'cached_tokens', 0) or 0,
        }

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        # Inject prompt and skill into existing developer/system message (merged)
        _inject_parts = []
        prompt_file = os.environ.get("PROMPT_INJECT_FILE")
        if prompt_file and os.path.isfile(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
            if prompt_text:
                _inject_parts.append(prompt_text)
        skill_file = os.environ.get("SKILL_INJECT_FILE")
        if skill_file and os.path.isfile(skill_file):
            with open(skill_file, "r", encoding="utf-8") as f:
                skill_text = f.read().strip()
            if skill_text:
                _inject_parts.append(skill_text)
        if _inject_parts:
            if inference_data["message"] and inference_data["message"][0]["role"] in ("system", "developer"):
                inference_data["message"][0]["content"] += "\n\n" + "\n\n".join(_inject_parts)
            else:
                inference_data["message"].insert(0, {"role": "developer", "content": "\n\n".join(_inject_parts)})
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
        inference_data["message"].extend(
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
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": execution_result,
            }
            inference_data["message"].append(tool_message)

        return inference_data

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        inference_data["inference_input_log"] = {"message": repr(inference_data["message"])}

        kwargs = {
            "input": inference_data["message"],
            "model": self.api_model_name,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "reasoning": self._build_reasoning_param(),
            "temperature": self.temperature,
        }

        # OpenAI reasoning models don't support temperature parameter
        if (
            "o3" in self.model_name
            or "o4-mini" in self.model_name
            or "gpt-5" in self.model_name
        ):
            del kwargs["temperature"]

        # Non-reasoning models don't support reasoning parameter
        else:
            del kwargs["reasoning"]
            del kwargs["include"]

        return self.generate_with_backoff(**kwargs)

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )

        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = self._substitute_prompt_role(
                test_entry["question"][round_idx]
            )

        return {"message": []}

    def _parse_query_response_prompting(self, api_response: Response) -> dict:
        # OpenAI reasoning models don't show full reasoning content in the api response,
        # but only a summary of the reasoning content.
        reasoning_content = ""
        for item in api_response.output:
            if item.type == "reasoning":
                for summary in item.summary:
                    reasoning_content += summary.text + "\n"

        return {
            "model_responses": api_response.output_text,
            "model_responses_message_for_chat_history": api_response.output,
            "reasoning_content": reasoning_content,
            "input_token": api_response.usage.input_tokens,
            "output_token": api_response.usage.output_tokens,
            "cached_token": getattr(api_response.usage.input_tokens_details, 'cached_tokens', 0) or 0,
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
        inference_data["message"].extend(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_prompting(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        formatted_results_message = format_execution_results_prompting(
            inference_data, execution_results, model_response_data
        )
        inference_data["message"].append(
            {"role": "user", "content": formatted_results_message}
        )

        return inference_data
