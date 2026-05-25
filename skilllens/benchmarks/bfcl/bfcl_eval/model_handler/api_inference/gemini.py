import atexit
import logging
import os
import signal
import time
from typing import Any

from skilllens.benchmarks.bfcl.bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from skilllens.benchmarks.bfcl.bfcl_eval.model_handler.base_handler import BaseHandler
from skilllens.benchmarks.bfcl.bfcl_eval.constants.enums import ModelStyle
from skilllens.benchmarks.bfcl.bfcl_eval.model_handler.utils import (
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    extract_system_prompt,
    format_execution_results_prompting,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
from google import genai
from google.genai.types import (
    AutomaticFunctionCallingConfig,
    Content,
    CreateCachedContentConfig,
    GenerateContentConfig,
    Part,
    ThinkingConfig,
    Tool,
)

logger = logging.getLogger(__name__)


class GeminiHandler(BaseHandler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.GOOGLE
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable must be set for Gemini models"
            )
        self.client = genai.Client(api_key=api_key)

        # Load skill/prompt inject content and create context cache if possible
        self._inject_text = self._load_inject_text()
        self._cache_name = None
        self._per_question_caches = []  # Track per-question caches for cleanup
        if self._inject_text:
            self._cache_name = self._create_cache(model_name, self._inject_text)

        # Register cleanup for per-question caches
        handler_ref = self
        def _cleanup_per_question():
            for cn in handler_ref._per_question_caches:
                try:
                    handler_ref.client.caches.delete(name=cn)
                except Exception:
                    pass
        atexit.register(_cleanup_per_question)

        # Ensure cache cleanup on SIGINT/SIGTERM (Ctrl+C resilience)
        def _signal_cleanup(signum, frame):
            _cleanup_per_question()
            # Re-raise to allow normal termination
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _signal_cleanup)
            except (OSError, ValueError):
                pass  # Can't set signal handler in non-main thread

    @staticmethod
    def _load_inject_text() -> str:
        """Load prompt and skill inject files from environment."""
        parts = []
        prompt_file = os.environ.get("PROMPT_INJECT_FILE", "")
        if prompt_file and os.path.isfile(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                parts.append(text)
        skill_file = os.environ.get("SKILL_INJECT_FILE", "")
        if skill_file and os.path.isfile(skill_file):
            with open(skill_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    def _create_cache(self, model_name: str, text: str) -> str | None:
        """Create a Gemini cached content object. Returns cache name or None."""
        try:
            cache = self.client.caches.create(
                model=model_name,
                config=CreateCachedContentConfig(
                    contents=[Content(parts=[Part(text=text)], role="user")],
                )
            )
            logger.info(f"Gemini context cache created: {cache.name}")
            # Register cleanup
            client_ref = self.client
            cache_name = cache.name
            def _cleanup():
                try:
                    client_ref.caches.delete(name=cache_name)
                    logger.info(f"Gemini context cache deleted: {cache_name}")
                except Exception:
                    pass
            atexit.register(_cleanup)
            # Also cleanup on signals
            prev_handlers = {}
            def _signal_cleanup_global(signum, frame):
                _cleanup()
                # Restore and re-raise
                signal.signal(signum, prev_handlers.get(signum, signal.SIG_DFL))
                os.kill(os.getpid(), signum)
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    prev_handlers[sig] = signal.getsignal(sig)
                    signal.signal(sig, _signal_cleanup_global)
                except (OSError, ValueError):
                    pass
            return cache.name
        except Exception as e:
            logger.warning(f"Failed to create Gemini context cache (will proceed without): {e}")
            return None

    def _delete_cache(self, cache_name: str):
        """Delete a cached content object."""
        try:
            self.client.caches.delete(name=cache_name)
        except Exception:
            pass

    @staticmethod
    def _substitute_prompt_role(prompts: list[dict]) -> list[dict]:
        # Allowed roles: user, model
        for prompt in prompts:
            if prompt["role"] == "user":
                prompt["role"] = "user"
            elif prompt["role"] == "assistant":
                prompt["role"] = "model"

        return prompts

    def decode_ast(self, result, language, has_tool_call_tag):
        if not self.is_fc_model:
            result = result.replace("```tool_code\n", "").replace("\n```", "")
            return default_decode_ast_prompting(result, language, has_tool_call_tag)
        else:
            if type(result) is not list:
                result = [result]
            return result

    def decode_execute(self, result, has_tool_call_tag):
        if not self.is_fc_model:
            result = result.replace("```tool_code\n", "").replace("\n```", "")
            return default_decode_execute_prompting(result, has_tool_call_tag)
        else:
            func_call_list = []
            for function_call in result:
                for func_name, func_args in function_call.items():
                    func_call_list.append(
                        f"{func_name}({','.join([f'{k}={repr(v)}' for k, v in func_args.items()])})"
                    )
            return func_call_list

    @retry_with_backoff(
        error_message_pattern=r".*(RESOURCE_EXHAUSTED|The model is overloaded|UNAVAILABLE|overloaded|503|429|500|502|504).*"
    )
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        api_response = self.client.models.generate_content(**kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        inference_data["inference_input_log"] = {
            "message": repr(inference_data["message"]),
            "tools": inference_data["tools"],
            "system_prompt": inference_data.get("system_prompt", None),
        }

        config = GenerateContentConfig(
            temperature=self.temperature,
            automatic_function_calling=AutomaticFunctionCallingConfig(disable=True),
            thinking_config=ThinkingConfig(include_thoughts=True),
        )

        # Priority: per-question cache > global cache > raw system_instruction
        per_q_cache = inference_data.get("_per_question_cache")
        if per_q_cache:
            config.cached_content = per_q_cache
        elif self._cache_name:
            config.cached_content = self._cache_name
        elif "system_prompt" in inference_data:
            config.system_instruction = inference_data["system_prompt"]

        if len(inference_data["tools"]) > 0:
            config.tools = [Tool(function_declarations=inference_data["tools"])]

        return self.generate_with_backoff(
            model=self.model_name,
            contents=inference_data["message"],
            config=config,
        )

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:

        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = self._substitute_prompt_role(
                test_entry["question"][round_idx]
            )

        inference_data["message"] = []

        system_prompt = extract_system_prompt(test_entry["question"][0])
        if system_prompt:
            inference_data["system_prompt"] = system_prompt

        # Create per-question cache for multi-turn efficiency
        # Includes: system_prompt + inject_text
        inference_data["_per_question_cache"] = None
        cache_content_parts = []
        if system_prompt:
            cache_content_parts.append(system_prompt)
        if self._inject_text:
            cache_content_parts.append(self._inject_text)
        if cache_content_parts:
            combined = "\n\n".join(cache_content_parts)
            try:
                cache = self.client.caches.create(
                    model=self.model_name,
                    config=CreateCachedContentConfig(
                        contents=[Content(parts=[Part(text=combined)], role="user")],
                    )
                )
                inference_data["_per_question_cache"] = cache.name
                self._per_question_caches.append(cache.name)
                logger.info(f"Per-question cache created: {cache.name}")
            except Exception as e:
                logger.debug(f"Per-question cache creation failed (content may be too short): {e}")

        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]

        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)

        inference_data["tools"] = tools

        return inference_data

    def _parse_query_response_FC(self, api_response: Any) -> dict:
        tool_call_func_names = []
        fc_parts = []
        text_parts = []
        reasoning_content = []

        if (
            len(api_response.candidates) > 0
            and api_response.candidates[0].content
            and api_response.candidates[0].content.parts
            and len(api_response.candidates[0].content.parts) > 0
        ):
            response_function_call_content = api_response.candidates[0].content

            for part in api_response.candidates[0].content.parts:
                # part.function_call is a FunctionCall object, so it will always be True even if it contains no function call
                # So we need to check if the function name is empty `""` to determine if Gemini returned a function call
                if part.function_call and part.function_call.name:
                    part_func_name = part.function_call.name
                    part_func_args = part.function_call.args
                    part_func_args_dict = {k: v for k, v in part_func_args.items()}

                    fc_parts.append({part_func_name: part_func_args_dict})
                    tool_call_func_names.append(part_func_name)
                # Aggregate reasoning content
                elif part.thought:
                    reasoning_content.append(part.text)
                else:
                    text_parts.append(part.text)

        else:
            response_function_call_content = Content(
                role="model",
                parts=[
                    Part(text="The model did not return any response."),
                ],
            )

        model_responses = fc_parts if fc_parts else text_parts

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": response_function_call_content,
            "tool_call_func_names": tool_call_func_names,
            "reasoning_content": "\n".join(reasoning_content),
            "input_token": api_response.usage_metadata.prompt_token_count,
            "output_token": api_response.usage_metadata.candidates_token_count,
            "cached_token": getattr(api_response.usage_metadata, 'cached_content_token_count', 0) or 0,
        }

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        # Inject skill/prompt text (if not using cache, add as first user message)
        if self._inject_text and not self._cache_name:
            inference_data["message"].append(
                Content(
                    role="user",
                    parts=[Part(text=self._inject_text)],
                )
            )
        for message in first_turn_message:
            inference_data["message"].append(
                Content(
                    role=message["role"],
                    parts=[
                        Part(text=message["content"]),
                    ],
                )
            )
        return inference_data

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        return self.add_first_turn_message_FC(inference_data, user_message)

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
        # Tool response needs to be converted to Content object as well.
        # One Content object for all tool responses.
        tool_response_parts = []
        for execution_result, tool_call_func_name in zip(
            execution_results, model_response_data["tool_call_func_names"]
        ):
            tool_response_parts.append(
                Part.from_function_response(
                    name=tool_call_func_name,
                    response={
                        "result": execution_result,
                    },
                )
            )

        tool_response_content = Content(role="user", parts=tool_response_parts)
        inference_data["message"].append(tool_response_content)

        return inference_data

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        inference_data["inference_input_log"] = {
            "message": repr(inference_data["message"]),
            "system_prompt": inference_data.get("system_prompt", None),
        }

        config = GenerateContentConfig(
            temperature=self.temperature,
            thinking_config=ThinkingConfig(include_thoughts=True),
        )

        # Priority: per-question cache > global cache > raw system_instruction
        per_q_cache = inference_data.get("_per_question_cache")
        if per_q_cache:
            config.cached_content = per_q_cache
        elif self._cache_name:
            config.cached_content = self._cache_name
        elif "system_prompt" in inference_data:
            config.system_instruction = inference_data["system_prompt"]

        api_response = self.generate_with_backoff(
            model=self.model_name,
            contents=inference_data["message"],
            config=config,
        )
        return api_response

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = self._substitute_prompt_role(
                test_entry["question"][round_idx]
            )

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )
        # Gemini has system prompt in a specific field
        system_prompt = extract_system_prompt(test_entry["question"][0])

        inference_data = {"message": []}
        if system_prompt:
            inference_data["system_prompt"] = system_prompt

        # Create per-question cache for multi-turn efficiency
        inference_data["_per_question_cache"] = None
        cache_content_parts = []
        if system_prompt:
            cache_content_parts.append(system_prompt)
        if self._inject_text:
            cache_content_parts.append(self._inject_text)
        if cache_content_parts:
            combined = "\n\n".join(cache_content_parts)
            try:
                cache = self.client.caches.create(
                    model=self.model_name,
                    config=CreateCachedContentConfig(
                        contents=[Content(parts=[Part(text=combined)], role="user")],
                    )
                )
                inference_data["_per_question_cache"] = cache.name
                self._per_question_caches.append(cache.name)
            except Exception:
                pass

        return inference_data

    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        if (
            len(api_response.candidates) > 0
            and api_response.candidates[0].content
            and api_response.candidates[0].content.parts
            and len(api_response.candidates[0].content.parts) > 0
        ):
            assert (
                len(api_response.candidates[0].content.parts) <= 2
            ), f"Length of response parts should be less than or equal to 2. {api_response.candidates[0].content.parts}"

            model_responses = ""
            reasoning_content = ""
            for part in api_response.candidates[0].content.parts:
                if part.thought:
                    reasoning_content = part.text
                else:
                    model_responses = part.text

        else:
            model_responses = "The model did not return any response."
            reasoning_content = ""

        return {
            "model_responses": model_responses,
            "reasoning_content": reasoning_content,
            "input_token": api_response.usage_metadata.prompt_token_count,
            "output_token": api_response.usage_metadata.candidates_token_count,
            "cached_token": getattr(api_response.usage_metadata, 'cached_content_token_count', 0) or 0,
        }

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        # Inject skill/prompt text (if not using cache, add as first user message)
        if self._inject_text and not self._cache_name:
            inference_data["message"].append(
                Content(
                    role="user",
                    parts=[Part(text=self._inject_text)],
                )
            )
        for message in first_turn_message:
            inference_data["message"].append(
                Content(
                    role=message["role"],
                    parts=[
                        Part(text=message["content"]),
                    ],
                )
            )
        return inference_data

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        return self.add_first_turn_message_prompting(inference_data, user_message)

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            Content(
                role="model",
                parts=[
                    Part(text=model_response_data["model_responses"]),
                ],
            )
        )
        return inference_data

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        formatted_results_message = format_execution_results_prompting(
            inference_data, execution_results, model_response_data
        )
        tool_message = Content(
            role="user",
            parts=[
                Part(text=formatted_results_message),
            ],
        )
        inference_data["message"].append(tool_message)
        return inference_data
