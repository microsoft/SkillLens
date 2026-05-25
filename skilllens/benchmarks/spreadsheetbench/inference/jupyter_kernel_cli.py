import json
import time
import logging
import requests

logger = logging.getLogger(__name__)


class ClientJupyterKernel:
    def __init__(self, url, conv_id, timeout=300, max_retries=3):
        self.url = url
        self.conv_id = conv_id
        self.timeout = timeout  # seconds per request (container creation can be slow)
        self.max_retries = max_retries
        print(f"ClientJupyterKernel initialized with url={url} and conv_id={conv_id}")

    def execute(self, code):
        payload = {"convid": self.conv_id, "code": code}

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    data=json.dumps(payload),
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code != 200:
                    raise RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:500]}"
                    )

                if not response.text or not response.text.strip():
                    raise RuntimeError("Empty response body from execution server")

                response_data = response.json()

                if response_data.get("new_kernel_created"):
                    logger.info(
                        f"New kernel created for conversation {self.conv_id}"
                    )
                return response_data["result"]

            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                json.JSONDecodeError,
                RuntimeError,
            ) as e:
                wait_time = 5 * (2 ** (attempt - 1))  # 5s, 10s, 20s
                logger.warning(
                    f"Exec RETRY | conv={self.conv_id} | attempt={attempt}/{self.max_retries} | "
                    f"error={type(e).__name__}: {str(e)[:200]} | waiting={wait_time}s"
                )
                if attempt < self.max_retries:
                    time.sleep(wait_time)
                    continue
                else:
                    error_msg = (
                        f"Code execution failed after {self.max_retries} retries: "
                        f"{type(e).__name__}: {str(e)[:300]}"
                    )
                    logger.error(f"Exec FAIL | conv={self.conv_id} | {error_msg}")
                    return f"[Execution Error] {error_msg}"

            except Exception as e:
                error_msg = f"Unexpected error: {type(e).__name__}: {str(e)[:300]}"
                logger.error(f"Exec FAIL | conv={self.conv_id} | {error_msg}")
                return f"[Execution Error] {error_msg}"
