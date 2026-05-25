import re

from skilllens.benchmarks.spreadsheetbench.inference.jupyter_kernel_cli import ClientJupyterKernel


def get_exec_client(url, conv_id):
    client = ClientJupyterKernel(url, conv_id)
    return client


# Patterns that strongly indicate executable Python code
# Match ```python ... ``` explicitly, or bare ``` ... ``` (no language tag).
# Do NOT match ```excel, ```sql, ```json etc. — those are not Python.
_PYTHON_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_BARE_BLOCK_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
_CODE_INDICATORS = re.compile(
    r"(?:^|\n)\s*(?:import |from .+ import |def |class |print\(|for |if |while |"
    r"with |open\(|pd\.|os\.|return |assert |\w+\s*=\s*)"
)


def extract_code(response: str) -> str:
    """Extract executable Python code from an LLM response.

    Strategy:
    1. If the response contains a ```python (or ```) fenced code block,
       extract the *last non-empty* block (the main/final code).
    2. If no code block is found, check whether the raw response itself
       looks like executable Python (contains import/def/print/… patterns).
       This covers models that output code without markdown fencing.
    3. Otherwise return "" — the response is natural-language text, NOT code.

    The old fallback of ``code = response`` when no code block was found
    caused natural-language "thinking" text to be sent to the kernel,
    resulting in SyntaxErrors and polluted trajectory records.
    """
    # 1. Try ```python blocks first (highest confidence)
    blocks = _PYTHON_BLOCK_RE.findall(response)
    if blocks:
        # Take the first non-empty ```python block — in multi-turn inference
        # the model should produce one code block per turn for execution.
        for block in blocks:
            code = block.strip()
            if code:
                return code

    # 2. Try bare ``` blocks (no language tag) — may be Python
    blocks = _BARE_BLOCK_RE.findall(response)
    if blocks:
        for block in blocks:
            code = block.strip()
            if code and _CODE_INDICATORS.search(code):
                return code

    # 3. No code block — does the raw text look like Python code?
    stripped = response.strip()
    if stripped and _CODE_INDICATORS.search(stripped):
        return stripped

    # 4. Not code — return empty so caller knows nothing was extracted
    return ""

def exec_code(client, code):
    res = client.execute(code)
    if res.find('-----') != -1:
        tracebacks = res.split('\n\n\n\n')
        error_feedback = ''
        for t in tracebacks:
            if t.find('Error') != -1:
                error_feedback += t + '\n'
                break
        for t in tracebacks:
            if len(t) >= len('Cell') and t[:len('Cell')] == 'Cell':
                error_feedback += t
                break
        error_feedback += tracebacks[-1]
        return error_feedback
    else:
        return res
