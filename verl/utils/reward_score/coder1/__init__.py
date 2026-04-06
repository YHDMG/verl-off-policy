import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from .utils import _ERROR_MSG_PREFIX

_MAX_CHAR_DISPLAY = 2048
CODER1_EXEC = os.environ.get("CODER1_EXEC", "firejail")
CODER1_VERBOSE = os.environ.get("CODER1_VERBOSE", "0").lower() in {"1", "true", "yes"}

if CODER1_EXEC == "docker":
    from .docker_exec import code_exec_docker

    code_exec = code_exec_docker
elif CODER1_EXEC == "firejail":
    from .firejail_exec import code_exec_firejail

    code_exec = code_exec_firejail
elif CODER1_EXEC == "ces":
    from .ces_exec import remote_code_exec_ces

    code_exec = remote_code_exec_ces
elif CODER1_EXEC == "kira":
    from .kira_exec import remote_code_exec_kira

    code_exec = remote_code_exec_kira
else:
    raise ValueError(f"Unknown CODER1_EXEC: {CODER1_EXEC}")


def remote_check_stdio(code, stdin, stdout):
    succ, output = code_exec(code=code, stdin=stdin)
    return succ, output, stdin, stdout


def validate_response_structure(processed_str: str) -> bool:
    pattern = re.compile(r"<think>.*</think>.*<answer>.*</answer>$", re.DOTALL)
    return bool(pattern.match(processed_str.strip()))


def try_extract_solution(solution_str: str) -> str:
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return solution_str


CODE_PATTERN = re.compile(r"```(?:\w+)?\n(.*?)\n```", re.DOTALL)


def extract_code_from_string(solution_str: str) -> str:
    candidate = try_extract_solution(solution_str).strip()
    code_blocks = CODE_PATTERN.findall(candidate)
    if code_blocks:
        return "\n\n".join(block.strip() for block in code_blocks if block.strip()).strip()
    return candidate


def _coerce_ground_truth(ground_truth: Any) -> dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.item()
        if isinstance(ground_truth, dict):
            return ground_truth
    if isinstance(ground_truth, str):
        return json.loads(ground_truth)
    raise ValueError(f"Unsupported code ground_truth type: {type(ground_truth)}")


def _get_prompt_text(extra_info: Any) -> str:
    if isinstance(extra_info, np.ndarray):
        extra_info = extra_info.item()
    if isinstance(extra_info, dict):
        prompt = extra_info.get("prompt", "")
        if isinstance(prompt, str):
            return prompt
    return ""


def _build_result(
    score: float,
    acc: float,
    format_valid: bool,
    executor: str,
    error_type: str | None = None,
) -> dict[str, float | str]:
    result: dict[str, float | str] = {
        "score": float(score),
        "acc": float(acc),
        "format_valid": float(format_valid),
        "executor": executor,
    }
    if error_type is not None:
        result["error_type"] = error_type
    return result


def _compute_score(solution_str, ground_truth, extra_info, format_reward=0.0, answer_reward=1.0):
    reward_log = []
    prompt_text = _get_prompt_text(extra_info)
    format_valid = validate_response_structure(solution_str)
    solution_code = extract_code_from_string(solution_str)
    format_bonus = float(format_reward) if format_valid else 0.0

    if len(solution_code) == 0:
        reward_log.append("-" * 16 + "No executable code detected" + "-" * 16)
        reward_log.append(solution_str[:_MAX_CHAR_DISPLAY])
        if prompt_text:
            reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
            reward_log.append(prompt_text.replace("\n\n", "\n"))
        return _build_result(
            score=-float(answer_reward) + format_bonus,
            acc=0.0,
            format_valid=format_valid,
            executor=CODER1_EXEC,
            error_type="no_code",
        ), "\n".join(reward_log)

    reward_log.append("-" * 16 + "Extracted Code to Execute" + "-" * 16)
    reward_log.append(solution_code)
    ground_truth = _coerce_ground_truth(ground_truth)
    t_start = time.time()

    if "functional" in ground_truth:
        succ, output = code_exec(solution_code + "\n" + ground_truth["functional"])
        if not succ:
            reward_log.append("!" * 16 + f"Test execution failed in {time.time() - t_start:.1f}s" + "!" * 16)
            reward_log.append(output[:_MAX_CHAR_DISPLAY])
            if prompt_text:
                reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
                reward_log.append(prompt_text.replace("\n\n", "\n"))
            return _build_result(
                score=format_bonus,
                acc=0.0,
                format_valid=format_valid,
                executor=CODER1_EXEC,
                error_type="functional_exec_failed",
            ), "\n".join(reward_log)
    elif "pytest" in ground_truth:
        if CODER1_EXEC not in {"firejail", "docker"}:
            reward_log.append(f"Executor '{CODER1_EXEC}' does not support pytest-style code evaluation")
            return _build_result(
                score=format_bonus,
                acc=0.0,
                format_valid=format_valid,
                executor=CODER1_EXEC,
                error_type="pytest_executor_unsupported",
            ), "\n".join(reward_log)
        succ, output = code_exec(solution_code, pytest=ground_truth["pytest"])
        if not succ:
            reward_log.append("!" * 16 + f"Test execution failed in {time.time() - t_start:.1f}s" + "!" * 16)
            reward_log.append(output[:_MAX_CHAR_DISPLAY])
            if prompt_text:
                reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
                reward_log.append(prompt_text.replace("\n\n", "\n"))
            return _build_result(
                score=format_bonus,
                acc=0.0,
                format_valid=format_valid,
                executor=CODER1_EXEC,
                error_type="pytest_exec_failed",
            ), "\n".join(reward_log)
    elif "inputs" in ground_truth and "outputs" in ground_truth:
        stdin_list = ground_truth["inputs"]
        stdout_list = ground_truth["outputs"]
        max_workers = int(os.environ.get("CODER1_MAX_WORKERS", min(16, len(stdin_list))))
        output = ""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(remote_check_stdio, solution_code, stdin, stdout)
                for stdin, stdout in zip(stdin_list, stdout_list, strict=True)
            ]
            for future in as_completed(futures):
                succ, output, stdin, stdout = future.result()
                if not succ or output.strip() != stdout.strip():
                    output = output[:_MAX_CHAR_DISPLAY]
                    reward_log.append("!" * 16 + f"Test execution failed in {time.time() - t_start:.1f}s" + "!" * 16)
                    reward_log.append(f"Input: {repr(stdin)}")
                    reward_log.append(f"Expected: {repr(stdout.strip())}")
                    reward_log.append(
                        f"Actual: {output if output.startswith(_ERROR_MSG_PREFIX) else repr(output.strip())}"
                    )
                    if prompt_text:
                        reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
                        reward_log.append(prompt_text.replace("\n\n", "\n"))
                    return _build_result(
                        score=format_bonus,
                        acc=0.0,
                        format_valid=format_valid,
                        executor=CODER1_EXEC,
                        error_type="stdio_exec_failed",
                    ), "\n".join(reward_log)
    else:
        raise ValueError(
            "Current supports for code ground truth are ['functional', 'pytest', 'inputs/outputs'], "
            f"got keys: {sorted(ground_truth.keys())}"
        )

    reward_log.append("+" * 16 + "Test execution passed" + "+" * 16)
    reward_log.append(output[:_MAX_CHAR_DISPLAY] if isinstance(output, str) else "")
    return _build_result(
        score=float(answer_reward) + format_bonus,
        acc=1.0,
        format_valid=format_valid,
        executor=CODER1_EXEC,
    ), "\n".join(reward_log)


def compute_score(solution_str, ground_truth, extra_info=None, format_reward=0.2, answer_reward=1.0):
    result, reward_log = _compute_score(
        solution_str,
        ground_truth,
        extra_info=extra_info,
        format_reward=format_reward,
        answer_reward=answer_reward,
    )
    if CODER1_VERBOSE:
        marker = "PASS" if float(result["acc"]) > 0.5 else "FAIL"
        print(f"[coder1:{marker}] score={result['score']} acc={result['acc']}\n{reward_log}\n")
    return result
