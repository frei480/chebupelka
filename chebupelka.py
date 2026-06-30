"""Minimal coding agent loop — one tool: bash."""

import json
import sys
import subprocess

import platform
import asyncio
import platform
import httpx


# Настройки
LLM_BASE_URL = "http://localhost:1234/v1"
LLM_API_KEY = "..."
LLM_MODEL = "qwen/qwen3.6-27b"
LLM_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {LLM_API_KEY}",
}
MAX_TURNS = 1000

# Определяем оболочку в зависимости от ОС
IS_WINDOWS = platform.system() == "Windows"
SHELL_NAME = "powershell" if IS_WINDOWS else "bash"
ENCODING_FALLBACKS = ["cp866", "cp1251", "utf-8"]


SYSTEM_PROMPT = f"""\
You are a coding agent. Your job is to help the user with programming tasks.
You have access to ONE tool: `{SHELL_NAME}` — which executes commands and returns stdout/stderr.

Workflow:
1. Plan what needs to be done.
2. Use `{SHELL_NAME}` to read files, run commands, write code, etc.
3. After gathering enough information or completing the task, give your final answer in natural language.
4. To finish, reply with a regular message (no tool call).

IMPORTANT: You are running on {platform.system()}. Use {SHELL_NAME} syntax.
Be concise."""

LLM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": SHELL_NAME,
            "description": f"Execute a {SHELL_NAME} command and return the output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    }
]


def decode_output(data: bytes) -> str:
    for encoding in ENCODING_FALLBACKS:
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def run_bash(command: str) -> str:
    try:
        if IS_WINDOWS:
            shell_cmd = ["powershell", "-NoProfile", "-Command", command]
        else:
            shell_cmd = command
        command_result = subprocess.run(
            shell_cmd, shell=not IS_WINDOWS, capture_output=True, timeout=120
        )
        out = decode_output(command_result.stdout) + (
            f"\nSTDERR:\n{decode_output(command_result.stderr)}"
            if command_result.stderr
            else ""
        )
        return f"Exit code: {command_result.returncode}\n{out}"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120s"


async def run_shell(command: str) -> str:
    """Запускает команду в оболочке (PowerShell или Bash) асинхронно."""
    try:
        if IS_WINDOWS:
            # Для Windows используем powershell
            process = await asyncio.create_subprocess_exec(
                "powershell.exe",
                "-Command",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            # Для Linux/Mac используем bash
            process = await asyncio.create_subprocess_shell(
                command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)

        result_out = stdout.decode(errors="replace")
        result_err = decode_output(stderr)

        out = result_out + (f"\nSTDERR:\n{result_err}" if result_err else "")
        return f"Exit code: {process.returncode}\n{out}"
    except asyncio.TimeoutError:
        return "Error: command timed out after 120s"
    except Exception as e:
        return f"Error: {str(e)}"


async def call_llm_streaming(messages, client: httpx.AsyncClient):
    """Выполняет стриминговый запрос к LLM, обрабатывая текст, размышления и инструменты."""
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "tools": LLM_TOOLS,
        "tool_choice": "auto",
        "temperature": 0.1,
        "stream": True,
    }

    full_content = ""
    full_reasoning = ""
    tool_calls_map = {}

    async with client.stream(
        "POST", f"{LLM_BASE_URL}/chat/completions", json=payload, timeout=None
    ) as response:
        response.raise_for_status()

        print("🤖 ", end="", flush=True)

        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            line = line[6:]
            if line == "[DONE]":
                break

            chunk = json.loads(line)
            delta = chunk["choices"][0].get("delta", {})

            # 1. Обработка Reasoning (для моделей типа DeepSeek-R1)
            if "reasoning_content" in delta and delta["reasoning_content"]:
                r_chunk = delta["reasoning_content"]
                full_reasoning += r_chunk
                # Выводим размышления серым цветом (в терминалах поддерживающих ANSI)
                print(f"\033[90m{r_chunk}\033[0m", end="", flush=True)

            # 2. Обработка обычного контента
            if "content" in delta and delta["content"]:
                c_chunk = delta["content"]
                full_content += c_chunk
                print(c_chunk, end="", flush=True)

            # 3. Сбор данных о вызове инструментов
            if "tool_calls" in delta:
                for tc_delta in delta["tool_calls"]:
                    idx = tc_delta["index"]
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }

                    if "id" in tc_delta:
                        tool_calls_map[idx]["id"] += tc_delta["id"]
                    if "type" in tc_delta:
                        tool_calls_map[idx]["type"] = tc_delta["type"]
                    if "function" in tc_delta:
                        f = tc_delta["function"]
                        if "name" in f:
                            tool_calls_map[idx]["function"]["name"] += f["name"]
                        if "arguments" in f:
                            tool_calls_map[idx]["function"]["arguments"] += f[
                                "arguments"
                            ]

        print()  # Конец строки после стриминга

    tool_calls = list(tool_calls_map.values()) if tool_calls_map else []
    return full_content, full_reasoning, tool_calls


async def agent_loop(user_message: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # Используем один клиент на весь цикл
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    async with httpx.AsyncClient(headers=headers, timeout=None) as client:
        for turn in range(1, MAX_TURNS + 1):
            print(f"\n{'=' * 60}\n🔄 Turn {turn}\n{'=' * 60}")

            content, reasoning, tool_calls = await call_llm_streaming(messages, client)
            cleaned_content = content.strip()
            # Добавляем ответ ассистента в историю (включая reasoning если нужно сохранить контекст мысли)
            assistant_msg = {"role": "assistant", "content": cleaned_content or None}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            # Добавляем размышления в историю, чтобы модель видела свой ход мысли в следующем ходу
            # if reasoning:
            #     assistant_msg["content"] = reasoning
            if not tool_calls:
                print("✅ Agent finished")
                break

            messages.append(assistant_msg)
            for tool_call in tool_calls:
                func_name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"])
                tc_id = tool_call["id"]

                print(f"🔧 Tool: {func_name}({args.get('command', '')})")

                result = await run_shell(args["command"])

                print(f"   → {result[:500]}{'...' if len(result) > 500 else ''}")

                messages.append(
                    {"role": "tool", "tool_call_id": tc_id, "content": result}
                )


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    if not prompt.strip():
        print("Usage: python agent.py 'your task'")
        sys.exit(1)

    try:
        asyncio.run(agent_loop(prompt))
    except KeyboardInterrupt:
        print("\nStopped by user.")
