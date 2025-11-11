from langchain_ollama import ChatOllama
import subprocess
import platform
import sys
import os
import argparse

## Must install ollama first: https://ollama.com/
# Then start the ollama server: ollama serve    
# then download the model: ollama pull gpt-oss:20b
model = ChatOllama(
    model="gpt-oss:20b",
    validate_model_on_init=True,
    temperature=0.8,
    num_predict=256,
    # other params ...
)


def get_shell():
    """Detect and return the appropriate shell for the current platform."""
    system = platform.system()
    
    if system == "Windows":
        # Use PowerShell on Windows
        return "powershell.exe"
    elif system == "Linux":
        # Use bash on Linux
        return "/bin/bash"
    elif system == "Darwin":  # macOS
        # Use zsh on macOS (default since Catalina)
        return "/bin/zsh"
    else:
        # Fallback to sh
        return "/bin/sh"

def run_shell_command(command: str, shell=None, interactive=False):
    """
    Run a command in the platform-specific shell.
    
    Args:
        command: Command string to execute
        shell: Shell executable path (auto-detected if None)
        interactive: If True, spawns interactive shell; if False, runs command and exits
    
    Returns:
        CompletedProcess object with stdout, stderr, returncode
    """
    if shell is None:
        shell = get_shell()
    
    system = platform.system()
    
    if interactive:
        # Spawn an interactive shell
        if system == "Windows":
            # PowerShell interactive mode
            subprocess.run([shell], shell=False)
        else:
            # Unix-like interactive shell
            subprocess.run([shell, "-i"], shell=False)
    else:
        # Run a single command
        if system == "Windows":
            # PowerShell: use -Command flag
            result = subprocess.run(
                [shell, "-Command", command],
                capture_output=True,
                text=True,
                shell=False
            )
        else:
            # Unix: use -c flag
            result = subprocess.run(
                [shell, "-c", command],
                capture_output=True,
                text=True,
                shell=False
            )
        return result

if __name__ == "__main__":
    prompt = sys.argv[1] ## This is the Source for Command Injection through an LLM proxy.
    messages = [
        ("system", f"You are a IT assistant who speaks only in shell commands. You are in a {get_shell()} shell."),
        ("human", prompt),
    ]
    model.invoke(messages)
    stream = model.stream(messages)
    full = next(stream)
    for chunk in stream:
        full += chunk
    print("Command: " + str(full.content))
    result = run_shell_command(str(full.content)) ## This is the Sink for Command Injection
    print(result)
    
