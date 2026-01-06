# LLM Abliteration

This project performs "abliteration" on a Large Language Model (LLM) to remove refusal mechanisms. It uses `transformer_lens` to analyze activation patterns and modify model weights to Orthogonalize refusal directions.

## Prerequisites

- Python 3.8+
- Git

## Setup

### Windows

1.  **Create a virtual environment:**
    ```powershell
    python -m venv venv
    ```

2.  **Activate the environment:**
    *   **PowerShell:**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
    *   **Command Prompt:**
        ```cmd
        venv\Scripts\activate.bat
        ```

3.  **Install dependencies:**
    ```powershell
    pip install -r requirements.txt
    ```

### Linux

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    ```
    *(Note: You might need to run `sudo apt install python3-venv` first on some distributions)*

2.  **Activate the environment:**
    ```bash
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Login to Hugging Face:**
    The script requires access to models and tries to push the result to the Hub. You need to log in with a token (ensure you have write permissions if pushing):
    ```bash
    huggingface-cli login
    ```

2.  **Run the script:**
    Ensure your virtual environment is active, then run:
    ```bash
    python LLM-Abliteration.py
    ```

**Note:** This script downloads large model files and performs intensive computations. Ensure you have sufficient RAM and VRAM available.
