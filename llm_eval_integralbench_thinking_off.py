import json
import math
import os
import re
import time

import mpmath
import ollama
import pandas as pd
import sympy as sp


# ============================================================
# KONFIGURACJA (TRYB: THINKING WYŁĄCZONY)
# ============================================================

base_dir = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(
    base_dir,
    "datasets",
    "integralbench",
    "dataset.jsonl"
)

output_path = os.path.join(
    base_dir,
    "eval",
    "integralbench",
    "evaluation_llm_results_thinking_off.csv"
)

MODEL_NAME = "qwen3.5:9b"
NUMERICAL_TOLERANCE = 1e-4  # Tolerancja błędu względnego/bezwzględnego

x, y, z, t = sp.symbols("x y z t")

# Słownik funkcji dostępnych podczas parsowania wyrażenia z LLM
local_dict = {
    "x": x,
    "y": y,
    "z": z,
    "t": t,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "cot": sp.cot,
    "sec": sp.sec,
    "csc": sp.csc,
    "asin": sp.asin,
    "arcsin": sp.asin,
    "acos": sp.acos,
    "arccos": sp.acos,
    "atan": sp.atan,
    "arctan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "asinh": sp.asinh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
    "abs": sp.Abs,
    "E": sp.E,
    "e": sp.E,
    "pi": sp.pi,
    "gamma": sp.gamma,
    "beta": sp.beta,
    "erf": sp.erf,
    "erfc": sp.erfc,
    "erfi": sp.erfi,
    "Si": sp.Si,
    "Ci": sp.Ci,
    "Ei": sp.Ei,
    "li": sp.li,
    "zeta": sp.zeta,
    "polylog": sp.polylog,
    "hyper": sp.hyper,
    "elliptic_k": sp.elliptic_k,
    "elliptic_e": sp.elliptic_e,
    "elliptic_f": sp.elliptic_f,
    "elliptic_pi": sp.elliptic_pi,
    "besselj": sp.besselj,
    "bessely": sp.bessely,
    "besseli": sp.besseli,
    "besselk": sp.besselk,
}


# ============================================================
# WCZYTYWANIE ZBIORU DANYCH (JSONL / JSON)
# ============================================================

def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono zbioru danych: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as file:
        first_char = file.read(1)
        file.seek(0)

        if first_char == "[":
            records = json.load(file)
        else:
            for line_num, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Błąd parsowania JSONL w linii {line_num}: {exc}"
                    ) from exc

    return pd.DataFrame(records)


# ============================================================
# WYKRYWANIE ZMIENNYCH CAŁKOWANIA Z ZAPISU LATEX
# ============================================================

def extract_variables_from_latex(latex_problem):
    """Wyciąga zmienne z końcówek typu dx, dy, dt w kolejności występowania."""
    if not isinstance(latex_problem, str):
        return ["x"]

    matches = re.findall(r"(?:\\mathrm\{d\}|\\text\{d\}|d)\s*([a-zA-Z])", latex_problem)
    if not matches:
        return ["x"]

    seen = set()
    ordered_vars = []
    for var in matches:
        if var not in seen:
            seen.add(var)
            ordered_vars.append(var)
    return ordered_vars


def extract_json_from_text(text):
    """Wyciąga ostatni poprawny obiekt JSON z odpowiedzi (obsługuje też zagnieżdżone klamry)."""
    if not text:
        raise ValueError("Model zwrócił pustą odpowiedź (empty content).")

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    decoder = json.JSONDecoder()
    last_valid_obj = None

    for idx, ch in enumerate(cleaned):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(cleaned[idx:])
                if isinstance(obj, dict) and "result_python" in obj:
                    last_valid_obj = obj
            except json.JSONDecodeError:
                continue

    if last_valid_obj is None:
        raise ValueError(
            f"Nie znaleziono poprawnego obiektu JSON z kluczem 'result_python': {cleaned[:200]}"
        )

    return last_valid_obj


# ============================================================
# SPRAWDZANIE POPRAWNOŚCI MATEMATYCZNEJ (EWALUACJA NUMERYCZNA)
# ============================================================

def evaluate_to_float(expr_str):
    """Konwertuje wyrażenie tekstowe SymPy/Python na wartość zmiennoprzecinkową."""
    cleaned = expr_str.strip().replace("^", "**")

    expr = sp.sympify(cleaned, locals=local_dict)
    numeric_val = complex(expr.evalf(30))

    if abs(numeric_val.imag) > 1e-7:
        raise ValueError(f"Wynik ma niezerową część urojoną: {numeric_val}")

    val = float(numeric_val.real)
    if math.isnan(val) or math.isinf(val):
        raise ValueError(f"Nieprawidłowa wartość numeryczna: {val}")

    return val


def check_definite_integral(actual_output, expected_numerical, tol=NUMERICAL_TOLERANCE):
    """Porównuje wartość numeryczną odpowiedzi modelu z wartością referencyjną."""
    actual_val = evaluate_to_float(actual_output)
    expected_val = float(expected_numerical)

    abs_diff = abs(actual_val - expected_val)

    if abs(expected_val) > 1e-8:
        rel_diff = abs_diff / abs(expected_val)
        is_correct = (rel_diff <= tol) or (abs_diff <= tol)
    else:
        is_correct = abs_diff <= tol

    return is_correct, abs_diff, actual_val


# ============================================================
# OBLICZANIE CAŁEK PRZEZ OLLAMA (THINKING = FALSE)
# ============================================================

def evaluate_integrals_with_ollama(model_name, dataset):
    results = []

    system_prompt = """
You are an expert mathematical assistant specializing in symbolic and definite integration.

Calculate the exact value of the requested definite integral given in LaTeX format.

Return ONLY valid JSON in the following format:
{
    "result_python": "your expression"
}

Rules for "result_python":
- Use valid Python / SymPy syntax:
  - ** for powers (never use ^)
  - * for multiplication (explicit multiplication everywhere, e.g., 2*x, not 2x)
  - Standard constants: pi, E
  - Standard functions: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, exp, log, sqrt, gamma, beta, erf, erfc, Si, Ci, Ei, zeta, polylog, elliptic_k, elliptic_e, besselj, bessely, besseli, besselk
- You may return either a closed-form symbolic constant expression (e.g., "pi**2 / 6 - 0.5*log(2)") OR a high-precision decimal number if no elementary closed form is straightforward.
- Do NOT include integrals, dummy variables (like x), arbitrary constants (+ C), explanations, or Markdown.
"""

    for idx, row in dataset.iterrows():
        problem_id = row.get("problem_number", idx + 1)
        source_category = row.get("source", "")
        difficulty = row.get("difficulty", "")
        prompt_latex = row.get("problem", "")
        symbolic_answer = row.get("symbolic_answer", "")
        numerical_answer = row.get("numerical_answer", None)

        variables_order = extract_variables_from_latex(prompt_latex)

        expected_output_str = (
            f"{numerical_answer} | {symbolic_answer}"
            if symbolic_answer
            else str(numerical_answer)
        )

        user_prompt = f"""
Evaluate the following definite integral:

{prompt_latex}

Return the exact result in the required JSON format using valid SymPy/Python syntax.
"""

        print(
            f"\n[{idx + 1}/{len(dataset)}] "
            f"Problem ID: {problem_id} | Source: {source_category}"
        )

        start_time = time.perf_counter()

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Jawnie wyłączony tryb thinking oraz wymuszony format JSON
            response = ollama.chat(
                model=model_name,
                messages=messages,
                format="json",
                think=False,
                options={
                    "temperature": 0.0,
                    "num_ctx": 8192,
                    "num_predict": 2048,
                },
            )

            elapsed_time = time.perf_counter() - start_time
            response_content = (response.message.content or "").strip()

            if not response_content:
                raise ValueError(
                    f"Model zwrócił pusty content (raw message: {response.message})."
                )

            response_json = extract_json_from_text(response_content)
            actual_output = str(response_json["result_python"]).strip()

            if pd.isna(numerical_answer):
                raise ValueError("Brak wartości 'numerical_answer' w rekordzie datasetu.")

            correct, abs_diff, actual_numeric = check_definite_integral(
                actual_output,
                numerical_answer
            )

            status = "Correct" if correct else "Incorrect"
            print(
                f"-> {status} | Pred: {actual_numeric:.8g} vs Ref: {float(numerical_answer):.8g} | "
                f"Time: {elapsed_time:.2f}s"
            )

            if not correct:
                print(f"   Abs difference: {abs_diff:.6e} | Raw output: {actual_output}")

            results.append(
                {
                    "id": problem_id,
                    "category": source_category,
                    "difficulty": difficulty,
                    "prompt": prompt_latex,
                    "integrand": prompt_latex,
                    "variables_order": str(variables_order),
                    "expected_output": expected_output_str,
                    "symbolic_answer": symbolic_answer,
                    "numerical_answer": numerical_answer,
                    "actual_output": actual_output,
                    "actual_numeric": actual_numeric,
                    "correct": correct,
                    "difference": f"{abs_diff:.8e}",
                    "time_seconds": elapsed_time,
                    "error": "",
                }
            )

        except Exception as error:
            elapsed_time = time.perf_counter() - start_time
            print(f"-> ERROR: {error}")

            results.append(
                {
                    "id": problem_id,
                    "category": source_category,
                    "difficulty": difficulty,
                    "prompt": prompt_latex,
                    "integrand": prompt_latex,
                    "variables_order": str(variables_order),
                    "expected_output": expected_output_str,
                    "symbolic_answer": symbolic_answer,
                    "numerical_answer": numerical_answer,
                    "actual_output": "ERROR",
                    "actual_numeric": None,
                    "correct": False,
                    "difference": "",
                    "time_seconds": elapsed_time,
                    "error": str(error),
                }
            )

    return results


# ============================================================
# PODSUMOWANIE WYNIKÓW
# ============================================================

def print_summary(results):
    df = pd.DataFrame(results)

    if df.empty:
        print("Brak wyników do podsumowania.")
        return

    total = len(df)
    correct_count = int(df["correct"].sum())
    accuracy = correct_count / total * 100
    average_time = df["time_seconds"].mean()
    total_time = df["time_seconds"].sum()

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (INTEGRALBENCH - THINKING OFF)")
    print("=" * 60)
    print(f"Total tasks: {total}")
    print(f"Correct answers: {correct_count}")
    print(f"Incorrect answers: {total - correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average time: {average_time:.2f}s")
    print(f"Total time: {total_time / 60:.2f} min")

    if "difficulty" in df.columns and df["difficulty"].notna().any():
        print("\nRESULTS BY DIFFICULTY:")
        diff_summary = (
            df.groupby("difficulty")
            .agg(
                total=("id", "count"),
                correct=("correct", "sum"),
                average_time=("time_seconds", "mean"),
            )
        )
        diff_summary["accuracy_percent"] = (
            diff_summary["correct"] / diff_summary["total"] * 100
        )
        print(diff_summary.to_string())


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("OLLAMA INTEGRALBENCH EVALUATION (THINKING OFF)")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")

    dataset = load_dataset(dataset_path)
    print(f"Loaded tasks: {len(dataset)}")

    evaluation_results = evaluate_integrals_with_ollama(
        MODEL_NAME,
        dataset
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results_df = pd.DataFrame(evaluation_results)
    results_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print_summary(evaluation_results)

    print("\n" + "=" * 60)
    print("EVALUATION FINISHED")
    print("=" * 60)
    print(f"Results saved to:\n{output_path}")