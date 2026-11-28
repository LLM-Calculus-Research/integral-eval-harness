
import json
import os
import re
import time

import ollama
import pandas as pd
import sympy as sp


# ============================================================
# KONFIGURACJA
# ============================================================

base_dir = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(
    base_dir,
    "datasets",
    "simple_own_dataset",
    "multivariable_integrals_dataset.json"
)

output_path = os.path.join(
    base_dir,
    "eval",
    "eval_own_dataset_llm_results",
    "evaluation_llm_results.csv"
)

MODEL_NAME = "exaone-deep:7.8b"

x, y, z = sp.symbols("x y z")

local_dict = {
    "x": x,
    "y": y,
    "z": z,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "E": sp.E,
    "pi": sp.pi,
}


# ============================================================
# WCZYTYWANIE ZBIORU DANYCH
# ============================================================

def load_dataset(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Nie znaleziono zbioru danych: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return pd.DataFrame(data)


# ============================================================
# NORMALIZACJA KOLEJNOŚCI ZMIENNYCH
# ============================================================

def normalize_variables_order(value):

    if isinstance(value, list):
        return value

    if isinstance(value, str):

        try:
            parsed = json.loads(value)

            if isinstance(parsed, list):
                return parsed

        except json.JSONDecodeError:
            pass

        return [
            variable.strip()
            for variable in value.split(",")
            if variable.strip()
        ]

    raise ValueError(
        f"Nieprawidłowy format variables_order: {value}"
    )


# ============================================================
# WYODRĘBNIANIE JSON Z ODPOWIEDZI EXAONE
# ============================================================

def extract_result_json(response_content):

    if not response_content or not response_content.strip():
        raise ValueError(
            "Model zwrócił pustą odpowiedź."
        )

    decoder = json.JSONDecoder()

    # Przeszukujemy odpowiedź w poszukiwaniu obiektu JSON.
    # Dzięki temu możemy pominąć rozumowanie w <thought>,
    # które EXAONE może umieszczać przed końcową odpowiedzią.

    for match in re.finditer(r"\{", response_content):

        try:
            parsed, end_index = decoder.raw_decode(
                response_content[match.start():]
            )

            if (
                isinstance(parsed, dict)
                and "result_python" in parsed
            ):

                result = parsed["result_python"]

                if not isinstance(result, str):
                    raise ValueError(
                        "Pole result_python nie jest tekstem."
                    )

                result = result.strip()

                if not result:
                    raise ValueError(
                        "Pole result_python jest puste."
                    )

                return result

        except json.JSONDecodeError:
            continue

    raise ValueError(
        "Nie znaleziono poprawnego JSON zawierającego "
        "'result_python'. Odpowiedź modelu:\n"
        + response_content
    )


# ============================================================
# SPRAWDZANIE POPRAWNOŚCI MATEMATYCZNEJ
# ============================================================

def check_integral(actual_output, integrand, variables_order):

    result = sp.sympify(
        actual_output,
        locals=local_dict
    )

    original_integrand = sp.sympify(
        integrand,
        locals=local_dict
    )

    # Różniczkujemy w odwrotnej kolejności
    # niż kolejność całkowania.

    for variable in reversed(variables_order):

        if variable not in local_dict:
            raise ValueError(
                f"Nieznana zmienna: {variable}"
            )

        result = sp.diff(
            result,
            local_dict[variable]
        )

    difference = sp.simplify(
        sp.trigsimp(result - original_integrand)
    )

    return difference == 0, difference


# ============================================================
# OBLICZANIE CAŁEK PRZEZ OLLAMA
# ============================================================

def evaluate_integrals_with_ollama(model_name, dataset):

    results = []

    for idx, row in dataset.iterrows():

        prompt = row.get("prompt_latex", "")

        if isinstance(prompt, str):
            prompt = prompt.replace(
                "Result should be expressed in latex format.",
                ""
            )

        variables_order = normalize_variables_order(
            row["variables_order"]
        )

        expected_output = row.get(
            "ground_truth_latex",
            ""
        )

        integrand = row["expression_str"]

        user_prompt = f"""
You are an expert mathematical assistant specializing
in symbolic integration.

Calculate the requested iterated indefinite integral.

Integration variables, in order from innermost
to outermost integral:

{variables_order}

Original integrand:

{integrand}

Task description:

{prompt}

Instructions:

1. Calculate the iterated indefinite integral.
2. Follow the specified integration order exactly.
3. Return a complete symbolic expression.
4. Omit arbitrary constants of integration.
5. Use valid Python/SymPy syntax:
   - ** for powers
   - * for multiplication
   - sin(x), cos(x), tan(x) for trigonometric functions
   - exp(x) for exponential functions
   - log(x) for logarithms
   - sqrt(x) for square roots
6. Do not include explanations in the final answer.
7. Do not use LaTeX notation in the result.
8. Return the final answer as a JSON object.

The JSON must have exactly this structure:

{{
    "result_python": "your answer"
}}

The final JSON object must contain the complete
symbolic antiderivative.
"""

        print(
            f"\n[{idx + 1}/{len(dataset)}] "
            f"Prompt ID: {row.get('id', idx)}"
        )

        start_time = time.perf_counter()

        try:

            # EXAONE: bez wymuszania format="json".
            # Model sam generuje końcowy obiekt JSON.

            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                options={
                    "temperature": 0.0
                }
            )

            elapsed_time = (
                time.perf_counter() - start_time
            )

            response_content = response.message.content

            actual_output = extract_result_json(
                response_content
            )

            correct, difference = check_integral(
                actual_output,
                integrand,
                variables_order
            )

            status = (
                "Correct"
                if correct
                else "Incorrect"
            )

            print(
                f"-> {status} | "
                f"Time: {elapsed_time:.2f}s"
            )

            if not correct:
                print(
                    f"   Difference: {difference}"
                )

            results.append(
                {
                    "id": row.get("id", idx),
                    "category": row.get("category", ""),
                    "prompt": prompt,
                    "integrand": integrand,
                    "variables_order": str(variables_order),
                    "expected_output": expected_output,
                    "actual_output": actual_output,
                    "correct": correct,
                    "difference": str(difference),
                    "time_seconds": elapsed_time,
                    "error": ""
                }
            )

        except Exception as error:

            elapsed_time = (
                time.perf_counter() - start_time
            )

            print(
                f"-> ERROR: {error}"
            )

            results.append(
                {
                    "id": row.get("id", idx),
                    "category": row.get("category", ""),
                    "prompt": prompt,
                    "integrand": integrand,
                    "variables_order": str(variables_order),
                    "expected_output": expected_output,
                    "actual_output": "ERROR",
                    "correct": False,
                    "difference": "",
                    "time_seconds": elapsed_time,
                    "error": str(error)
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

    correct_count = int(
        df["correct"].sum()
    )

    accuracy = (
        correct_count / total * 100
    )

    average_time = df["time_seconds"].mean()

    total_time = df["time_seconds"].sum()

    error_count = int(
        (df["actual_output"] == "ERROR").sum()
    )

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"Total tasks: {total}")
    print(f"Correct answers: {correct_count}")
    print(f"Incorrect answers: {total - correct_count}")
    print(f"Errors: {error_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average time: {average_time:.2f}s")
    print(f"Total time: {total_time / 60:.2f} min")

    if "category" in df.columns:

        print("\nRESULTS BY CATEGORY:")

        summary = (
            df.groupby("category")
            .agg(
                total=("id", "count"),
                correct=("correct", "sum"),
                average_time=("time_seconds", "mean")
            )
        )

        summary["accuracy_percent"] = (
            summary["correct"] /
            summary["total"] * 100
        )

        print(summary.to_string())


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("OLLAMA INTEGRAL EVALUATION")
    print("=" * 60)

    print(f"Model: {MODEL_NAME}")

    dataset = load_dataset(dataset_path)

    print(f"Loaded tasks: {len(dataset)}")

    evaluation_results = evaluate_integrals_with_ollama(
        MODEL_NAME,
        dataset
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    results_df = pd.DataFrame(
        evaluation_results
    )

    results_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print_summary(evaluation_results)

    print("\n" + "=" * 60)
    print("EVALUATION FINISHED")
    print("=" * 60)

    print(
        f"Results saved to:\n{output_path}"
    )
