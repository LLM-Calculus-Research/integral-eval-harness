import json

import ollama
import pandas as pd
import sympy as sp

dataset = pd.read_json("multivariable_integrals_dataset.json")
x, y, z = sp.symbols("x y z")


def evaluate_integrals_with_ollama(model_name, dataset):
    results = []
    local_dict = {
        "x": x,
        "y": y,
        "z": z,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "E": sp.E,
    }

    for idx, row in dataset.iterrows():
        prompt = row["prompt_latex"].replace(
            "Result should be expressed in latex format.", ""
        )
        variables_order = row["variables_order"]
        expected_output = row["ground_truth_latex"]
        system_prompt = """Calculate the integral and return only JSON:
{"result_python": "your answer"}
Use Python syntax (** for powers, * for multiplication) and functions such as
sin(x), cos(x), exp(x). Omit the constant of integration."""

        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"{system_prompt}\nVariables order: {variables_order}\n"
                            f"Task: {prompt}"
                        ),
                    }
                ],
                format="json",
                options={"temperature": 0.0},
            )
            actual_output = json.loads(response["message"]["content"])[
                "result_python"
            ].strip()

            result = sp.sympify(actual_output, locals=local_dict)
            integrand = sp.sympify(row["expression_str"], locals=local_dict)
            for variable in reversed(variables_order):
                result = sp.diff(result, local_dict[variable])
            difference = sp.simplify(result - integrand)
            correct = difference == 0

            print(
                f"-> Prompt {idx}: "
                f"{'Correct' if correct else f'Incorrect (difference: {difference})'}"
            )
            results.append(
                {
                    "id": row.get("id", idx),
                    "prompt": prompt,
                    "expected_output": expected_output,
                    "actual_output": actual_output,
                    "correct": correct,
                }
            )
        except Exception as error:
            print(f"Error evaluating prompt {idx}: {error}")
            results.append(
                {
                    "id": row.get("id", idx),
                    "prompt": prompt,
                    "expected_output": expected_output,
                    "actual_output": "ERROR",
                    "correct": False,
                }
            )

    return results


if __name__ == "__main__":
    evaluation_results = evaluate_integrals_with_ollama("gemma4", dataset)
    pd.DataFrame(evaluation_results).to_csv("evaluation_llm_results.csv", index=False)
    print("\nTabela wyników została pomyślnie zapisana do pliku CSV!")
