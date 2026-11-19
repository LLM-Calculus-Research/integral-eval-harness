import json
import os
import time
import ollama
import pandas as pd
import sympy as sp

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset = os.path.join(
    base_dir, "datasets", "simple_own_dataset", "multivariable_integrals_dataset.json"
)
dataset2 = os.path.join(
    base_dir, "datasets", "simple_own_dataset", "n_polynomail_dataset.json"
)
output = os.path.join(
    base_dir, "eval", "eval_own_dataset_llm_results", "evaluation_llm_results.csv"
)


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
        evaluation_start = time.perf_counter()
        prompt = row["prompt_latex"].replace(
            "Result should be expressed in latex format.", ""
        )
        variables_order = list(row["variables_order"])
        for variable in variables_order:
            local_dict[str(variable)] = sp.Symbol(str(variable))
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
                    "category": row["category"],
                    "prompt": prompt,
                    "expected_output": expected_output,
                    "actual_output": actual_output,
                    "correct": correct,
                    "time": time.perf_counter() - evaluation_start,
                }
            )
        except Exception as error:
            print(f"Error evaluating prompt {idx}: {error}")
            results.append(
                {
                    "id": row.get("id", idx),
                    "category": row["category"],
                    "prompt": prompt,
                    "expected_output": expected_output,
                    "actual_output": "ERROR",
                    "correct": False,
                    "time": time.perf_counter() - evaluation_start,
                }
            )

    return results


def write_evaluation_summary(results, summary_path):
    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise ValueError("Cannot write an evaluation summary without results.")

    total_tasks = len(results_df)
    correct_answers = int(results_df["correct"].sum())
    errors = int((results_df["actual_output"] == "ERROR").sum())
    accuracy = correct_answers / total_tasks * 100
    average_time = results_df["time"].mean()
    total_time_minutes = results_df["time"].sum() / 60

    category_summary = results_df.groupby("category").agg(
        total=("correct", "count"),
        correct=("correct", "sum"),
        average_time=("time", "mean"),
    )
    category_summary["accuracy_percent"] = (
        category_summary["correct"] / category_summary["total"] * 100
    )

    category_width = max(
        len("category"), *(len(str(name)) for name in category_summary.index)
    )
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write("=" * 60 + "\n")
        summary_file.write("EVALUATION SUMMARY\n")
        summary_file.write("=" * 60 + "\n")
        summary_file.write(f"Total tasks: {total_tasks}\n")
        summary_file.write(f"Correct answers: {correct_answers}\n")
        summary_file.write(f"Incorrect answers: {total_tasks - correct_answers}\n")
        summary_file.write(f"Errors: {errors}\n")
        summary_file.write(f"Accuracy: {accuracy:.2f}%\n")
        summary_file.write(f"Average time: {average_time:.2f}s\n")
        summary_file.write(f"Total time: {total_time_minutes:.2f} min\n\n")
        summary_file.write("RESULTS BY CATEGORY:\n")
        summary_file.write(
            f"{'category':<{category_width}}  {'total':>5}  {'correct':>7}  "
            f"{'average_time':>12}  {'accuracy_percent':>16}\n"
        )
        for category, row in category_summary.iterrows():
            summary_file.write(
                f"{category:<{category_width}}  {int(row['total']):>5}  "
                f"{int(row['correct']):>7}  {row['average_time']:>12.2f}  "
                f"{row['accuracy_percent']:>16.2f}\n"
            )
        summary_file.write("\n" + "=" * 60 + "\n")
        summary_file.write("EVALUATION FINISHED\n")
        summary_file.write("=" * 60 + "\n")


if __name__ == "__main__":
    dataset_df = pd.read_json(dataset2)
    evaluation_results = evaluate_integrals_with_ollama("gemma4", dataset_df)
    pd.DataFrame(evaluation_results).to_csv(output, index=False)
    summary_output = os.path.splitext(output)[0] + "_summary.txt"
    write_evaluation_summary(evaluation_results, summary_output)
    print("\nTabela wyników została pomyślnie zapisana do pliku CSV!")
    print(f"Podsumowanie ewaluacji zapisano do pliku TXT: {summary_output}")
