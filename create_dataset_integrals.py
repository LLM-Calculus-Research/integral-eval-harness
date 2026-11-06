import json
import random
import sympy as sp

x, y, z = sp.symbols("x y z")


def gen_polynomial():
    """Simple polynomial generator for testing purposes."""

    a, b, c = random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)
    f = a * x**2 + b * x + c
    return f, [x], "Polynomial"


def gen_polynomial_2d():
    """Simple polynomial 2D generator for testing purposes."""

    f = random.randint(1, 10) * x ** random.randint(1, 3) + random.randint(
        1, 10
    ) * y ** random.randint(1, 3)
    return f, [x, y], "Polynomial_2D"


def gen_mixed_2d():
    """Integral mixed 2D: polynomial + trigonometric/exp function"""
    f = x * random.choice([sp.sin(y), sp.cos(y)]) + sp.exp(x) * y
    return f, [y, x], "Mixed_2D"


def gen_polynomial_3d():
    """Simple polynomial 3D generator for testing purposes."""

    f = (
        random.randint(1, 10) * x ** random.randint(1, 3)
        + random.randint(1, 10) * y ** random.randint(1, 3)
        + random.randint(1, 10) * z ** random.randint(1, 3)
    )
    return f, [z, y, x], "Polynomial_3D"


def generate_multivariable_dataset(count_per_type=5):
    dataset = []
    generators = [gen_polynomial, gen_polynomial_2d, gen_mixed_2d, gen_polynomial_3d]

    idx = 1

    for gen in generators:
        for _ in range(count_per_type):
            f, vars_order, category = gen()

            solutions = sp.integrate(f, *vars_order)

            diff_elements = "".join([f"d{v}" for v in vars_order])

            integral_symbols = "\\int" * len(vars_order)
            prompt_latex = f"{integral_symbols} {sp.latex(f)} {diff_elements}"

            dataset.append(
                {
                    "id": idx,
                    "category": category,
                    "variables_order": [str(v) for v in vars_order],
                    "expression_str": str(f),
                    "prompt_latex": f"Calculate the integral of the following expression: {prompt_latex}. Result should be expressed in latex format.",
                    "ground_truth_latex": sp.latex(solutions),
                }
            )
            idx += 1
    return dataset


if __name__ == "__main__":

    multi_dataset = generate_multivariable_dataset(count_per_type=50)
    with open("multivariable_integrals_dataset.json", "w") as f:
        json.dump(multi_dataset, f, indent=4, ensure_ascii=False)

    print("Dataset generated and saved to multivariable_integrals_dataset.json")
