import json
from pathlib import Path

OUTPUT_DIR = Path("output")
GROUND_TRUTH_DIR = Path("ground_truth")

def compare(output: dict, truth: dict) -> tuple[int, int, list[str]]:
    correct = 0
    wrong_fields = []
    for field, expected in truth.items():
        actual = output.get(field, "").strip()
        if actual == expected.strip():
            correct += 1
        else:
            wrong_fields.append(field)
    return correct, len(truth), wrong_fields


def main():
    truth_files = sorted(GROUND_TRUTH_DIR.glob("*.json"))
    if not truth_files:
        print("No files found in ground_truth/")
        return

    total_correct = total_fields = 0
    files_passed = 0

    print(f"{'File':<22} {'Correct/Total':<16} {'Accuracy':<24} Wrong fields")
    print("-" * 90)

    for truth_path in truth_files:
        output_path = OUTPUT_DIR / truth_path.name
        if not output_path.exists():
            print(f"{truth_path.stem:<22} Not found in output/")
            continue

        with open(truth_path, encoding="utf-8") as f:
            truth = json.load(f)
        with open(output_path, encoding="utf-8") as f:
            output = json.load(f)

        correct, total, wrong = compare(output, truth)
        acc = correct / total * 100
        total_correct += correct
        total_fields += total

        if correct == total:
            files_passed += 1

        wrong_str = ", ".join(wrong) if wrong else "-"
        print(f"{truth_path.stem:<22} {correct}/{total:<14} {acc:5.1f}%{'':<20} {wrong_str}")

    if total_fields == 0:
        return

    total_files = len(truth_files)
    acc_per_field = total_correct / total_fields * 100
    acc_per_files = files_passed / total_files * 100

    print("-" * 90)
    print(f"Accuracy per field  = {acc_per_field:.2f} %")
    print(f"Accuracy per files  = {acc_per_files:.0f} %")


if __name__ == "__main__":
    main()
