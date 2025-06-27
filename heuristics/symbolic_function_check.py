correct = 0
total = 0
matches = []
mismatches = []

for key, true_entry in ground_truth.items():
    predicted_entry = predicted.get(key, (None,))
    predicted_name = predicted_entry[0]

    # Skip if predicted was 'already_symbolic'
    if predicted_name == 'already_symbolic':
        continue

    true_name = true_entry[0] if isinstance(true_entry, tuple) else true_entry

    if predicted_name == true_name:
        correct += 1
        matches.append((key, predicted_name))
    else:
        mismatches.append((key, predicted_name, true_name))

    total += 1

accuracy = correct / total if total > 0 else 0.0

print(f"Symbolic match accuracy (excluding already symbolic): {accuracy:.2%}")

if matches:
    print("\nMatches:")
    for key, name in matches:
        print(f"  {key}: matched '{name}'")

if mismatches:
    print("\nMismatches:")
    for key, pred, true in mismatches:
        print(f"  {key}: predicted '{pred}', expected '{true}'")