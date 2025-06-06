from kan import *
import torch
import copy
import matplotlib.pyplot as plt
from collections import Counter

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# ===== Dataset =====
def create_dataset(train_num=500, test_num=500):
    def generate_contrastive(x):
        batch = x.shape[0]
        x[:,2] = torch.exp(torch.sin(torch.pi * x[:,0]) + x[:,1]**2)
        x[:,3] = x[:,4]**3

        def corrupt(tensor):
            y = copy.deepcopy(tensor)
            for i in range(y.shape[1]):
                y[:,i] = y[:,i][torch.randperm(y.shape[0])]
            return y

        x_cor = corrupt(x)
        x = torch.cat([x, x_cor], dim=0)
        y = torch.cat([torch.ones(batch,), torch.zeros(batch,)], dim=0)[:,None]
        return x, y

    x = torch.rand(train_num, 6) * 2 - 1
    x_train, y_train = generate_contrastive(x)

    x = torch.rand(test_num, 6) * 2 - 1
    x_test, y_test = generate_contrastive(x)

    return {
        'train_input': x_train.to(device),
        'test_input': x_test.to(device),
        'train_label': y_train.to(device),
        'test_label': y_test.to(device)
    }

# ===== Ground Truth Relations =====
true_sets = [
    {'x1', 'x2', 'x3'},
    {'x4', 'x5'},
    {'x6'}
]

# ===== Helpers =====
def check_match(predicted, true_sets):
    predicted_set = set(predicted)
    for true_set in true_sets:
        if true_set.issubset(predicted_set):
            return True
    return False

def normalize(feature_list):
    return tuple(sorted(feature_list))

# ===== Experiment =====
num_seeds = 100
results = []
found_relations = []

for seed in range(num_seeds):
    torch.manual_seed(seed)
    print(f"\nRunning seed {seed}...")

    model = KAN(width=[6,1,1], grid=3, k=3, seed=seed, device=device)

    dataset = create_dataset()

    model(dataset['train_input'])
    model.fix_symbolic(1,0,0,'gaussian', fit_params_bool=False)
    model(dataset['train_input'])

    model.fit(dataset, opt="LBFGS", steps=50, lamb=0.002, lamb_entropy=10.0,
              lamb_coef=1.0)
    model.plot(in_vars=[r'$x_{}$'.format(i) for i in range(1,7)])

    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
    model.auto_symbolic(lib=lib)

    scores = model.edge_scores[0]
    flat = scores.view(-1)
    threshold = (flat.mean() + flat.std()) * 0.2

    significant_indices = (flat > threshold).nonzero(as_tuple=True)[0]
    formatted = [f'x{i.item() + 1}' for i in significant_indices]

    print(f"Seed {seed} significant features: {formatted}")

    is_match = check_match(formatted, true_sets)
    results.append((formatted, is_match))
    found_relations.append(normalize(formatted))

# ===== Metrics =====
# Accuracy
correct = sum(1 for _, match in results if match)
accuracy = correct / num_seeds

# Coverage
covered_sets = set()
for formatted, _ in results:
    for i, true_set in enumerate(true_sets):
        if true_set.issubset(set(formatted)):
            covered_sets.add(i)
coverage = len(covered_sets) / len(true_sets)

# Per-relation accuracy and coverage
counts_per_set = [0] * len(true_sets)
for formatted, _ in results:
    formatted_set = set(formatted)
    for i, true_set in enumerate(true_sets):
        if true_set.issubset(formatted_set):
            counts_per_set[i] += 1

accuracy_per_relation = [count / num_seeds for count in counts_per_set]
coverage_per_relation = [1 if count > 0 else 0 for count in counts_per_set]

# ===== Print Summary =====
print(f"\nFinal results over {num_seeds} seeds:")
print(f"Overall Accuracy: {accuracy:.2f}")
print(f"Overall Coverage: {coverage:.2f}")

print("\nPer-relation metrics:")
for i, true_set in enumerate(true_sets):
    print(f"Relation {i+1} ({sorted(true_set)}): Accuracy = {accuracy_per_relation[i]:.2f}, Coverage = {coverage_per_relation[i]}")

# ===== Unique Feature Sets Found =====
relation_counts = Counter(found_relations)

print("\nDiscovered unique feature sets and their frequencies:")
for rel, freq in relation_counts.most_common():
    print(f"{rel} -> {freq} times")

# ===== Plots =====
# Accuracy per relation
relation_labels = [f"Relation {i+1}" for i in range(len(true_sets))]

plt.figure(figsize=(10, 4))
plt.bar(relation_labels, accuracy_per_relation, color='skyblue')
plt.ylim(0, 1.0)
plt.ylabel('Accuracy')
plt.title('Accuracy per Relation')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Coverage per relation
plt.figure(figsize=(10, 4))
plt.bar(relation_labels, coverage_per_relation, color='lightgreen')
plt.ylim(0, 1.0)
plt.ylabel('Coverage')
plt.title('Coverage per Relation')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Top-K most common discovered feature sets
top_k = 10
most_common = relation_counts.most_common(top_k)
labels = ['{' + ', '.join(r) + '}' for r, _ in most_common]
values = [v for _, v in most_common]

plt.figure(figsize=(12, 6))
plt.barh(labels[::-1], values[::-1], color='orchid')
plt.xlabel('Frequency')
plt.title(f'Top {top_k} Discovered Feature Sets Across {num_seeds} Seeds')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
