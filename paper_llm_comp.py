"""
NASBench-201 robust comparison with CodeLlama and ModernBERT embeddings.
Safe to run multiple times - will only compute what's missing
"""

from robust_surrogate_predict import run_comparison
from embed_corpus import add_embeddings_to_corpus
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Compare two LLM embeddings for NAS')
parser.add_argument('--model1', type=str, default='codellama_python_7b', help='First model name')
parser.add_argument('--model2', type=str, default='modernbert_large', help='Second model name')
parser.add_argument('--corpus_path', type=str, required=True, help='Path to the input corpus.')
parser.add_argument('--output_corpus_path', type=str, required=True, help='Path to save the output corpus.')
parser.add_argument('--output_path', type=str, required=True, help='Path to save the comparison results CSV.')
parser.add_argument('--output_path_csv', type=str, required=True, help='Path to output directory for per-embedding CSVs.')
args = parser.parse_args()

model1 = args.model1
model2 = args.model2

# Paths
COMPARISON_LABEL = f'{model1}_VS_{model2}'

NASBENCH_CORPUS_PATH = args.corpus_path
NASBENCH_CORPUS_OUTPUT_PATH = args.output_corpus_path
OUTPUT_PATH = args.output_path
OUTPUT_PATH_CSV = args.output_path_csv

os.makedirs(os.path.dirname(NASBENCH_CORPUS_OUTPUT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
os.makedirs(OUTPUT_PATH_CSV, exist_ok=True)

# Configuration
# SAMPLE_SIZES = [15, 50, 150, 500, 1500, 5000]  # Training set sizes to test
SAMPLE_SIZES = [14, 55, 220, 879, 3516, 14062]  # Training set sizes to test
# SAMPLE_SIZES = [8, 39, 78, 224, 896, 3584, 14062]  # Training set sizes to test
N_FOLDS = 10       # 10-fold cross-validation
N_REPEATS = 20    # If Repeat CV 50 times, then 500 total trials per sample size
FORCE = False     # If True, recompute everything; if False, only add missing data

df = add_embeddings_to_corpus(
    corpus_path=NASBENCH_CORPUS_PATH,
    model_name=model1,
    output_path=NASBENCH_CORPUS_OUTPUT_PATH,
    pytorch_only=True,
    use_echo_embeddings=False,
    device='cuda',
    max_length=512,
    pooling_mode='avg_avg'
)
df = add_embeddings_to_corpus(
    corpus_path=NASBENCH_CORPUS_PATH,
    model_name=model2,
    output_path=NASBENCH_CORPUS_OUTPUT_PATH,
    pytorch_only=True,
    use_echo_embeddings=False,
    device='cuda',
    max_length=512,
    pooling_mode='avg_avg'
)



# Run robust comparison (only computes missing trials)
print("\n" + "=" * 80)
print("NASBench-201 Robust Comparison:")
print(f"  Sample sizes: {SAMPLE_SIZES}")
print(f"  CV setup: {N_FOLDS}-fold × {N_REPEATS} repeats = {N_FOLDS * N_REPEATS} trials per size")
print("  (Will only compute missing trials, preserves existing results)")
print("=" * 80)

results_df = run_comparison(
    embedding1_name=f'{model1}_pytorch_code_exclude_helper_avg_avg_embedding',
    corpus1_name='core',
    embedding2_name=f'{model2}_pytorch_code_exclude_helper_avg_avg_embedding',
    corpus2_name='core',
    corpus_path1=NASBENCH_CORPUS_OUTPUT_PATH,
    corpus_path2=NASBENCH_CORPUS_OUTPUT_PATH,
    comparison_label=COMPARISON_LABEL,
    sample_sizes=SAMPLE_SIZES,
    n_folds=N_FOLDS,
    n_repeats=N_REPEATS,
    benchmark_type='nasbench',
    comparison_output_path=OUTPUT_PATH,
    per_embedding_output_dir=OUTPUT_PATH_CSV,
    device='cuda',
    force=FORCE,
    dim_reduction_method_embedding1='softpca',
    dim_reduction_components_embedding1=128,
    pca_whitening_epsilon_embedding1=None,
    dim_reduction_method_embedding2='softpca',
    dim_reduction_components_embedding2=128,
    pca_whitening_epsilon_embedding2=None,
    use_pairwise_loss_embedding1=True,
    use_pairwise_loss_embedding2=True,
    use_single_target_embedding1=True,
    use_single_target_embedding2=True,
    head_type_embedding1='mlp',
    head_type_embedding2='mlp'
)


print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)
if len(results_df) > 0:
    print(results_df[['sample_size', 'model1', 'model2', 'model1_mean_ktau', 'model2_mean_ktau', 'mean_diff_ktau', 'p_value_ktau', 'significant_ktau', 'n_trials']])
    print(f"\nTotal result rows: {len(results_df)}")
    print(f"Results saved to {OUTPUT_PATH}")
else:
    print("No new results computed - all comparisons already complete!")
print("=" * 80)
