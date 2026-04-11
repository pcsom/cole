## Towards Code-Oriented LM Embeddings for Surrogate-Assisted Neural Architecture Search

### Introduction

Neural Architecture Search (NAS) automates the discovery of high-performing neural architectures for a given task in a search space, using methods such as evolutionary algorithms. However, conducting NAS in high-cost search spaces becomes computationally expensive due to complex evaluation functions. This motivates the area of performance predictors (surrogates) that substitute expensive ground-truth evaluations of architectures with quick prediction inferences. For this method, architectures must be converted into input representations compatible with surrogate models. Structural representations capture network topology sufficiently. However, capturing hyperparameters is computationally inefficient. Furthermore, these encoding schemes are inherently rigid, and must be re-engineered when applying the predictor to novel architecture spaces. Recent efforts to make generalized encodings proposed using Language Models (LMs) as universal regressors. While effective, this entails the high computational cost of fine-tuning the LMs. Instead, it would be beneficial to identify promising regions of the search space out-of-the-box, enabling cold-start searches.

A key insight driving our work is that publicly available language models can efficiently represent code. Language models like CodeLlama and ModernBERT have been pre-trained on trillions of tokens, many of which are code. We investigate the efficacy of using frozen LMs as embedding engines for NAS. We feed raw PyTorch class definition text directly into an LM (without fine-tuning) and use the model’s hidden state to compute a Code-Oriented LM Embedding (COLE). We show that COLE is superior to other text-based encodings with frozen LMs and improves performance of an existing surrogate-assisted NAS algorithm. Code can represent any neural architecture, so our work can be incorporated into most surrogate-enabled NAS pipelines.

______

### Setup

Please install and configure Anaconda 3.

#### 1. Primary environment setup
- Execute: `conda env create -f environment.yml` to create the conda environment `cole`

#### 2. NASLib setup
- For compatibility between existing NASLib functionality and our modification, we followed a different setup process than that provided by NASLib's documentation.
- Navigate into the `NASLib` folder, then execute the commands:
  - `conda create -n cole_naslib python=3.9`
  - `conda activate cole_naslib`
  - `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`
    - Adjust your cuda version dependency as necessary
  - `pip install transformers accelerate bitsandbytes scipy ConfigSpace pyyaml networkx numpy fvcore pytest lightgbm ngboost xgboost emcee pybnn grakel pyro-ppl scipy tqdm scikit-learn scikit-image pytorch-msssim tensorwatch transforms3d gdown`
  - `pip install --upgrade git+https://github.com/romulus0914/NASBench-PyTorch@master`
  - `pip install git+https://github.com/automl/nasbench301@no_gin`
  - `pip install --force-reinstall --no-cache-dir "numpy<2.0" "ConfigSpace==0.4.21" scipy scikit-learn pandas tornado seaborn`
- If you face errors when executing the `pip install --upgrade ...` command, you may need to rerun the `pip install torch ...` command and try again.
- Follow the directions present in NASLib's `README.md` to install NAS-Bench-201 files (find the “Queryable Benchmarks” table present in the README)

#### 3. Main corpus setup
- Download the `NAS-Bench-201-v1_1-096897.pth` file from the [NAS-Bench-201 repository](https://github.com/D-X-Y/NAS-Bench-201). Set the `NASBENCH_201_API_PATH` variable in `embedding_config.py` to the path to this downloaded `NAS-Bench-201-v1_1-096897.pth` file.
- Use `generate_corpus.py` to create a corpus consisting of all NAS-Bench-201 architectures and corresponding PyTorch code representations. Ensure this generated .pkl file is used as an input argument to the first 4 experiments (in the section Running Experiments below).

#### 3. ONNX setup
- Use the [ONNX-Net codebase](https://github.com/shiwenqin/ONNX-Net) to generate a CSV file containing ONNX encodings for NAS-Bench-201 architectures.

______

### Running Experiments

For the first 4 experiments, use the `cole` conda environment.

#### 1. Comparing COLE configurations

Use `paper_ablation_comp.py` to compare different configurations of COLE.

#### 2. Comparing LMs

Use `paper_llm_comp.py` to compare performance of various LMs using COLE.

#### 3. COLE vs. ONNX-to-text encodings

Use `onnx_comp.py`. Ensure that as an argument you provide the path to the ONNX-Net CSV generated at Setup step 3.

#### 4. COLE vs. Derivation tree strings

Use `einspace_comp.py`. Ensure that as an argument you provide `einspace_corpus.pkl`. 

#### 5. NASLib: Downstream NAS performance using COLE vs. Path encodings

Use the `cole_naslib` conda environment, and navigate into the `NASLib` folder. Use `run_nb201_comparison.py` to execute the experiment. Use `results_paper.py` and `analyze_surrogate_results.py` to create visualizations and metrics comparing the NAS trials.

______

### Additional Documentation

The `docs.md` files in the project root and inside the `NASLib` folder provide information about key scripts.

