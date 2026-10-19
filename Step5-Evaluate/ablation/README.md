# Ablation Study - Core API Identification

This directory contains scripts for the ablation experiment that removes the core API identification component from the original approach.

## Experiment Overview

The ablation study evaluates the impact of removing the LLM-based core API identification step (Step 1). Instead of using a single identified core API as the query, this experiment uses **all APIs explicitly mentioned in each issue** as replacement query APIs.

## Workflow

```
Step 1: Identify replacement APIs
    ↓
Step 2: Generate similarity candidates from Step2 data
    ↓
Step 3: Build multi-API replacement sets
    ↓
Step 4: Generate prompts with replacement requirements
    ↓
Step 5: Execute prompts with LLM
    ↓
Step 6: Parse responses, extract test cases, execute, and categorize
```

## Scripts

### 1. `1_no_coreapi_identification.py`

**Purpose**: Generate similarity candidates using all replacement APIs instead of the identified core API.

**Input**:
- Step1 issue data (`Step1-CoreAPIIdentification/data/pytorch_core_api_identification`)
- Step2 similarity files (`Step2-CalculateSimilarity/data`)

**Output**:
- `data/no_coreapi_identification/similar_api_list/{issue_id}.json`

**Key Features**:
- Collects all API-like tokens from issue metadata (title, description, repro code)
- Looks up each replacement API in Step2's pre-computed similarity indexes
- Generates per-issue similarity lists compatible with Step3 format
- Supports both merged mode (one file per issue) and split mode (one file per replacement API)

**Usage**:
```bash
python Step5-Evaluate/ablation/1_no_coreapi_identification.py

# With custom parameters
python Step5-Evaluate/ablation/1_no_coreapi_identification.py \
    --step1-dir /path/to/step1/data \
    --similarity-dir /path/to/step2/data \
    --output-dir /path/to/output \
    --top-n 11 \
    --limit 100
```

### 2. `2_generate_prompt_all_replacements.py`

**Purpose**: Generate prompts that instruct the LLM to replace multiple source APIs together.

**Input**:
- Step1 issue data
- Step3 base prompts (`Step3-ReuseTestCase/prompts`)
- ablation similarity candidates (`data/no_coreapi_identification/similar_api_list`)

**Output**:
- `data/no_coreapi_identification/prompts_all_replacements/{issue_id}__replacement_set__{rank}.txt`

**Key Features**:
- Reads existing base prompts from Step3
- Builds replacement sets by selecting top-k candidates for each source API
- Appends multi-API replacement instructions to each prompt
- Each issue generates up to `--top-n` replacement set prompts

**Usage**:
```bash
python Step5-Evaluate/ablation/2_generate_prompt_all_replacements.py

# With custom parameters
python Step5-Evaluate/ablation/2_generate_prompt_all_replacements.py \
    --step1-dir /path/to/step1/data \
    --base-prompt-dir /path/to/step3/prompts \
    --similarity-dir /path/to/rq3/similar_api_list \
    --output-dir /path/to/output \
    --top-n 10 \
    --limit 100
```

### 3. `3_execute_prompt_all_replacements.py`

**Purpose**: Execute generated prompts using Zhipu AI LLM and save responses.

**Input**:
- Prompt files from `2_generate_prompt_all_replacements.py`

**Output**:
- `data/no_coreapi_identification/llm_responses_all_replacements/llm_io/`
  - `{prompt_name}.input.txt` - Original prompt
  - `{prompt_name}.output.txt` - LLM response
  - `{prompt_name}.output.events.json` - Raw streaming events

**Key Features**:
- Uses Zhipu AI `glm-4.7` model by default
- Supports streaming responses
- Saves both clean responses and raw event data
- Skips existing responses by default (use `--overwrite` to re-run)

**Usage**:
```bash
# Set API key (or use ZHIPUAI_API_KEY environment variable)
python Step5-Evaluate/ablation/3_execute_prompt_all_replacements.py \
    --api-key your_api_key

# With custom parameters
python Step5-Evaluate/ablation/3_execute_prompt_all_replacements.py \
    --prompt-dir /path/to/prompts \
    --output-dir /path/to/output \
    --api-key your_api_key \
    --model glm-4.7 \
    --temperature 0.0 \
    --max-tokens 8192 \
    --limit 100 \
    --overwrite
```

### 4. `4_parse_and_execute.py`

**Purpose**: Parse LLM responses, extract test cases, execute them, and categorize results.

This script combines functionality from:
- `Step3-ReuseTestCase/3-parse-llm-results.py` (code extraction)
- `Step4-RunTestCase/1-run-test-case.py` (test execution)
- `Step4-RunTestCase/2-category.py` (result classification)

**Input**:
- LLM response files from `3_execute_prompt_all_replacements.py`

**Output**:
- `data/no_coreapi_identification/test_cases/` - Extracted test case .py files
- `data/no_coreapi_identification/test_results/` - Execution logs (.log files)
- `data/no_coreapi_identification/categorized_results.csv` - Categorized results

**Key Features**:
- Extracts Python code from LLM responses (handles ```python blocks and def test_ functions)
- Repairs incomplete assertions and missing imports
- Executes test cases in subprocess with timeout
- Classifies failures into categories (Success, Assertion Mismatch, API Misuse, etc.)
- Supports skipping extraction or execution steps

**Usage**:
```bash
# Full pipeline: extract, execute, categorize
python Step5-Evaluate/ablation/4_parse_and_execute.py

# With custom parameters
python Step5-Evaluate/ablation/4_parse_and_execute.py \
    --response-dir /path/to/llm_io \
    --test-case-dir /path/to/test_cases \
    --output-dir /path/to/test_results \
    --limit 100

# Skip extraction (test cases already exist)
python Step5-Evaluate/ablation/4_parse_and_execute.py \
    --skip-extract

# Skip execution (results already exist, just categorize)
python Step5-Evaluate/ablation/4_parse_and_execute.py \
    --skip-extract \
    --skip-execute
```

## Directory Structure

```
ablation/
├── 1_no_coreapi_identification.py
├── 2_generate_prompt_all_replacements.py
├── 3_execute_prompt_all_replacements.py
├── 4_parse_and_execute.py
├── README.md
└── data/
    └── no_coreapi_identification/
        ├── similar_api_list/           # Output from script 1
        │   ├── 159974.json
        │   └── ...
        ├── prompts_all_replacements/   # Output from script 2
        │   ├── 159974__replacement_set__1.txt
        │   └── ...
        ├── llm_responses_all_replacements/  # Output from script 3
        │   └── llm_io/
        │       ├── *.input.txt
        │       ├── *.output.txt
        │       └── *.output.events.json
        ├── test_cases/                 # Output from script 4 (extracted)
        │   └── *.py
        ├── test_results/               # Output from script 4 (execution)
        │   └── *.log
        └── categorized_results.csv     # Output from script 4 (categorization)
```

## Execution Order

1. Run `1_no_coreapi_identification.py` to generate similarity candidates
2. Run `2_generate_prompt_all_replacements.py` to generate prompts
3. Run `3_execute_prompt_all_replacements.py` to execute prompts
4. Run `4_parse_and_execute.py` to parse responses, extract test cases, execute them, and categorize results

## Requirements

- Python 3.8+
- `zai` package for Zhipu AI API (`pip install zai`)
- Zhipu AI API key (set as environment variable `ZHIPUAI_API_KEY` or via `--api-key` argument)

## Configuration

Default paths are relative to the repository root. You can override them using command-line arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--step1-dir` | Step1 issue data directory | `Step1-CoreAPIIdentification/data/pytorch_core_api_identification` |
| `--similarity-dir` | Step2 similarity data directory | `Step2-CalculateSimilarity/data` |
| `--base-prompt-dir` | Step3 base prompts directory | `Step3-ReuseTestCase/prompts` |
| `--output-dir` | Output directory | Script-specific default |
| `--top-n` | Number of top similar APIs to consider | 11 (script 1), 10 (script 2) |
| `--limit` | Process only first N items (0 = all) | 0 |
