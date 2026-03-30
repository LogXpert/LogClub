# LogClub

LogClub: Accuracy–Efficiency Optimization for LLM-Based Log Parsing via Coarse-to-Fine Clustering and Batched Templating


## Directory Structure

- `code/`: Core modules for masking, profiling, and template mining
- `evaluator/`: Evaluation scripts, configuration, and datasets

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/LogXpert/LogClub.git

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run dataset evaluation:

   ```bash
   # logHub-2K Corrected Datasets
   python evaluator/evaluator.py --dataset=2k --type=corrected --method LogClub

   # logHub-2.0 Datasets
   python evaluator/evaluator.py --dataset=full --method LogClub

   # logHub-2.0 sample Datasets
   python evaluator/evaluator.py --dataset=full50K --method LogClub   
   ```


## Result

Result can be found in result.log under running directory

## Usage

- Configure log parsing and template mining using `.ini` files in `evaluator/`.
- Use provided datasets for benchmarking and evaluation.


## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements and new features.

## License

This project is open-source. See the LICENSE file for details.
