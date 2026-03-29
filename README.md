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

**logHub-2.0:**

|    | Dataset     |     GA |     PA |    FGA |    FTA |   Dura |   #Line |    #Tpl |   #Invoke |   #Tokens |   Match |   Group |   Extract |   Correct |
|----|-------------|--------|--------|--------|--------|--------|---------|---------|-----------|-----------|---------|---------|-----------|-----------|
|  0 | Proxifier   | 1      | 1      | 1      | 1      |  8.32  |   21320 |   8     |    5      |   1145    |  4.11   |  1.84   |    2.31   |    0.01   |
|  1 | Apache      | 0.9931 | 0.9908 | 0.8437 | 0.7187 | 11.63  |   49999 |  35     |    1      |   1106    |  8.21   |  1.92   |    1.39   |    0.01   |
|  2 | OpenSSH     | 1      | 0.9238 | 1      | 0.6786 | 15.83  |   49997 |  28     |    2      |      0    | 12.67   |  3.01   |    0.04   |    0.01   |
|  3 | HDFS        | 1      | 1      | 1      | 0.9444 |  9.39  |   49997 |  18     |    0      |      0    |  7.49   |  1.81   |    0.01   |    0      |
|  4 | OpenStack   | 1      | 0.9874 | 1      | 0.8889 | 15.1   |   50001 |  45     |    1      |   1237    | 10.54   |  1.92   |    2.52   |    0.01   |
|  5 | HPC         | 0.9798 | 0.9908 | 0.7571 | 0.7286 | 11.99  |   49997 |  78     |   11      |   1243    |  5.18   |  4.05   |    2.61   |    0.05   |
|  6 | Zookeeper   | 0.9837 | 0.9337 | 0.7831 | 0.7229 | 10.32  |   49990 |  77     |    3      |      0    |  7.38   |  2.8    |    0.02   |    0.02   |
|  7 | HealthApp   | 0.9978 | 0.9591 | 0.9592 | 0.8027 | 23.14  |   49990 | 146     |    3      |   4070    |  6      |  2.09   |   14.93   |    0.01   |
|  8 | Hadoop      | 0.9507 | 0.8153 | 0.9201 | 0.7128 | 16.13  |   49992 | 239     |    1      |   1484    |  8.68   |  2.94   |    4.36   |    0.01   |
|  9 | Spark       | 0.9975 | 0.8316 | 0.9583 | 0.7416 | 14.3   |   49983 | 122     |    1      |   1125    |  9.66   |  2.11   |    2.41   |    0.01   |
| 10 | BGL         | 0.9909 | 0.9606 | 0.9097 | 0.8014 | 41.46  |   50026 | 290     |   10      |   7205    |  8.04   |  4.25   |   26.61   |    2.4    |
| 11 | Linux       | 0.9931 | 0.8792 | 0.9433 | 0.7424 |  7.33  |   23921 | 349     |    7      |      0    |  4.81   |  2.3    |    0.03   |    0.1    |
| 12 | Mac         | 0.9366 | 0.6129 | 0.8623 | 0.5009 | 47.73  |   49862 | 621     |   14      |   5118    | 13.77   |  4.7    |   28.64   |    0.35   |
| 13 | Thunderbird | 0.9188 | 0.7296 | 0.89   | 0.5911 | 56.5   |   49989 | 626     |   14      |   7442    | 11.37   |  3.89   |   37.19   |    3.84   |
| 14 | Average     | 0.9816 | 0.9011 | 0.9162 | 0.7554 | 20.655 |   46076 | 191.571 |    5.2143 |   2226.79 |  8.4221 |  2.8307 |    8.7907 |    0.4879 |


## Usage

- Configure log parsing and template mining using `.ini` files in `evaluator/`.
- Use provided datasets for benchmarking and evaluation.


## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements and new features.

## License

This project is open-source. See the LICENSE file for details.