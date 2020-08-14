# ngboost-tuner
A CLI Tuner of NGBoost

## Install

```
pip install ngboost-tuner
```

Build from source
```bash
# Pull the code
git clone git@github.com:ryan-wolbeck/ngboost-tuner.git
# Build the container and detach
docker-compose up --build -d
# Exec into the container
docker-compose exec tuner bash
# Run the tuner
python -m ngboost_tuner tune -i file.tsv
```

Example docker-compose.yml
```yaml
version: '3.7' 
services:
  tuner:
    container_name: tuner
    build: .
    volumes:
      - .:/usr/src/app
    environment:
      - TARGET=target
      - ID=userid
      - TRAIN_COLUMNS=var1,var2
      - INPUT_FILE=/usr/src/app/file.tsv
      - TRAIN_FILE=/usr/src/data/train_dataset.csv
      - TEST_FILE=/usr/src/data/test_dataset.csv
      - VALIDATION_FILE=/usr/src/data/val_dataset.csv
      - SEPERATOR=,
      - COMPRESSION=gzip
      - LIGHTGBM=True
    restart: unless-stopped
    command: tail -f /dev/null
```

## Run

```bash 
ngboost_tuner tune 2> file.log

usage: ngboost_tuner tune [-h] [-i INPUT] [-s INPUT_FILE_SEPERATOR]
                          [-ct COMPRESSION_TYPE] [-tf TRAIN_FILE]
                          [-tef TEST_FILE] [-vf VALIDATION_FILE] [-l LIMIT]
                          [-id ID_KEY] [-t TARGET] [-c COLUMN]
                          [-ef EVALUATION_FRACTION] [-m MINIBATCH_FRAC]
                          [-d MAX_DEPTH_RANGE] [-n N_SEARCH_BOOSTERS]
                          [-nf FINAL_BOOSTERS] [-lgbm LIGHTGBM]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT, --input-file INPUT
                        Input file data; defaults to $INPUT_FILE if not set
  -s INPUT_FILE_SEPERATOR, --input-file-seperator INPUT_FILE_SEPERATOR
                        Input data file seperator, i.e. commas or tabs;
                        defaults to $SEPERATOR if not set
  -ct COMPRESSION_TYPE, --compression-type COMPRESSION_TYPE
                        Input data compression, i.e. gzip or None; defaults to
                        $COMPRESSION if not set
  -tf TRAIN_FILE, --train-file TRAIN_FILE
                        Input train data; defaults to $TRAIN_FILE if not set
  -tef TEST_FILE, --test-file TEST_FILE
                        Input test data; defaults to $TEST_FILE if not set
  -vf VALIDATION_FILE, --validation-file VALIDATION_FILE
                        Input validation data; defaults to $VALIDATION_FILE if
                        not set
  -l LIMIT, --limit LIMIT
                        Proportion of input tsv to use, .2 is 20 percent.
                        Default: 1.0 or all of input
  -id ID_KEY, --id-key ID_KEY
                        ID to consider for splits to prevent leakage. Default:
                        ID environment variable
  -t TARGET, --target TARGET
                        Target variable (predicted variable). Default value:
                        TARGET environment variable
  -c COLUMN, --column COLUMN
                        The full list of columns: Defaults to TRAIN_COLUMNS
                        environment variable
  -ef EVALUATION_FRACTION, --evaluation-fraction EVALUATION_FRACTION
                        Proportion of loadnums used for evaluation .2 is 20
                        percent of training leaving 80 percent train, 10
                        percent test, 10 percent validation. Default = .2
  -m MINIBATCH_FRAC, --minibatch-frac MINIBATCH_FRAC
                        Sample proportion for each boosting round during
                        hyperopt. Default = 1.0 or 100 percent
  -d MAX_DEPTH_RANGE, --max-depth-range MAX_DEPTH_RANGE
                        The range to test the max depth of the base learner.
                        Default 5 tests max_depth 2-5
  -n N_SEARCH_BOOSTERS, --n-search-boosters N_SEARCH_BOOSTERS
                        Number of n_estimators(booster) to use when searching.
                        Default = 20
  -nf FINAL_BOOSTERS, --final-boosters FINAL_BOOSTERS
                        Number of n_estimators(booster) to use to run the
                        final model. Default = 500
  -lgbm LIGHTGBM, --lightgbm LIGHTGBM
                        Set to true for lightgbm as base regresor. Default
                        $LIGHTGBM
```

## Reference
[1] [T. Duan, et al., NGBoost: Natural Gradient Boosting for Probabilistic Prediction (2019), ArXiv 1910.03225](https://arxiv.org/abs/1910.03225)