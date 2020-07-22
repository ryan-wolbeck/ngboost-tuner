# ngboost-tuner
A CLI Tuner of NGBoost

## Install

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
    restart: unless-stopped
    command: tail -f /dev/null
```

This pacakge requires python 3.6+

Update soon to come to allow this
```bash
pip install ngboost-tuner
```

## Run

```bash 
ngboost_tuner tune 2> file.log

usage: ngboost_tuner tune [-h] [-i INPUT] [-l LIMIT] [-id ID_KEY] [-t TARGET]
                          [-c COLUMN] [-ef EVALUATION_FRACTION]
                          [-m MINIBATCH_FRAC] [-d MAX_DEPTH_RANGE]
                          [-n N_SEARCH_BOOSTERS] [-nf FINAL_BOOSTERS]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT, --input-file INPUT
                        Input file data; defaults to $INPUT_FILE if not set
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
```

## Reference
[1] [T. Duan, et al., NGBoost: Natural Gradient Boosting for Probabilistic Prediction (2019), ArXiv 1910.03225](https://arxiv.org/abs/1910.03225)