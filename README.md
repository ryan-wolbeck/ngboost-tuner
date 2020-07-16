# ngboost-tuner
A CLI Tuner of NGBoost

## Install

This pacakge requires python 3.6+

```bash
pip install ngboost-tuner
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

## Run

```bash 
ngboost_tuner tune > file.log

usage: ngboost_tuner tune [-h] [-i INPUT] [-t TARGET] [-id ID_KEY] [-c COLUMN]
                          [-ef EVALUATION_FRACTION]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT, --input-file INPUT
                        Input file data; defaults to $INPUT_FILE if not set
  -t TARGET, --target TARGET
                        Target variable (predicted variable). Default value:
                        hours_to_destination
  -id ID_KEY, --id-key ID_KEY
                        ID to consider for splits to prevent leakage. Default
                        loadnum
  -c COLUMN, --column COLUMN
                        The full list of columns: Defaults to prepick_flag,tea
                        m_flag,hazmat_flag,reefer_flag,hours_to_destination,ho
                        urs_to_scheduled_open,hours_to_scheduled_close,hours_e
                        lapsed,update_latitude,update_longitude,origin_latitud
                        e,origin_longitude,destination_latitude,destination_lo
                        ngitude,miles_traveled,miles_remaining,flatbed_flag,va
                        n_flag
  -ef EVALUATION_FRACTION, --evaluation-fraction EVALUATION_FRACTION
                        Proportion of loadnums used for evaluation .2 is 20
                        percent of training leaving 80 percent train, 10
                        percent test, 10 percent validation. Default = .2
```