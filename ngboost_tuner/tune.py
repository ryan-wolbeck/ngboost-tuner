from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ngboost.ngboost import NGBoost
from ngboost.learners import default_tree_learner
from ngboost.distns import Normal
from ngboost.scores import MLE
from ngboost import NGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    median_absolute_error,
    mean_absolute_error,
    mean_squared_error,
)
from hyperopt import hp, tpe, space_eval
from hyperopt.pyll.base import scope
from hyperopt.fmin import fmin
from hyperopt import STATUS_OK, Trials
import pandas as pd
import warnings
import logging as log
import pickle
from pathlib import Path
import os


def run(args):

    log.info("Reading data into Memory")
    data = pd.read_csv(args.input, sep="\t")
    data = data.sample(frac=args.limit, random_state=1)
    log.info("Data finished loading into Memory")

    if args.id_key != None:
        ids = data[args.id_key].unique()
        ids = pd.Series(ids)

        eval_ids = ids.sample(frac=args.evaluation_fraction, random_state=1)
        train_ids = ids.drop(eval_ids.index)
        test_ids = eval_ids.sample(frac=0.5, random_state=1)
        val_ids = eval_ids.drop(test_ids.index)

        da_df_train = data[data[args.id_key].isin(train_ids.to_list())]
        da_df_test = data[data[args.id_key].isin(test_ids.to_list())]
        da_df_val = data[data[args.id_key].isin(val_ids.to_list())]
    else:
        X_intermediate, da_df_test = train_test_split(data, 
                                                        shuffle=True,
                                                        test_size=args.evaluation_fraction/2, 
                                                        random_state=1)

        # train/validation split (gives us train and validation sets)
        da_df_train, da_df_val = train_test_split(X_intermediate,
                                                        shuffle=False,
                                                        test_size=args.evaluation_fraction/2,
                                                        random_state=1)

        # delete intermediate variables
        del X_intermediate

    if args.column != None and args.target != None:
        args.column.remove(args.target)
        x_test = da_df_test[args.column]
        y_test = da_df_test[args.target]

        log.info("Feature set : {}".format(args.column))
        log.info("Target : {}".format(args.target))

        x_valid = da_df_val[args.column]
        y_valid = da_df_val[args.target]

        x = da_df_train[args.column]
        y = da_df_train[args.target]
    else:
        sys.exit("Columns were not supplied")

    base_models = [
        DecisionTreeRegressor(criterion="friedman_mse", max_depth=i)
        for i in range(2, args.max_depth_range + 1)
    ]
    log.info(base_models)

    space = {
        "learning_rate": hp.uniform("learning_rate", 0.05, 1),
        "Base": hp.choice("Base", base_models),
    }

    default_params = {
        "n_estimators": args.n_search_boosters,
        "verbose_eval": 10,
        "random_state": 1,
        "minibatch_frac": args.minibatch_frac,
    }

    def objective(params):

        params.update(default_params)

        print(params)
        ngb = NGBRegressor(**params).fit(
            x.values,
            y.values,
            X_val=x_valid.values,
            Y_val=y_valid.values,
            early_stopping_rounds=2,
        )
        loss = ngb.evals_result["val"]["LOGSCORE"][ngb.best_val_loss_itr]
        log.info(params)
        results = {"loss": loss, "status": STATUS_OK}

        return results

    TRIALS = Trials()
    log.info("Start parameter optimization...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best = fmin(
            fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=TRIALS
        )
    log.info("...done")

    best_params = space_eval(space, best)

    log.info(f"The Best parameters from hypteropt{best_params}")

    default_params = {
        "n_estimators": args.final_boosters,
        "verbose_eval": 1,
        "random_state": 1,
    }

    best_params.update(default_params)

    log.info(f"Running a model on the best parameter set {best_params}")

    ngb = NGBRegressor(**best_params).fit(
        x.values,
        y.values,
        X_val=x_valid.values,
        Y_val=y_valid.values,
        early_stopping_rounds=2,
    )

    log.info("Finished training the final model, running diagnostics")

    Y_pred = ngb.predict(x_test)
    Mae = median_absolute_error(y_test, Y_pred)
    log.info(f"Median Absolute Error = {Mae}")

    mea = mean_absolute_error(y_test, Y_pred)
    log.info(f"Mean Absolute Error = {mea}")

    log.info("Saving the model file")

    path = os.path.expanduser("/usr/src/app/models/")

    if not os.path.exists(path):
        os.mkdir(path)

    pickle.dump(ngb, open(f"{path}ngbtest.p", "wb"))

    log.info(f"Model saved to: {path}")
