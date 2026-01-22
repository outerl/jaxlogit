import pytest
import numpy as np
import pandas as pd
import pathlib

import jax
import json

from jaxlogit.utils import wide_to_long
from jaxlogit.mixed_logit import (
    MixedLogit,
    ConfigData,
)
from jaxlogit.MixedLogitEncoder import MixedLogitEncoder, mixed_logit_decoder


def estimate_model_parameters():
    model, df, varnames, config = setup_correlated_example()
    model.fit(df[varnames], df["CHOICE"], varnames, df["alt"], df["custom_id"], {"TT": "n"}, config)
    return model


def fix_parameters():
    model, df, varnames, config = setup_correlated_example()
    varnames = ["ASC_CAR", "ASC_TRAIN", "ASC_SM", "CO", "TT"]
    df["ASC_SM"] = np.ones(len(df)) * (df["alt"] == "SM")
    set_vars = {"ASC_SM": 0.0}
    config = ConfigData(
        avail=df["AV"],
        panels=df["ID"],
        set_vars=set_vars,
        n_draws=1500,
        optim_method="BFGS",
        skip_std_errs=True,
    )
    model.fit(
        X=df[varnames],
        y=df["CHOICE"],
        varnames=varnames,
        alts=df["alt"],
        ids=df["custom_id"],
        randvars={"TT": "n"},
        config=config,
    )
    return model


def error_components():
    model, df, varnames, config = setup_correlated_example()
    varnames = ["ASC_CAR", "ASC_TRAIN", "ASC_SM", "CO", "TT"]
    df["ASC_SM"] = np.ones(len(df)) * (df["alt"] == "SM")
    randvars = {"ASC_CAR": "n", "ASC_TRAIN": "n", "ASC_SM": "n"}
    set_vars = {
        "ASC_SM": 0.0,
        "sd.ASC_TRAIN": 1.0,
        "sd.ASC_CAR": 0.0,
        "chol.ASC_CAR.ASC_TRAIN": 0.0,
        "chol.ASC_CAR.ASC_SM": 0.0,
    }  # Identification of error components, see J. Walker's PhD thesis (MIT 2001)

    config = ConfigData(
        avail=df["AV"],
        panels=df["ID"],
        set_vars=set_vars,
        n_draws=1500,
        include_correlations=True,  # Enable correlation between random parameters
        optim_method="BFGS",
        skip_std_errs=True,
    )

    model = MixedLogit()
    model.fit(
        X=df[varnames],
        y=df["CHOICE"],
        varnames=varnames,
        alts=df["alt"],
        ids=df["custom_id"],
        randvars=randvars,
        config=config,
    )
    return model


def save_correlated_example():
    models = [estimate_model_parameters, fix_parameters, error_components]
    files = [
        "correlated_example_estimate_params_output.json",
        "correlated_example_fix_params_output.json",
        "correlated_example_error_components_output.json",
    ]

    for i in range(len(models)):
        model = models[i]()
        with open(pathlib.Path(__file__).parent / "test_data" / files[i], "w") as f:
            json.dump(model, f, indent=4, cls=MixedLogitEncoder)


def setup_correlated_example():
    jax.config.update("jax_enable_x64", True)

    df_wide = pd.read_table("http://transp-or.epfl.ch/data/swissmetro.dat", sep="\t")
    # Keep only observations for commute and business purposes that contain known choices
    df_wide = df_wide[(df_wide["PURPOSE"].isin([1, 3]) & (df_wide["CHOICE"] != 0))]
    df_wide["custom_id"] = np.arange(len(df_wide))  # Add unique identifier
    df_wide["CHOICE"] = df_wide["CHOICE"].map({1: "TRAIN", 2: "SM", 3: "CAR"})

    df = wide_to_long(
        df_wide,
        id_col="custom_id",
        alt_name="alt",
        sep="_",
        alt_list=["TRAIN", "SM", "CAR"],
        empty_val=0,
        varying=["TT", "CO", "HE", "AV", "SEATS"],
        alt_is_prefix=True,
    )

    df["ASC_TRAIN"] = np.ones(len(df)) * (df["alt"] == "TRAIN")
    df["ASC_CAR"] = np.ones(len(df)) * (df["alt"] == "CAR")
    df["TT"], df["CO"] = df["TT"] / 100, df["CO"] / 100  # Scale variables
    annual_pass = (df["GA"] == 1) & (df["alt"].isin(["TRAIN", "SM"]))
    df.loc[annual_pass, "CO"] = 0  # Cost zero for pass holders

    varnames = ["ASC_CAR", "ASC_TRAIN", "CO", "TT"]
    model = MixedLogit()

    config = ConfigData(
        n_draws=1500,
        avail=(df["AV"]),
        panels=(df["ID"]),
        optim_method="L-BFGS-B",
        skip_std_errs=True,
    )

    return model, df, varnames, config


def save_batching_example():
    model = setup_batching_example()
    with open("tests/system_tests/test_data/batching_example_output.json", "w") as f:
        json.dump(model, f, indent=4, cls=MixedLogitEncoder)


def setup_batching_example():
    jax.config.update("jax_enable_x64", True)

    df = pd.read_csv(pathlib.Path(__file__).parent.parent.parent / "examples/electricity_long.csv")
    n_draws = 1000
    varnames = ["pf", "cl", "loc", "wk", "tod", "seas"]
    model = MixedLogit()

    config = ConfigData(
        panels=df["id"],
        n_draws=n_draws,
        skip_std_errs=True,  # skip standard errors to speed up the example
        batch_size=539,
        optim_method="L-BFGS-B",
    )

    model.fit(
        X=df[varnames],
        y=df["choice"],
        varnames=varnames,
        ids=df["chid"],
        alts=df["alt"],
        randvars={"pf": "n", "cl": "n", "loc": "n", "wk": "n", "tod": "n", "seas": "n"},
        config=config,
    )
    return model


@pytest.mark.parametrize(
    "example,file",
    [
        (estimate_model_parameters, "correlated_example_estimate_params_output.json"),
        (fix_parameters, "correlated_example_fix_params_output.json"),
        (error_components, "correlated_example_error_components_output.json"),
        (setup_batching_example, "batching_example_output.json"),
    ],
)
def test_previous_results(example: callable, file: str):
    with open(pathlib.Path(__file__).parent / "test_data" / file, "r") as f:
        previous_model = json.load(f, object_hook=mixed_logit_decoder)
    model = example()
    compare_models(model, previous_model)


def test_json():
    before = estimate_model_parameters()
    with open(pathlib.Path(__file__).parent / "test_data" / "test_json.json", "w") as f:
        json.dump(before, f, indent=4, cls=MixedLogitEncoder)
    with open(pathlib.Path(__file__).parent / "test_data" / "test_json.json", "r") as f:
        after = json.load(f, object_hook=mixed_logit_decoder)
    compare_models(after, before)


def test_predict():
    with open(pathlib.Path(__file__).parent / "test_data" / "batching_example_output.json", "r") as f:
        model = json.load(f, object_hook=mixed_logit_decoder)
    with open(pathlib.Path(__file__).parent / "test_data" / "predict_prob_output.json", "r") as f:
        # calculated from biogeme
        expected = json.load(f)
    expected = np.array(expected)
    df = pd.read_csv(pathlib.Path(__file__).parent.parent.parent / "examples/electricity_long.csv")
    varnames = ["pf", "cl", "loc", "wk", "tod", "seas"]

    config = ConfigData(
        panels=df["id"],
        skip_std_errs=True,  # skip standard errors to speed up the example
        batch_size=539,
        optim_method="L-BFGS-B",
    )
    config.init_coeff = model.coeff_
    prob = model.predict(
        df[varnames],
        varnames,
        df["alt"],
        df["chid"],
        {"pf": "n", "cl": "n", "loc": "n", "wk": "n", "tod": "n", "seas": "n"},
        config,
    )

    assert len(prob) == len(expected)
    for i in range(len(prob)):
        assert len(prob[i]) == len(expected[i])
        for j in range(len(prob[i])):
            assert prob[i][j] == pytest.approx(expected[i][j], rel=2e-1)


def compare_models(new, previous):
    assert list(new.coeff_names) == list(previous.coeff_names)
    assert list(new.coeff_) == pytest.approx(list(previous.coeff_), rel=7e-2)
    # assert list(new.stderr) == pytest.approx(list(previous.stderr), rel=1e-2)
    # assert list(new.zvalues) == pytest.approx(list(previous.zvalues), rel=7e-2)
    # assert new.loglikelihood == pytest.approx(previous.loglikelihood, rel=1e-2)
    # assert new.aic == pytest.approx(previous.aic, rel=1e-2)
    # assert new.bic == pytest.approx(previous.bic, rel=1e-2)


def main():
    save_correlated_example()
    save_batching_example()


if __name__ == "__main__":
    main()
