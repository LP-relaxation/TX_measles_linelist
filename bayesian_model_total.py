# Please refer to `bayesian_model_timeseries.py` for more information
#   and documentation
# Remy was interested in what the results would look like without
#   the time series, and whether we needed time series at all
#   -- here is a simpler hierarchical Bayesian model that just
#   attempts to estimate the total symptomatic cases over a time
#   horizon (1 single scalar), rather than symptomatic cases per day

import numpy as np
import pandas as pd
import pymc as pm

import matplotlib.pyplot as plt
import arviz as az

from bayesian_model_timeseries import C_obs, H_obs, D_obs, IHR

num_days = len(C_obs)

######################################################################
############################# PYMC MODEL #############################
######################################################################

total_num_cases_linelist = np.sum(C_obs)
total_num_hosp_linelist = np.sum(H_obs)
total_num_deaths_linelist = np.sum(D_obs)

if __name__ == "__main__":

    with pm.Model() as measles_model:

        case_fatal_rate = pm.Uniform("case_fatal_rate", lower=1e-3, upper=3e-3)
        reporting_rate = pm.Uniform("reporting_rate", lower=0.1, upper=0.9)

        # True total (symptomatic) infections
        I_total_total = pm.DiscreteUniform("I_total", lower=total_num_cases_linelist, upper=10*total_num_cases_linelist)

        # Observed reported cases
        C_total = pm.Binomial("C_total", n=I_total_total, p=reporting_rate, observed=total_num_cases_linelist)

        H_total = pm.Binomial("H", n=I_total_total, p=IHR, observed=total_num_hosp_linelist)

        D_total = pm.Binomial("D", n=I_total_total, p=case_fatal_rate, observed=total_num_deaths_linelist)

        measles_model.debug(verbose=True)

        trace = pm.sample(10000, tune=1000, target_accept=0.98, chains=8, cores=1, progressbar=False)

    ######################################################################
    ############################# PLOTTING ###############################
    ######################################################################



    posterior_I = trace.posterior["I_total"]
    I_mean = posterior_I.mean(dim=["chain", "draw"])
    I_hdi = az.hdi(posterior_I, hdi_prob=0.95)

    mean_val = int(np.round(I_mean.values))
    lower_val = int(np.round(I_hdi["I_total"].sel(hdi="lower").values))
    upper_val = int(np.round(I_hdi["I_total"].sel(hdi="higher").values))
    reported_val = int(np.sum(C_obs))

    I_df = pd.DataFrame([{
        "Mean estimated total symptomatic infections": mean_val,
        "Lower bound (95% credible)": lower_val,
        "Upper bound (95% credible)": upper_val,
        "Linelist total reported case counts": reported_val
    }])

    I_df.to_csv("estimated_symptomatic_infections_TOTAL_95percent.csv")