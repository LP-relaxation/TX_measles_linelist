# TODO
#

# Lauren ideas (not necessarily to-do, but to-think-about)
# - Instead of using random walk, can we make another modeling assumption?
#   For example, somehow incorporate an S-I model in the infectious spread?
#   Lauren mentioned a con of this is that we may have to estimate more parameters.
# - My initial thought:

# TX measles line-list data 2025
# Objective: infer "true/total" number of cases --
#   we will refer to this as the true number of symptomatic infections --
#   we ignore asymptomatic infections because those cannot infect others
#   (source: https://academic.oup.com/jid/article-abstract/189/Supplement_1/S165/821757?redirectedFrom=fulltext
#   and https://www.health.state.mn.us/diseases/measles/hcp/clinical.html)
# Name of methods: Bayesian hierarchical model + sampling using NUTS!

import numpy as np
import pandas as pd
import pymc as pm

import matplotlib.pyplot as plt
import arviz as az

###############################################################################
############################# DATA CLEANING ###################################
###############################################################################

df = pd.read_excel("UT_Measles_07may.xlsx")


def extract_boolean_status(value, status):
    if pd.isna(value) or value == "Unknown":
        return "Unknown"
    return status in value


# Rename the columns
df.rename(columns={df.columns[0]: "ID",
                   df.columns[1]: "IsHospitalized",
                   df.columns[7]: "AttendsSchoolOrChildcare",
                   df.columns[9]: "IsVaccinated",
                   df.columns[10]: "DateRashOnset"}, inplace=True)

# Note: we don't use these columns -- I was just cleaning the data and making this column more sensible
# School or childcare
df["IsStudent"] = df["AttendsSchoolOrChildcare"].apply(lambda x: extract_boolean_status(x, "Student"))
df["IsEmployee"] = df["AttendsSchoolOrChildcare"].apply(lambda x: extract_boolean_status(x, "Employee"))
df["IsVolunteer"] = df["AttendsSchoolOrChildcare"].apply(lambda x: extract_boolean_status(x, "Volunteer"))

# Blank values in IsHospitalized columns mean that person was NOT hospitalized
# How do we know? Well, according to DSHS there have been 94 hospitalizations
#   since late January, and there are 94 "Yes" values in this column
#   (source: https://www.dshs.texas.gov/news-alerts/measles-outbreak-2025)
df["IsHospitalized"] = df["IsHospitalized"].fillna("No")

###############################################################################
############################# EXTRACT TIME SERIES #############################
###############################################################################

# Linelist data rows are case entries, and dates are contained in another column
# What we want is a time series: for each date, we want a count of cases
#   or hospitalizations
# Deaths time series is "hardcoded" based on DSHS announcements (see sources below)

case_time_series = df["DateRashOnset"].value_counts().sort_index()
case_time_series = case_time_series.asfreq("D")
case_time_series = case_time_series.fillna(0).astype(int)

hosp_time_series = df[df["IsHospitalized"] == "Yes"]["DateRashOnset"].value_counts().sort_index()
hosp_time_series = hosp_time_series.asfreq("D")
hosp_time_series = hosp_time_series.fillna(0).astype(int)

hosp_time_series = hosp_time_series.reindex(case_time_series.index, fill_value=0)

# Deaths sources
# Feb 26th 2025 first TX death https://www.dshs.texas.gov/news-alerts/texas-announces-first-death-measles-outbreak
# Apr 6th 2025 second TX death https://www.dshs.texas.gov/news-alerts/texas-announces-second-death-measles-outbreak

death_time_series = pd.Series(0, index=case_time_series.index)
death_time_series.loc["2025-02-26"] = 1
death_time_series.loc["2025-04-06"] = 1

C_obs = np.asarray(case_time_series)
H_obs = np.asarray(hosp_time_series)
D_obs = np.asarray(death_time_series)

num_days = len(C_obs)

######################################################################
############################# PARAMETERS #############################
######################################################################

# Note: this parameter is unused -- but computed here just in case
# 0.1273 based on TX average -- as of 05/30/2025, 94 hospitalizations, 738 reported cases
#   since late January: https://www.dshs.texas.gov/news-alerts/measles-outbreak-2025
# case_hosp_rate = pm.Triangular("case_hosp_rate", lower=0.1223, c=0.1273, upper=0.1323)
# Remy found a Canada source with about a 7.6% CHR https://health-infobase.canada.ca/measles-rubella/
case_hosp_rate = 0.1273

# Extremely preliminary
# CDC: 1 in 20 unvaccinated people in the US who get measles will be hospitalized
#   https://www.cdc.gov/measles/signs-symptoms/index.html
#   As Remy pointed out, do we have a paper cite for this?
# DSHS: about 90.9% of Texas children born in 2020 are vaccinated for measles
#   https://www.dshs.texas.gov/immunizations/data/surveys/nis/children
IHR = 0.02
# 0.05 x .909 is 0.018 --> rounded to 0.02
# CURRENTLY THIS IS FIXED -- BUT WE CAN ASSIGN A PRIOR TO THIS!

# From ECDC fact sheet on measles: https://www.ecdc.europa.eu/en/measles/facts
#   (this sheet is also used in TACC simulator)
# case_fatal_rate = pm.Uniform("case_fatal_rate", lower=1e-3, upper=3e-3)

# 11-12 days from exposure to first symptoms
# https://www.cdc.gov/measles/hcp/communication-resources/clinical-diagnosis-fact-sheet.html
incubation_period = 12

# TOTAL GUESS: need to ask public health officials for this!
# We define hospital sojourn here as the time someone spends in the hospital before dying
# Struggled to find literature on this -- LP
hosp_sojourn = 7

######################################################################
############################# PYMC MODEL #############################
######################################################################

with pm.Model() as measles_model:

    case_fatal_rate = pm.Uniform("case_fatal_rate", lower=1e-3, upper=3e-3)

    scaling_factor = pm.Uniform("scaling_factor", lower=1, upper=5)

    mu = scaling_factor * C_obs

    # Gaussian Random Walk on log-scale for smoothness
    # Note we use log-scale to enforce positivity -- but then we convert it back
    init = pm.Normal.dist(mu=1, sigma=0.1)
    log_I_symp = pm.GaussianRandomWalk("log_I_symp", sigma=0.2, shape=num_days, init_dist=init)
    I_symp = pm.Deterministic("I_symp", pm.math.exp(log_I_symp))

    # Reporting rate between 0 and 1 -- nothing crazy
    reporting_rate = pm.Beta("reporting_rate", alpha=2, beta=5)

    # Lauren was wondering we why would want to use Poisson instead of Binomial --
    #   here, I chose Poisson because I_symp * reporting_rate is not necessarily
    #   an integer, and if we force conversions to integers, this screws up the
    #   derivative computation that is done in the back-end of pyMC to make the
    #   NUTS sampler work -- so, Poisson is nice because its rate can be non-integer
    #   but it always outputs an integer value
    C_reported = pm.Poisson("C_reported", mu=I_symp * reporting_rate, observed=C_obs)

    # I had significant trouble with implementing the shifts -- it was likely because
    #   of the Binomial issue explained below -- but in any case, I_shifted_for_H is
    #   H shifted BACKWARDS by incubation period -- and I made the values before time 0
    #   (so, the first N values in the shifted array, where N = incubation period)
    #   equal to 1. We can look into this -- LP
    I_shifted_for_H = pm.Deterministic("I_shifted_for_H",
                                       pm.math.concatenate((pm.math.ones(incubation_period),
                                                           I_symp[:-incubation_period])))

    # I originally had these values as Binomial -- however, the likelihood
    #   function is not defined when the historical/observed H is greater than
    #   the parameter n -- so, this leads to issues -- but using the Poisson
    #   distribution does not have this constraint -- LP

    H = pm.Poisson("H", mu = I_shifted_for_H * IHR, observed=H_obs)

    D = pm.Poisson("D", mu = I_symp * case_fatal_rate, observed=D_obs)

    measles_model.debug(verbose=True)

    # One could probably use multiple cores to speed this up
    # My multiprocessing module behaves horribly for whatever reason, so
    #   I set the number of cores to 1 to avoid issues -- but we can play around with
    #   this -- with 1 core, on my CPU it takes about 7 minutes to sample -- LP
    trace = pm.sample(10000, tune=1000, target_accept=0.98, chains=8, cores=1, progressbar=False)

######################################################################
############################# PLOTTING ###############################
######################################################################

posterior_C = trace.posterior["I_symp"]
C_mean = posterior_C.mean(dim=["chain", "draw"])
C_hdi = az.hdi(posterior_C, hdi_prob=0.9)

days = np.arange(len(C_mean))
plt.fill_between(days, C_hdi["I_symp"].sel(hdi="lower"), C_hdi["I_symp"].sel(hdi="higher"), alpha=0.3, label="90% HDI")
plt.plot(days, C_mean, label="Estimated symptomatic infections (incidence), posterior mean")
plt.plot(days, C_obs, label="Reported cases")
plt.xlabel("Day")
plt.ylabel("Cases")
plt.legend()
plt.tight_layout()
plt.savefig("est_symp_inf_90percent.png", dpi=1200)

plt.show()

