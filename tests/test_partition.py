import sys
import pytest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray
from src.seir import *
from src.implicit_node import *
from copy import deepcopy

SHOW_PLT = False


# the only actual test function in this module
def test_partition(comparison, atol, rtol):
    # comparison is a tuple of resolved fixture values. expand it
    test, ref = comparison
    kwargs = dict(atol=atol, rtol=rtol)

    # always use allclose
    method = xarray.testing.assert_allclose

    # generate and run models
    test_model = SEIR(test)
    ref_model = SEIR(ref)
    test_model.seir()
    ref_model.seir()

    test_model_s = xr_summary(test_model.final.S, sel={'age': 'young'}, timeslice=slice(0, test_model.duration), sum_over='node')
    ref_model_s = xr_summary(ref_model.final.S, sel={'age': 'young'}, timeslice=slice(0, ref_model.duration), sum_over='node')
    diff_s = test_model_s - ref_model_s

    test_model_e = xr_summary(test_model.final.E, sel={'age': 'young'}, timeslice=slice(0, test_model.duration), sum_over='node')
    ref_model_e = xr_summary(ref_model.final.E, sel={'age': 'young'}, timeslice=slice(0, ref_model.duration), sum_over='node')
    diff_e = test_model_e - ref_model_e

    test_model_i = xr_summary(test_model.final.I, sel={'age': 'young'}, timeslice=slice(0, test_model.duration), sum_over='node')
    ref_model_i = xr_summary(ref_model.final.I, sel={'age': 'young'}, timeslice=slice(0, ref_model.duration), sum_over='node')
    diff_i = test_model_i - ref_model_i

    test_model_r = xr_summary(test_model.final.R, sel={'age': 'young'}, timeslice=slice(0, test_model.duration), sum_over='node')
    ref_model_r = xr_summary(ref_model.final.R, sel={'age': 'young'}, timeslice=slice(0, ref_model.duration), sum_over='node')
    diff_r = test_model_r - ref_model_r

    try:
        method(test_model_s, ref_model_s, **kwargs)
    except AssertionError as _err:
        print('Differing values for susceptible timeseries.')
        raise

    try:
        method(test_model_e, ref_model_e, **kwargs)
    except AssertionError as _err:
        print('Differing values for exposed timeseries.')
        raise

    try:
        method(test_model_i, ref_model_i, **kwargs)
    except AssertionError as _err:
        print('Differing values for infected timeseries.')
        raise

    try:
        method(test_model_r, ref_model_r, **kwargs)
    except AssertionError as _err:
        print('Differing values for recovered timeseries.')
        raise

    return diff_s, diff_e, diff_i, diff_r



# plot test1 vs ref_model
def plt_test1_vs_ref_params():
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    s_diff.plot(ax=ax, color='b')
    e_diff.plot(ax=ax, color='g')
    i_diff.plot(ax=ax, color='r')
    r_diff.plot(ax=ax, color='k')
    plt.legend(("S", "E", "I", "R"), loc=0)
    plt.ylabel("N Partition - N Baseline")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.title("Two Local, One Contextual vs Baseline Mixing = 5 contacts/person/day")
    plt.tight_layout()
    #ax.set_ylim(-2, 3)
    plt.axhline(y=0, c='gray', ls='dotted')
    if SHOW_PLT is True:
        plt.show()


# strict 1
# test_partition(test1_model, ref_model, method=xarray.testing.assert_equal)
# update_contact_rate(contact, 0.3)

# test_max_model vs ref_max_model
# ds_max, de_max, di_max, dr_max = test_partition(test_max_model, ref_max_model, atol=abs_tol, rtol=rel_tol)


# plot
def plt_max_test_vs_ref_max():
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ds_max.plot(ax=ax, color='b')
    de_max.plot(ax=ax, color='g')
    di_max.plot(ax=ax, color='r')
    dr_max.plot(ax=ax, color='k')
    plt.legend(("S", "E", "I", "R"), loc=0)
    plt.ylabel("N Partition - N Baseline")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.title("Two Local, One Contextual vs Baseline Mixing = 10.2 contacts/person/day")
    plt.tight_layout()
    #ax.set_ylim(-2, 3)
    plt.axhline(y=0, c='gray', ls='dotted')
    if SHOW_PLT is True:
        plt.show()

# test min
# ds_min, de_min, di_min, dr_min = test_partition(test_min_model, ref_min_model, atol=abs_tol, rtol=rel_tol)


def plt_min_test_vs_ref_min():
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ds_min.plot(ax=ax, color='b')
    de_min.plot(ax=ax, color='g')
    di_min.plot(ax=ax, color='r')
    dr_min.plot(ax=ax, color='k')
    plt.legend(("S", "E", "I", "R"), loc=0)
    plt.ylabel("N Partition - N Baseline")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.title("Two Local, One Contextual vs Baseline Mixing = 0.2 contacts/person/day")
    plt.tight_layout()
    #ax.set_ylim(-2,3)
    plt.axhline(y=0, c='gray', ls='dotted')
    if SHOW_PLT is True:
        plt.show()

#
# ref_min_model.plot_timeseries()
# test_min_model.plot_timeseries()
# ref_max_model.plot_timeseries()
# test_max_model.plot_timeseries()


# # Sensitivity analysis: population size
@pytest.fixture
def pop_template():
    return {
      "mu": 0.0,
      "sigma": 0.5,
      "beta": 0.1,
      "gamma": 0.2,
      "omega": 0.1,
      "start_S": [[24, 0], [49, 0]],
      "start_E": [[0, 0], [0, 0]],
      "start_I": [[1, 0], [1, 0]],
      "start_R": [[0, 0], [0, 0]],
      "days": 30,
      "outpath": "outputs/multiple_nodes",
      "phi": [], # fill in after partitioning
      "n_sims": 1,
      "stochastic": "False",
      "n_age": 2,
      "n_nodes": 2,
      "sim_idx": 0, # single deterministic run
      "interval_per_day": 10
    }

# test model 3
# s3_diff, e3_diff, i3_diff, r3_diff = test_partition(test3_model, ref3_model, atol=abs_tol, rtol=rel_tol)

# plot
def plt_test3_vs_ref3():
    test3_model.plot_timeseries()
    ref3_model.plot_timeseries()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    s3_diff.plot(ax=ax, color='b')
    e3_diff.plot(ax=ax, color='g')
    i3_diff.plot(ax=ax, color='r')
    r3_diff.plot(ax=ax, color='k')
    plt.legend(("S", "E", "I", "R"), loc=0)
    plt.ylabel("N Partition - N Baseline")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.title("Three Local, One Contextual vs Baseline Mixing = 5 contacts/person/day")
    plt.tight_layout()
    #ax.set_ylim(-2, 3)
    plt.axhline(y=0, c='gray', ls='dotted')
    # plt.show()



def plot_local_3v2():
    local3v2_contact5_s = s3_diff - s_diff
    local3v2_contact5_e = e3_diff - e_diff
    local3v2_contact5_i = i3_diff - i_diff
    local3v2_contact5_r = r3_diff - r_diff

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    local3v2_contact5_s.plot(ax=ax, color='b')
    local3v2_contact5_e.plot(ax=ax, color='g')
    local3v2_contact5_i.plot(ax=ax, color='r')
    local3v2_contact5_r.plot(ax=ax, color='k')
    plt.legend(("S", "E", "I", "R"), loc=0)
    plt.ylabel("Population Size")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.title("Three Local, One Contextual vs Two Local, One Contextual")
    plt.tight_layout()
    # plt.show()

    ref3_model.plot_timeseries()
    test3_model.plot_timeseries()


def plt_test3_min_vs_ref3_min():
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    s3_diff_min.plot(ax=ax, color='b')
    e3_diff_min.plot(ax=ax, color='g')
    i3_diff_min.plot(ax=ax, color='r')
    r3_diff_min.plot(ax=ax, color='k')
    plt.legend(("S", "E", "I", "R"), loc=0)
    plt.ylabel("N Partition - N Baseline")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.title("Three Local, One Contextual vs Baseline Mixing = 0.2 contacts/person/day")
    plt.tight_layout()
    plt.axhline(y=0, c='gray', ls='dotted')
    #ax.set_ylim(-2, 3)
    # plt.show()

# plot
def plt_test3_max_vs_ref3_max():
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    s3_diff_max.plot(ax=ax, color='b')
    e3_diff_max.plot(ax=ax, color='g')
    i3_diff_max.plot(ax=ax, color='r')
    r3_diff_max.plot(ax=ax, color='k')
    plt.legend(("S", "E", "I", "R"), loc=0)
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.title("Three Local, One Contextual vs Baseline Mixing = 10.2 contacts/person/day")
    plt.tight_layout()
    plt.ylabel("N Partition - N Baseline")
    plt.axhline(y=0, c='gray', ls='dotted')
    #ax.set_ylim(-2, 3)
    # plt.show()



def sixteen_nodes():
    raise NotImplementedError()
    # # 16 Nodes

    travel16 = pd.read_csv('inputs/travel16.csv')
    partition16 = partition_contacts(travel16, contact, daily_timesteps=10)
    phi_matrix16 = contact_matrix(partition16)
    pop_s16, pop_e16, pop_i16, pop_r16 = update_start_pop(travel16)

    test16 = deepcopy(params_template)
    test16['phi'] = phi_matrix16
    test16['start_S'] = pop_s16
    test16['start_E'] = pop_e16
    test16['start_I'] = pop_i16
    test16['start_R'] = pop_r16
    test16['n_nodes'] = 16

    ref16 = deepcopy(params_template)
    ref16['phi'] = [[[[5/10, 0], [0, 0]]]]
    ref16_s = np.array([sorted(travel16.groupby(['age_src'])['n'].sum(), reverse=True)])
    ref16['start_S'] = ref16_s[0, 0] - 16
    ref16['start_E'] = np.array([[0, 0]])
    ref16['start_I'] = np.array([[16, 0]])
    ref16['start_R'] = np.array([[0, 0]])
    ref16['n_nodes'] = 1

    test16_model = SEIR(test16)
    test16_model.seir()

    ref16_model = SEIR(ref16)
    ref16_model.seir()


    # In[130]:


    ref16_s


    # In[131]:


    s16_diff_max, e16_diff_max, i16_diff_max, r16_diff_max = test_partition(test16_model, ref16_model, atol=abs_tol, rtol=rel_tol)


    # In[132]:


    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    s16_diff_max.plot(ax=ax, color='b')
    e16_diff_max.plot(ax=ax, color='g')
    i16_diff_max.plot(ax=ax, color='r')
    r16_diff_max.plot(ax=ax, color='k')

    plt.legend(("S", "E", "I", "R"), loc=0)
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.title("16 Local, One Contextual vs Baseline Mixing = 5 contacts/person/day")
    plt.tight_layout()
    plt.ylabel("N Partition - N Baseline")
    plt.axhline(y=0, c='gray', ls='dotted')
    #ax.set_ylim(-2, 3)

    plt.show()
