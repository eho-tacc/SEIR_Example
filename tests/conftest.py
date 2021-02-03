import sys
import pytest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray
from src.seir import *
from src.implicit_node import *
from copy import deepcopy


def update_start_pop(travel_df):
    grouped = travel_df.groupby(['source', 'age_src'])['n'].sum().reset_index()
    nodes = sorted(travel_df['source'].unique())
    ages = sorted(travel_df['age_src'].unique(), reverse=True)
    pop_arr_s = np.zeros([len(nodes), len(ages)])
    pop_arr_e = np.zeros([len(nodes), len(ages)])
    pop_arr_i = np.zeros([len(nodes), len(ages)])
    pop_arr_r = np.zeros([len(nodes), len(ages)])
    for i, node in enumerate(nodes):
        for j, age in enumerate(ages):
            new_total = grouped[(grouped['source']==node) & (grouped['age_src']==age)]['n'].item()
            if new_total > 2:
                pop_arr_s[i, j] = new_total-1
                pop_arr_i[i, j] = 1
    return pop_arr_s, pop_arr_e, pop_arr_i, pop_arr_r


def update_travel(travel_df, new_count, source, destination, age_src='young', age_dest='young'):
    travel_idx = travel_df.set_index(['source', 'destination', 'age_src', 'age_dest'])
    travel_dict = travel_idx.to_dict()
    travel_dict['n'][(source, destination, age_src, age_dest)] = new_count
    updated_df = pd.DataFrame.from_dict(travel_dict).reset_index()
    updated_df.columns = ['source', 'destination', 'age_src', 'age_dest', 'destination_type', 'n']
    return updated_df


def update_contact_rate(contact_df, new_rate, age1='young', age2='young'):
    contact_idx = contact_df.set_index(['age1', 'age2'])
    contact_dict = contact_idx.to_dict()
    contact_dict['daily_per_capita_contacts'][(age1, age2)] = new_rate
    updated_df = pd.DataFrame.from_dict(contact_dict).reset_index()
    updated_df.columns = ['age1', 'age2', 'daily_per_capita_contacts']
    return updated_df


def discrete_time_approx(rate, timestep):
    """
    :param rate: daily rate
    :param timestep: timesteps per day
    :return: rate rescaled by time step
    """
    return (1 - (1 - rate)**(1/timestep))


@pytest.fixture
def atol():
    """ #0.001 max persons difference"""
    return 0.001


@pytest.fixture
def rtol():
    """ #0.05% max difference"""
    return 0.0005


@pytest.fixture
def params_template():
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


@pytest.fixture
def travel():
    return pd.read_csv('inputs/travel2.csv')


@pytest.fixture
def contact():
    return pd.read_csv('inputs/contact.csv')


@pytest.fixture
def daily_timesteps():
    return 10


@pytest.fixture
def partition(travel, contact, daily_timesteps):
    return partition_contacts(travel, contact, daily_timesteps=daily_timesteps)


@pytest.fixture
def phi_matrix(partition):
    return contact_matrix(partition)


@pytest.fixture
def test1(params_template, phi_matrix):
    test1 = deepcopy(params_template)
    test1['phi'] = phi_matrix
    return test1


@pytest.fixture
def ref_params():
    return {
      "mu": 0.0,
      "sigma": 0.5,
      "beta": 0.1,
      "gamma": 0.2,
      "omega": 0.1,
      "start_S": [[73, 0]],
      "start_E": [[0, 0]],
      "start_I": [[2, 0]],
      "start_R": [[0, 0]],
      "days": 30,
      "outpath": "outputs/single_node",
      "phi": [[[[5/10, 0], [0, 0]]]],
      "n_sims": 1,
      "stochastic": "False",
      "n_age": 2,
      "n_nodes": 1,
      "sim_idx": 0,
      "interval_per_day": 10
    }


@pytest.fixture
def polymod():
    return pd.read_csv('./data/Cities_Data/ContactMatrixAll_5AgeGroups.csv',
                        header=None)


@pytest.fixture
def max_cr(polymod):
    return polymod.max().max()


@pytest.fixture
def min_cr(polymod):
    return polymod.min().min()


@pytest.fixture
def max_test(max_contact, travel, params_template):
    max_partition = partition_contacts(travel, max_contact, daily_timesteps=10)
    max_phi_matrix = contact_matrix(max_partition)

    test_max = deepcopy(params_template)
    test_max['phi'] = max_phi_matrix
    return test_max
    # test_max_model = SEIR(test_max)
    # test_max_model.seir()


@pytest.fixture
def max_ref(max_cr, ref_params):
    ref_max = deepcopy(ref_params)
    ref_max['phi'] = [[[[max_cr/10, 0], [0, 0]]]]
    return ref_max
    # ref_max_model = SEIR(ref_max)
    # ref_max_model.seir()


@pytest.fixture
def min_contact(contact, min_cr):
    return update_contact_rate(contact, min_cr)


@pytest.fixture
def max_contact(contact, max_cr):
    return update_contact_rate(contact, max_cr)


@pytest.fixture
def min_test(params_template, travel, min_contact):
    min_partition = partition_contacts(travel, min_contact, daily_timesteps=10)
    min_phi_matrix = contact_matrix(min_partition)
    test_min = deepcopy(params_template)
    test_min['phi'] = min_phi_matrix
    return test_min


@pytest.fixture
def ref_min(min_cr, ref_params):
    d = deepcopy(ref_params)
    d['phi'] = [[[[min_cr/10, 0], [0, 0]]]]
    return d


@pytest.fixture
def travel3():
    """# # Sensitivity analysis: additional nodes"""
    return pd.read_csv('inputs/travel3.csv')


@pytest.fixture
def test3(travel, travel3, params_template, contact):
    new_travel = update_travel(travel, 50, source='A', destination='A')
    pop_s, pop_e, pop_i, pop_r = update_start_pop(new_travel)

    partition3 = partition_contacts(travel3, contact, daily_timesteps=10)
    phi_matrix3 = contact_matrix(partition3)
    pop_s, pop_e, pop_i, pop_r = update_start_pop(travel3)
    test3_params = deepcopy(params_template)
    test3_params['phi'] = phi_matrix3
    test3_params['start_S'] = pop_s
    test3_params['start_E'] = pop_e
    test3_params['start_I'] = pop_i
    test3_params['start_R'] = pop_r
    test3_params['n_nodes'] = 3
    return test3_params


@pytest.fixture
def ref3(params_template, travel3):
    ref3_params = deepcopy(params_template)
    ref3_params['phi'] = [[[[5/10, 0], [0, 0]]]]
    ref3_params_s = np.array([sorted(travel3.groupby(['age_src'])['n'].sum(), reverse=True)])
    ref3_params['start_S'] = ref3_params_s[0, 0] - 3
    ref3_params['start_E'] = np.array([[0, 0]])
    ref3_params['start_I'] = np.array([[3, 0]])
    ref3_params['start_R'] = np.array([[0, 0]])
    ref3_params['n_nodes'] = 1
    return ref3_params


@pytest.fixture
def test3_min(min_contact, travel3, ref3, test3):
    partition3_min = partition_contacts(travel3, min_contact, daily_timesteps=10)
    phi_matrix3_min = contact_matrix(partition3_min)
    test3_min_params = deepcopy(test3)
    test3_min_params['phi'] = phi_matrix3_min
    return test3_min_params


@pytest.fixture
def ref3_min(min_cr, ref3):
    ref3_min_params = deepcopy(ref3)
    ref3_min_params['phi'] = [[[[min_cr/10, 0], [0, 0]]]]
    return ref3_min_params


@pytest.fixture
def test3_max(test3, travel3, max_contact):
    partition3_max = partition_contacts(travel3, max_contact, daily_timesteps=10)
    phi_matrix3_max = contact_matrix(partition3_max)
    test3_max_params = deepcopy(test3)
    test3_max_params['phi'] = phi_matrix3_max
    return test3_max_params


@pytest.fixture
def ref3_max(ref3, max_cr):
    ref3_max_params = deepcopy(ref3)
    ref3_max_params['phi'] = [[[[max_cr/10, 0], [0, 0]]]]
    return ref3_max_params


@pytest.fixture(params=[
    ('test1', 'ref_params'),
    ('max_test', 'max_ref'),
    ('min_test', 'ref_min'),
    ('test3', 'ref3'),
    ('test3_min', 'ref3_min'),
    ('test3_max', 'ref3_max'),
])
def comparison(request):
    """A workaround alternative to passing fixtures in pytest.mark.parametrize"""
    return [request.getfixturevalue(f) for f in request.param]
