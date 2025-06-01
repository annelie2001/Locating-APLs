"""
Python model 'APL-SFD-Würzburg-V3.py'
Translated using PySD
"""

from pathlib import Path
import numpy as np

from pysd.py_backend.functions import integer
from pysd.py_backend.statefuls import Integ
from pysd import Component

__pysd_version__ = "3.14.3"

__data = {"scope": None, "time": lambda: 0}

_root = Path(__file__).parent


component = Component()

#######################################################################
#                          CONTROL VARIABLES                          #
#######################################################################

_control_vars = {
    "initial_time": lambda: 0,
    "final_time": lambda: 10,
    "time_step": lambda: 1,
    "saveper": lambda: time_step(),
}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


@component.add(name="Time")
def time():
    """
    Current time of the model.
    """
    return __data["time"]()


@component.add(
    name="FINAL TIME", units="Year", comp_type="Constant", comp_subtype="Normal"
)
def final_time():
    """
    The final time for the simulation.
    """
    return __data["time"].final_time()


@component.add(
    name="INITIAL TIME", units="Year", comp_type="Constant", comp_subtype="Normal"
)
def initial_time():
    """
    The initial time for the simulation.
    """
    return __data["time"].initial_time()


@component.add(
    name="SAVEPER",
    units="Year",
    limits=(0.0, np.nan),
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"time_step": 1},
)
def saveper():
    """
    The frequency with which output is stored.
    """
    return __data["time"].saveper()


@component.add(
    name="TIME STEP",
    units="Year",
    limits=(0.0, np.nan),
    comp_type="Constant",
    comp_subtype="Normal",
)
def time_step():
    """
    The time step for the simulation.
    """
    return __data["time"].time_step()


#######################################################################
#                           MODEL VARIABLES                           #
#######################################################################


@component.add(
    name="APL market share",
    units="Percentage",
    comp_type="Constant",
    comp_subtype="Normal",
)
def apl_market_share():
    return 0.19


@component.add(
    name="APL users",
    units="Inhabitants",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_apl_users": 1},
    other_deps={
        "_integ_apl_users": {
            "initial": {"apl_market_share": 1, "potential_ecustomers": 1},
            "step": {
                "apl_market_share": 1,
                "potential_ecustomers": 1,
                "apl_market_growth_rate": 1,
            },
        }
    },
)
def apl_users():
    return _integ_apl_users()


_integ_apl_users = Integ(
    lambda: integer(
        apl_market_share() * potential_ecustomers() * apl_market_growth_rate()
    ),
    lambda: integer(apl_market_share() * potential_ecustomers()),
    "_integ_apl_users",
)


@component.add(
    name='"Potential e-customers"',
    units="Inhabitants",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_potential_ecustomers": 1},
    other_deps={
        "_integ_potential_ecustomers": {
            "initial": {"market_size": 1, "eshopper_share": 1},
            "step": {"market_size": 1, "eshopper_share": 1, "eshoppers_growth_rate": 1},
        }
    },
)
def potential_ecustomers():
    return _integ_potential_ecustomers()


_integ_potential_ecustomers = Integ(
    lambda: market_size() * eshopper_share() * eshoppers_growth_rate(),
    lambda: market_size() * eshopper_share(),
    "_integ_potential_ecustomers",
)


@component.add(
    name="Online purchases per year",
    units="Units",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_online_purchases_per_year": 1},
    other_deps={
        "_integ_online_purchases_per_year": {
            "initial": {"initial_online_purchases_per_year": 1},
            "step": {
                "initial_online_purchases_per_year": 1,
                "online_purchase_growth_rate": 1,
            },
        }
    },
)
def online_purchases_per_year():
    return _integ_online_purchases_per_year()


_integ_online_purchases_per_year = Integ(
    lambda: initial_online_purchases_per_year() * online_purchase_growth_rate(),
    lambda: initial_online_purchases_per_year(),
    "_integ_online_purchases_per_year",
)


@component.add(
    name="Initial online purchases per year",
    units="Units",
    comp_type="Constant",
    comp_subtype="Normal",
)
def initial_online_purchases_per_year():
    return 60


@component.add(name='"E-shopper share"', comp_type="Constant", comp_subtype="Normal")
def eshopper_share():
    return 0.63


@component.add(
    name="Number of APLs",
    units="Units",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"number_of_deliveries": 1, "avg_apl_capacity_per_year": 1},
)
def number_of_apls():
    return integer(number_of_deliveries() / avg_apl_capacity_per_year())


@component.add(
    name="APL market growth rate",
    units="Percentage",
    comp_type="Constant",
    comp_subtype="Normal",
)
def apl_market_growth_rate():
    return 0.02


@component.add(
    name='"Avg. APL capacity per year"', comp_type="Constant", comp_subtype="Normal"
)
def avg_apl_capacity_per_year():
    return 48000


@component.add(
    name="Number of deliveries",
    units="Units",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"apl_users": 1, "online_purchases_per_year": 1},
)
def number_of_deliveries():
    return integer(apl_users() * online_purchases_per_year())


@component.add(
    name="Online purchase growth rate",
    units="Percentage",
    comp_type="Constant",
    comp_subtype="Normal",
)
def online_purchase_growth_rate():
    return 0.02


@component.add(
    name="Market Size",
    units="Inhabitants",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_market_size": 1},
    other_deps={
        "_integ_market_size": {
            "initial": {"population": 1},
            "step": {"population": 1, "population_growth_rate": 1},
        }
    },
)
def market_size():
    return _integ_market_size()


_integ_market_size = Integ(
    lambda: population() * population_growth_rate(),
    lambda: population(),
    "_integ_market_size",
)


@component.add(
    name='"E-shoppers growth rate"',
    units="Percentage",
    comp_type="Constant",
    comp_subtype="Normal",
)
def eshoppers_growth_rate():
    return 0.01


@component.add(
    name="Population", units="Inhabitants", comp_type="Constant", comp_subtype="Normal"
)
def population():
    return 138154


@component.add(
    name="Population growth rate",
    units="Percentage",
    comp_type="Constant",
    comp_subtype="Normal",
)
def population_growth_rate():
    return 0.008
