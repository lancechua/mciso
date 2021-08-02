import numpy as np
import pyomo.environ as pyo

_bigM = 1e5


def create_model() -> pyo.AbstractModel:
    """Creates a model with all the constraints for multiple products"""
    model = pyo.AbstractModel()

    # INDICES
    model.n_months = pyo.Param(within=pyo.NonNegativeIntegers)
    model.I = pyo.RangeSet(1, model.n_months)

    model.n_scenarios = pyo.Param(within=pyo.NonNegativeIntegers)
    model.J = pyo.RangeSet(1, model.n_scenarios)

    model.n_skus = pyo.Param(within=pyo.NonNegativeIntegers)
    model.K = pyo.RangeSet(1, model.n_skus)

    # PARAMETERS
    ## demand in # of units for month i in scenario j for sku k
    model.D = pyo.Param(model.I, model.J, model.K, within=pyo.NonNegativeReals)

    ## sell price, unit price per unit
    model.SP = pyo.Param(model.K, within=pyo.NonNegativeReals)
    model.UP = pyo.Param(model.K, within=pyo.NonNegativeReals)

    ## warehousing /  storage cost,multiplier applied to inventory no. of units; e.g. a * cost price
    model.W = pyo.Param(model.K, within=pyo.NonNegativeReals)

    ## out of stock cost, multiplier applied to inventory no. of units; e.g. a * sell price
    model.O = pyo.Param(model.K, within=pyo.NonNegativeReals)

    ## logistics / shipping cost associated with each order from supplier
    model.L = pyo.Param(model.K, within=pyo.NonNegativeReals)

    ## min order quantity  for any month
    model.MOQ = pyo.Param(model.K, within=pyo.NonNegativeReals)

    ## starting inventory
    model.BI = pyo.Param(model.K, within=pyo.NonNegativeReals, default=0)

    ## max inventory value for any month in terms of units; i.e. max capital tied to inventory; e.g. a / cost price
    model.UB = pyo.Param(within=pyo.NonNegativeReals)

    ## min inventory for any month in terms of units; i.e. inventory buffer
    model.LB = pyo.Param(model.K, within=pyo.NonNegativeReals)

    # VARIABLES

    ## no. of units to order for month i for sku k
    model.X = pyo.Var(model.I, model.K, domain=pyo.NonNegativeIntegers)

    ## volume sold for month i for sku k
    model.V = pyo.Var(model.I, model.J, model.K, domain=pyo.NonNegativeReals)

    ## beginning inventory (# of units) for month i for sku k
    model.S = pyo.Var(model.I, model.J, model.K, domain=pyo.NonNegativeReals)

    ## surplus for month i; Di - Xi if  Di < Xi, 0 otherwise
    model.s = pyo.Var(model.I, model.J, model.K, domain=pyo.NonNegativeReals)

    ## deficit for month i; Di - Xi if  Di > Xi, 0 otherwise
    model.d = pyo.Var(model.I, model.J, model.K, domain=pyo.NonNegativeReals)

    ## switch to fulfill minimum order quantity for month i of product k; binary variable
    model.z = pyo.Var(model.I, model.K, domain=pyo.Binary)

    # CONSTRAINTS

    def _const_surplus_deficit(model, i, j, k):
        """volume + surplus = stock"""
        return model.V[i, j, k] + model.s[i, j, k] == model.S[i, j, k]

    model.const_surplus_deficit = pyo.Constraint(
        model.I, model.J, model.K, rule=_const_surplus_deficit
    )

    def _const_stock1(model, j, k):
        """Set stock / beginning inventory for first period"""
        return model.S[1, j, k] == model.X[1, k] + model.BI[k]

    model.const_stock1 = pyo.Constraint(model.J, model.K, rule=_const_stock1)

    def _const_stock(model, i, j, k):
        """Set stock auxiliary variable

        Stock = Order + Previous Surplus
        """
        if pyo.value(i) <= 1:
            return pyo.Constraint.Skip
        else:
            return model.S[i, j, k] == model.X[i, k] + model.s[i - 1, j, k]

    model.const_stock = pyo.Constraint(model.I, model.J, model.K, rule=_const_stock)

    def _const_stock_buffer(model, i, j, k):
        """Min stock constraint (buffer)"""
        return model.S[i, j, k] - model.V[i, j, k] >= model.LB[k]

    def _const_stock_max(model, i, j):
        """Max stock constraint (capital in inventory)"""
        return sum([model.S[i, j, k] * model.UP[k] for k in model.K]) <= model.UB

    model.const_stock_buffer = pyo.Constraint(
        model.I, model.J, model.K, rule=_const_stock_buffer
    )
    model.const_stock_max = pyo.Constraint(model.I, model.J, rule=_const_stock_max)

    def _const_volume_bounds_demand(model, i, j, k):
        """volume + deficit = demand constraint"""
        return model.V[i, j, k] + model.d[i, j, k] == model.D[i, j, k]

    model.const_volume_bounds_demand = pyo.Constraint(
        model.I, model.J, model.K, rule=_const_volume_bounds_demand
    )

    def _const_moq1(model, i, k):
        return model.X[i, k] >= model.MOQ[k] * model.z[i, k]

    def _const_moq2(model, i, k):
        return model.X[i, k] <= _bigM * model.z[i, k]

    model.const_moq1 = pyo.Constraint(model.I, model.K, rule=_const_moq1)
    model.const_moq2 = pyo.Constraint(model.I, model.K, rule=_const_moq2)

    # OBJECTIVE

    def obj_expression(model):
        return sum(
            [
                (model.SP[k] - model.UP[k]) * model.V[i, j, k]  # contribution margin
                - model.W[k] * model.s[i, j, k]  # storage cost
                - model.O[k] * model.d[i, j, k]  # out of stock cost
                for i in model.I
                for j in model.J
                for k in model.K
            ]
        ) / model.n_scenarios - sum(
            [model.L[k] * model.z[i, k] for i in model.I for k in model.K]
        )  # shipping cost

    model.OBJ = pyo.Objective(rule=obj_expression, sense=pyo.maximize)

    return model


def convert_data(
    n_months: int,
    n_scenarios: int,
    D: np.ndarray,
    SP: np.ndarray,
    UP: np.ndarray,
    W: np.ndarray,
    O: np.ndarray,
    L: np.ndarray,
    MOQ: int,
    BI: np.ndarray,
    UB: float,
    LB: np.ndarray,
):
    """
    Convert data

    Parameters
    ----------
    n_months: int
    n_scenarios: int
    D: np.ndarray
        demand scenarios across months; shape (i, j, k)
    SP: np.ndarray
        sell price; shape (k)
    UP: np.ndarray
        unit price; shape (k)
    W: np.ndarray
        storage cost; shape (k)
    O: np.ndarray
        out of stock cost; shape (k)
    L: np.ndarray
        logistics cost per order; shape (k)
    MOQ: int
        shape (k)
        TODO WIP
    BI: np.ndarray
        beginning inventory; shape (k)
    UB: float
        upper bound for inventory value
    LB: np.ndarray
        min stock; shape (k)
    """

    return {
        None: {
            # INDICES TODO: can be computed from D
            "n_months": {None: 5},
            "n_scenarios": {None: 2},
            # PARAMETERS
            "D": {
                (1, 1): 100,
                (2, 1): 120,
                (3, 1): 80,
                (4, 1): 90,
                (5, 1): 100,
                (1, 2): 80,
                (2, 2): 100,
                (3, 2): 150,
                (4, 2): 120,
                (5, 2): 80,
            },
            "CM": {None: 500.0},
            "W": {None: 800 * 0.05},
            "O": {None: 2 * 500},
            "L": {None: 1000},
            "MOQ": {None: 300},
            "BI": {None: 0},
            "UB": {None: 999999},
            "LB": {None: 0},
        },
    }


if __name__ == "__main__":
    data = generate_test_data()
    model = create_model()
    i = model.create_instance(data)

    i.pprint()
