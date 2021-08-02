import pyomo.environ as pyo

_bigM = 1e5


def create_model() -> pyo.AbstractModel:
    """Create model for one product only

    Note: This is mainly used for debugging / initial proof of concept.
    """
    model = pyo.AbstractModel()

    # INDICES
    model.n_months = pyo.Param(within=pyo.NonNegativeIntegers)
    model.I = pyo.RangeSet(1, model.n_months)

    model.n_scenarios = pyo.Param(within=pyo.NonNegativeIntegers)
    model.J = pyo.RangeSet(1, model.n_scenarios)

    # PARAMETERS
    ## demand in # of units for month i in scenario j
    model.D = pyo.Param(model.I, model.J, within=pyo.NonNegativeIntegers)

    ## contribution margin per unit
    model.CM = pyo.Param(within=pyo.NonNegativeReals)

    ## warehousing /  storage cost,multiplier applied to inventory no. of units; e.g. a * cost price
    model.W = pyo.Param(within=pyo.NonNegativeReals)

    ## out of stock cost, multiplier applied to inventory no. of units; e.g. a * sell price
    model.O = pyo.Param(within=pyo.NonNegativeReals)

    ## logistics / shipping cost associated with each order from supplier
    model.L = pyo.Param(within=pyo.NonNegativeReals)

    ## min order quantity  for any month
    model.MOQ = pyo.Param(within=pyo.NonNegativeIntegers)

    ## starting inventory
    model.BI = pyo.Param(within=pyo.NonNegativeIntegers, default=0)

    ## max inventory for any month in terms of units; i.e. max capital tied to inventory; e.g. a / cost price
    model.UB = pyo.Param(within=pyo.NonNegativeReals)

    ## min inventory for any month in terms of units; i.e. inventory buffer
    model.LB = pyo.Param(within=pyo.NonNegativeReals)

    # VARIABLES

    ## no. of units to order for month i
    model.X = pyo.Var(model.I, domain=pyo.NonNegativeIntegers)

    ## volume sold for month i
    model.V = pyo.Var(model.I, model.J, domain=pyo.NonNegativeIntegers)

    ## beginning inventory (# of units) for month i
    model.S = pyo.Var(model.I, model.J, domain=pyo.NonNegativeIntegers)

    ## surplus for month i; Di - Xi if  Di < Xi, 0 otherwise
    model.s = pyo.Var(model.I, model.J, domain=pyo.NonNegativeIntegers)

    ## deficit for month i; Di - Xi if  Di > Xi, 0 otherwise
    model.d = pyo.Var(model.I, model.J, domain=pyo.NonNegativeIntegers)

    ## switch to fulfill minimum order quantity for month i; binary variable
    model.z = pyo.Var(model.I, domain=pyo.Binary)

    # CONSTRAINTS

    def _const_surplus_deficit(model, i, j):
        """Set surplus and deficit auxiliary variables"""
        return model.V[i, j] + model.s[i, j] - model.d[i, j] == model.D[i, j]

    model.const_surplus_deficit = pyo.Constraint(
        model.I, model.J, rule=_const_surplus_deficit
    )

    def _const_stock1(model, j):
        """Set stock / beginning inventory for first period"""
        return model.S[1, j] == model.X[1] + model.BI

    model.const_stock1 = pyo.Constraint(model.J, rule=_const_stock1)

    def _const_stock(model, i, j):
        """Set stock auxiliary variable"""
        if pyo.value(i) <= 1:
            return pyo.Constraint.Skip
        else:
            return model.S[i, j] == model.X[i] + model.S[i - 1, j] - model.V[i - 1, j]

    model.const_stock = pyo.Constraint(model.I, model.J, rule=_const_stock)

    def _const_stock_bounds(model, i, j):
        """Min (buffer) / Max (capital in inventory) stock constraint"""
        return (model.LB, model.S[i, j], model.UB)

    model.const_stock_bounds = pyo.Constraint(
        model.I, model.J, rule=_const_stock_bounds
    )

    def _const_volume_bounds(model, i, j):
        """volume <= stock constraint"""
        return model.V[i, j] <= model.S[i, j]

    model.const_volume_bounds = pyo.Constraint(
        model.I, model.J, rule=_const_volume_bounds
    )

    def _const_moq1(model, i):
        return model.X[i] >= model.MOQ * model.z[i]

    def _const_moq2(model, i):
        return model.X[i] <= _bigM * model.z[i]

    model.const_moq1 = pyo.Constraint(model.I, rule=_const_moq1)
    model.const_moq2 = pyo.Constraint(model.I, rule=_const_moq2)

    # OBJECTIVE

    def obj_expression(model):
        return sum(
            [
                model.CM * model.V[i, j]  # contribution margin
                - model.W * model.s[i, j]  # storage cost
                - model.O * model.d[i, j]  # out of stock cost
                for i in model.I
                for j in model.J
            ]
        ) / model.n_scenarios - sum(
            [model.L * model.z[i] for i in model.I]
        )  # shipping cost

    model.OBJ = pyo.Objective(rule=obj_expression, sense=pyo.maximize)

    return model


def generate_test_data():
    """Generate data for debugging"""
    return {
        None: {
            # INDICES
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
