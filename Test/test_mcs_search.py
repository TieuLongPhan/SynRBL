import pytest

from synrbl.mcs_search import MCSSearch
from synrbl.SynProcessor import CheckCarbonBalance

ID_COL = "id"
SOLVED_COL = "solved"
MCS_DATA_COL = "mcs"
REACTION_COL = "reaction"
REACTANTS_COL = "reactants"
PRODUCTS_COL = "products"
C_BALANCE_CHECK_COL = "carbon_balance_check"


def _init_mcs():
    return MCSSearch(ID_COL, solved_col=SOLVED_COL, mcs_data_col=MCS_DATA_COL)


def _add(data, reaction, solved=False):
    r_token = reaction.split(">>")
    entry = {
        ID_COL: len(data),
        REACTION_COL: reaction,
        SOLVED_COL: solved,
        REACTANTS_COL: r_token[0],
        PRODUCTS_COL: r_token[1],
    }
    check = CheckCarbonBalance(
        [entry], rsmi_col=REACTION_COL, symbol=">>", atom_type="C"
    )
    entry[C_BALANCE_CHECK_COL] = check.check_carbon_balance()[0]["carbon_balance_check"]
    data.append(entry)


def test_simple_mcs():
    mcs = _init_mcs()
    data = []
    _add(data, "COC(C)=O>>OC(C)=O")

    results = mcs.find(data)

    assert 1 == len(results)
    result = results[0]
    assert MCS_DATA_COL in result.keys()
    result = result[MCS_DATA_COL]

    assert 1 == len(result["smiles"])
    assert "C" == result["smiles"][0]

    assert 1 == len(result["boundary_atoms_products"])
    assert 1 == len(result["boundary_atoms_products"][0])
    assert {"C": 0} == result["boundary_atoms_products"][0][0]

    assert 1 == len(result["nearest_neighbor_products"])
    assert 1 == len(result["nearest_neighbor_products"][0])
    assert {"O": 1} == result["nearest_neighbor_products"][0][0]

    # assert 0 == len(result["issue"])
    # assert result["Certainty"] is True

    assert 1 == len(result["sorted_reactants"])
    assert "COC(C)=O" == result["sorted_reactants"][0]

    assert 1 == len(result["mcs_results"])
    assert "[#8]-&!@[#6](-&!@[#6])=&!@[#8]" == result["mcs_results"][0]


@pytest.mark.parametrize(
    "rxn, exp_smiles, exp_ba, exp_nn, exp_issue, exp_sr, exp_mcs",
    [
        (
            (
                "Nc1nc2c(ncn2C2OC(COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])C"
                + "(O)C2O)c(=O)[nH]1.O>>Nc1nc2c(ncn2C2OC(COP(=O)([O-])[O-])C"
                + "(O)C2O)c(=O)[nH]1"
            ),
            ["O=[PH]([O-])OP(=O)([O-])[O-]", None],
            [[{"P": 1}], None],
            [[{"O": 16}], None],
            "",
            [
                (
                    "Nc1nc2c(ncn2C2OC(COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])"
                    + "[O-])C(O)C2O)c(=O)[nH]1"
                ),
                "O",
            ],
            [
                (
                    "[#7]-&!@[#6]1:&@[#7]:&@[#6]2:&@[#6](:&@[#6](:&@[#7]:&@1)"
                    + "=&!@[#8]):&@[#7]:&@[#6]:&@[#7]:&@2-&!@[#6]1-&@[#8]-&@"
                    + "[#6](-&@[#6](-&@[#6]-&@1-&!@[#8])-&!@[#8])-&!@[#6]-&!"
                    + "@[#8]-&!@[#15](=&!@[#8])(-&!@[#8])-&!@[#8]"
                ),
                "",
            ],
        ),
    ],
)
def test_two_boundaries(rxn, exp_smiles, exp_ba, exp_nn, exp_issue, exp_sr, exp_mcs):
    mcs = _init_mcs()
    data = []
    _add(data, rxn)

    results = mcs.find(data)

    result = results[0][MCS_DATA_COL]
    assert exp_smiles == result["smiles"]
    assert exp_ba == result["boundary_atoms_products"]
    assert exp_nn == result["nearest_neighbor_products"]
    assert exp_issue == result["issue"]
    assert exp_sr == result["sorted_reactants"]
    assert exp_mcs == result["mcs_results"]
