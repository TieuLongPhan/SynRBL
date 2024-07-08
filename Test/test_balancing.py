import pytest

from synrbl import Balancer


def test_e2e_1():
    n = 100
    reactant = "Br" + n * "[Si](C)(C)O" + "[Si](C)(C)Br"
    product = "O" + n * "[Si](C)(C)O" + "[Si](C)(C)O"
    reaction = "{}>>{}".format(reactant, product)
    exp_result = "{}.{}>>{}.{}".format(reactant, "O.O", product, "Br.Br")

    balancer = Balancer()
    balancer.rb_method.n_jobs = 1
    balancer.rb_validator.n_jobs = 1
    balancer.mcs_validator.n_jobs = 1
    balancer.input_validator.n_jobs = 1

    result = balancer.rebalance(reaction)
    assert exp_result == result[0]


@pytest.mark.parametrize(
    "smiles,exp_smiles",
    [
        ["CC(=O)C>>CC(O)C", "CC(=O)C.[H][H]>>CC(O)C"],
        [
            "CCO.[O]>>CC=O",
            "CCO.O.O=[Cr](Cl)(-[O-])=O.c1cc[nH+]cc1.O>>"
            + "CC=O.O.O.O=[Cr](O)O.c1cc[nH+]cc1.[Cl-]",
        ],
    ],
)
def test_post_process(smiles, exp_smiles):
    blncer = Balancer(n_jobs=1)
    result = blncer.rebalance(smiles)
    assert exp_smiles == result[0]


def test_standardizing_of_merged_compounds():
    blncer = Balancer(n_jobs=1)
    smiles = "CC(=O)OC=C>>CC(=O)O"
    exp_result = "CC(=O)OC=C.O>>CC(=O)O.CC=O"
    result = blncer.rebalance(smiles)
    assert exp_result == result[0]
