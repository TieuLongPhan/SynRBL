from synrbl import Balancer


def test_e2e_1():
    n = 100
    reactant = "[Br]" + n * "[Si](C)(C)O" + "[Si](C)(C)[Br]"
    product = "O" + n * "[Si](C)(C)O" + "[Si](C)(C)O"
    reaction = "{}>>{}".format(reactant, product)
    exp_result = "{}.{}>>{}.{}".format(reactant, "O.O", product, "Br.Br")
    balancer = Balancer()
    result = balancer.rebalance(reaction)
    assert exp_result == result[0]
