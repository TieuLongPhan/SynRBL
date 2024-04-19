from synrbl.SynProcessor.rsmi_both_side_process import BothSideReact


def test_charge_fix():
    both_side = BothSideReact(
        [{"H": 2}], [{"H": 1, "Q": -1}], ["Both"], [{"H": 1, "Q": -1}]
    )
    diff_formula, unbalance = both_side.fit()
    assert "Products" == unbalance[0]
    assert [{"H": 1, "Q": 1}] == diff_formula


def test_leave_as_is():
    both_side = BothSideReact(
        [{"H": 2, "O": 1}], [{"H": 1, "Q": -1}], ["Both"], [{"H": 1, "O": 1, "Q": -1}]
    )
    diff_formula, unbalance = both_side.fit()
    assert "Both" == unbalance[0]
    assert [{"H": 1, "O": 1, "Q": 1}] == diff_formula
