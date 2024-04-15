from synrbl import Balancer
from synrbl.SynUtils import normalize_smiles

smiles = "Nc1nc2c(ncn2C2OC(COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])C(O)C2O)c(=O)[nH]1.O>>Nc1nc2c(ncn2C2OC(COP(=O)([O-])[O-])C(O)C2O)c(=O)[nH]1"
print(normalize_smiles(smiles))
balancer = Balancer()
result = balancer.rebalance(smiles, output_dict=True)
print(result)
