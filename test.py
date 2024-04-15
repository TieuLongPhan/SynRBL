from synrbl import Balancer
from synrbl.SynUtils import normalize_smiles

smiles = "[NH3+]C(CCC=O)C(=O)[O-]>>O=C([O-])C1CCC=N1"
balancer = Balancer()
result = balancer.rebalance(smiles, output_dict=True)
print(result)
