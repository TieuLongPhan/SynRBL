from SynRBL import SynRBL

reactions = ["CC=O>>CC=O", "CCO>>CC=O", "CCOC(=O)C>>CCO"]
synrbl = SynRBL()
result = synrbl.rebalance(reactions)
print(result)
