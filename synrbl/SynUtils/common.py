def update_reactants_and_products(
    reactions,
    reaction_col,
    reactants_col="reactants",
    products_col="products",
    symbol=">>",
):
    for reaction in reactions:
        tokens = reaction[reaction_col].split(symbol)
        reaction[reactants_col] = tokens[0]
        reaction[products_col] = tokens[1]
