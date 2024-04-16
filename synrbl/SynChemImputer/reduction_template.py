from typing import List, Dict, Union
from rdkit import Chem
from fgutils import FGQuery
import rdkit.RDLogger as RDLogger

RDLogger.DisableLog("rdApp.*")


class ReductionTemplate:
    @staticmethod
    def count_radical_isolated_hydrogens(smiles):

        mol = Chem.MolFromSmiles(smiles)

        # Initialize count for isolated radical hydrogens
        hydrogen_count = 0

        # Iterate over all atoms in the molecule
        for atom in mol.GetAtoms():
            # Check if the atom is a hydrogen atom
            if atom.GetAtomicNum() == 1:
                # Check if the hydrogen atom is isolated (has no neighbors)
                if len(atom.GetNeighbors()) == 0:
                    # Check if the hydrogen is a radical (has unpaired electrons)
                    if atom.GetNumRadicalElectrons() > 0:
                        hydrogen_count += 1

        return hydrogen_count

    @staticmethod
    def find_reactive_functional_groups(reaction_smiles: str) -> List[str]:
        query = FGQuery(use_smiles=True)
        reactant, product = reaction_smiles.split(">>")
        fg_reactant = query.get(reactant)
        fg_product = query.get(product)
        fg_reactant = [value[0] for value in fg_reactant]
        fg_product = [value[0] for value in fg_product]
        return [fg for fg in fg_reactant if fg not in fg_product]

    @staticmethod
    def process_template(
        reaction_smiles: str,
        neutralize: bool = False,
        all_templates: Dict = None,
        template: str = None,
    ) -> str:
        if template is None:
            selected_template = all_templates[
                0
            ]  # Default to template_1 if none provided
        else:
            selected_template = all_templates[template]
        reactants, products = reaction_smiles.split(">>")
        hydrogen_count = ReductionTemplate.count_radical_hydrogens(reactants)
        if hydrogen_count % 2 != 0:
            return reaction_smiles
        hh_count = hydrogen_count // 2
        reactant_list = [x for x in reactants.split(".") if x != "[H]"]
        product_list = products.split(".")
        template_type = "neutral" if neutralize else "ion"
        for _ in range(hh_count):
            reactant_list.extend(selected_template[template_type]["reactants"])
            product_list.extend(selected_template[template_type]["products"])
        updated_reactants = ".".join(reactant_list)
        updated_products = ".".join(product_list)
        return f"{updated_reactants}>>{updated_products}"

    @classmethod
    def reduction_template(
        cls,
        reaction_smiles: str,
        compound_template: Dict,
        all_templates: Dict = None,
        return_all: bool = False,
    ) -> Union[str, List[str]]:
        try:
            fg_reactive = cls.find_reactive_functional_groups(reaction_smiles)
            if len(fg_reactive) == 0:
                fg_reactive = ["other"]
            processed_smiles = []
            for group, templates in compound_template.items():
                if group in fg_reactive:
                    # print(f"Processing {group} with template {templates}")
                    processed_smiles.extend(
                        [
                            cls.process_template(
                                reaction_smiles,
                                neutralize=False,
                                all_templates=all_templates,
                                template=tpl,
                            )
                            for tpl in templates
                        ]
                    )
            return (
                processed_smiles
                if return_all
                else (processed_smiles[0] if processed_smiles else None)
            )
        except Exception as e:
            print(e)
            return [reaction_smiles]
