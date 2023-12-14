from rdkit.Chem import rdmolops


class NoMoreHError(Exception):
    """
    Exception if an atom has no more Hydrogen atoms that can be removed.
    """

    def __init__(self, atom):
        """
        Exception if an atom has no more Hydrogen atoms that can be removed.
        """
        super().__init__(
            "Could not remove any more neighboring H atoms from {}.".format(atom)
        )


class Property:
    """
    Generic property for dynamic rule configuration.

    Attributes:
        neg_values (list): List of forbidden values. Forbidden value check is
            skipped if this list is empty.
        pos_values (list): List of valid values. If this list is empty, all
            values are excepted.
        allow_none (bool): If a value of None is allowed and checked as valid.
    """

    def __init__(
        self, config: str | list[str] | None = None, dtype=str, allow_none=False
    ):
        """
        Generic property for dynamic rule configuration.

        Arguments:
            config (str | list[str]): Configuration for this property. This is
                a list of acceptable and forbiden values.
                e.g.: ['A', 'B'] -> check is true if value is A or B
                      ['!C', '!D'] -> check is true if value is not C and not D
            dtype (optional): Datatype of the property. Must implement
                conversion from string as constructor (e.g.: int).
            allow_none (bool): Flag if a check with value None is valid or not.
        """
        self.__dtype = dtype
        self.allow_none = allow_none
        self.neg_values = []
        self.pos_values = []
        if config is not None:
            if not isinstance(config, list):
                config = [config]
            for i, s in enumerate(config):
                config[i] = str(s)
            for item in config:
                if not isinstance(item, str):
                    raise ValueError("value must be str or a list of strings.")
                if len(item) > 0 and item[0] == "!":
                    self.neg_values.append(dtype(item[1:]))
                else:
                    self.pos_values.append(dtype(item))

    def check(self, value):
        """
        Check if the property is true for the given value.

        Arguments:
            value: The value to check. It must be of the same datatype as
                specified in the constructor.

        Returns:
            bool: True if the value fulfills the property, false otherwise.
        """
        if self.allow_none and value is None:
            return True
        if not isinstance(value, self.__dtype):
            raise ValueError("value must be of type '{}'.".format(self.__dtype))
        if len(self.pos_values) > 0:
            found = False
            for pv in self.pos_values:
                if pv == value:
                    found = True
                    break
            if not found:
                return False
        if len(self.neg_values) > 0:
            for nv in self.neg_values:
                if nv == value:
                    return False
        return True


class AtomCondition:
    """
    Atom condition class to check if a rule is applicable to a specific
    molecule. Property configs (atom, rad_e, ...) can be prefixed with '!'
    to negate the check. See SynRBL.SynMCS.mol_merge.Property for more
    information.

    Example:
        Check if atom is Carbon and has Oxygen or Nitrogen as neighbor.
        >>> cond = AtomCondition(atom=['C'], neighbors=['O', 'N'])
        >>> mol = rdkit.Chem.MolFromSmiles('CO')
        >>> cond.check(mol.GetAtomFromIdx(0), neighbor='O')
        True

    Attributes:
        atom (SynRBL.SynMCS.mol_merge.Property): Atom property
        rad_e (SynRBL.SynMCS.mol_merge.Property): Radical electron property
        charge (SynRBL.SynMCS.mol_merge.Property): Charge porperty
        neighbors (SynRBL.SynMCS.mol_merge.Property): Neighbors property
    """

    def __init__(self, atom=None, rad_e=None, charge=None, neighbors=None, **kwargs):
        """
        Atom condition class to check if a rule is applicable to a specific
        molecule. Property configs (atom, rad_e, ...) can be prefixed with '!'
        to negate the check. See SynRBL.SynMCS.mol_merge.Property for more
        information.

        Arguments:
            atom: Atom property configuration.
            rad_e: Radical electron property configuration.
            charge: Charge porperty configuration.
            neighbors: Neighbors property configuration.
        """
        atom = kwargs.get("atom", atom)
        rad_e = kwargs.get("rad_e", rad_e)
        charge = kwargs.get("charge", charge)
        neighbors = kwargs.get("neighbors", neighbors)

        self.atom = Property(atom)
        self.rad_e = Property(rad_e, dtype=int)
        self.charge = Property(charge, dtype=int)
        self.neighbors = Property(neighbors, allow_none=True)

    def check(self, atom, neighbor=None):
        """
        Check if the atom meets the condition.

        Arguments:
            atom (rdkit.Chem.Atom): Atom the condition should be checked for.
            neighbor (str): A boundary atom.

        Returns:
            bool: True if the atom fulfills the condition, false otherwise.
        """
        return all(
            [
                self.atom.check(atom.GetSymbol()),
                self.rad_e.check(atom.GetNumRadicalElectrons()),
                self.charge.check(atom.GetFormalCharge()),
                self.neighbors.check(neighbor),
            ]
        )


class ActionSet:
    """
    Class to configure a set of actions to perform on a compound.
    """

    def __init__(self, actions=None):
        self.__actions = actions
        if actions is not None and not isinstance(actions, list):
            self.__actions = [actions]

    @staticmethod
    def apply(action_name, mol, atom):
        """
        Apply an action to an atom.

        Arguments:
            action_name (str): The name of the action to apply.
            mol (rdkit.Chem.Mol): The molecule object where the action should
                be applied.
            atom (rdkit.Chem.Atom): The atom where the action should be
                applied.
        """
        if action_name == "removeH":
            found_H = False
            for n_atom in atom.GetNeighbors():
                if n_atom.GetAtomicNum() == 1:
                    mol.RemoveAtom(n_atom.GetIdx())
                    found_H = True
                    break
            if not found_H:
                raise NoMoreHError(atom.GetSymbol())
        elif action_name == "removeRadE":
            atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() - 1)
        elif action_name == "addRadE":
            atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() + 1)
        elif action_name == "chargeNeg":
            atom.SetFormalCharge(atom.GetFormalCharge() - 1)
        elif action_name == "chargePos":
            atom.SetFormalCharge(atom.GetFormalCharge() + 1)
        else:
            raise NotImplementedError(
                ("Action '{}' is not implemented.").format(action_name)
            )

    def __call__(self, mol, atom):
        """
        Apply the configured actions to the atom in the molecule.

        Arguments:
            mol (rdkit.Chem.Mol): The molecule object where the action should
                be applied.
            atom (rdkit.Chem.Atom): The atom where the action should be
                applied.
        """
        if self.__actions is None:
            return
        for a in self.__actions:
            ActionSet.apply(a, mol, atom)
