from .structure import Boundary


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
        self,
        config: str | list[str] | None = None,
        dtype: type[int | str] = str,
        allow_none=False,
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
            for item in config:
                if not isinstance(item, str):
                    raise ValueError(
                        "Property configuration must be of type str or list[str]."
                    )
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


class BoundaryCondition:
    """
    Atom condition class to check if a rule is applicable to a specific
    molecule. Property configs can be prefixed with '!' to negate the check.
    See SynRBL.SynMCS.rule_formation.Property for more information.

    Example:
        Check if atom is Carbon and has Oxygen or Nitrogen as neighbor.
        >>> cond = AtomCondition(atom=['C'], neighbors=['O', 'N'])
        >>> mol = rdkit.Chem.rdmolfiles.MolFromSmiles('CO')
        >>> cond.check(mol.GetAtomFromIdx(0), neighbor='O')
        True

    Attributes:
        atom (SynRBL.SynMCS.rule_formation.Property): Atom property
        neighbors (SynRBL.SynMCS.rule_formation.Property): Neighbors property
    """

    def __init__(self, atom=None, neighbors=None, **kwargs):
        """
        Atom condition class to check if a rule is applicable to a specific
        molecule. Property configs can be prefixed with '!' to negate the
        check. See SynRBL.SynMCS.rule_formation.Property for more information.

        Arguments:
            atom: Atom property configuration.
            neighbors: Neighbors property configuration.
        """
        atom = kwargs.get("atom", atom)
        neighbors = kwargs.get("neighbors", neighbors)

        self.atom = Property(atom)
        self.neighbors = Property(neighbors, allow_none=True)

    def check(self, boundary: Boundary):
        """
        Check if the boundary meets the condition.

        Arguments:
            boundary (SynRBS.SynMCS.structure.Boundary): Boundary the
                condition should be checked for.

        Returns:
            bool: True if the boundary fulfills the condition, false otherwise.
        """
        return all(
            [
                self.atom.check(boundary.symbol),
                self.neighbors.check(boundary.neighbor_symbol),
            ]
        )
