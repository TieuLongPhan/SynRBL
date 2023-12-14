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

    def __init__(self, value: str | list[str], dtype=str, allow_none=False):
        """
        Generic property for dynamic rule configuration.

        Arguments:
            value (str | list[str]): Configuration for this property. This is
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
        if value is not None:
            if not isinstance(value, list):
                value = [value]
            for i, s in enumerate(value):
                value[i] = str(s)
            for item in value:
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
