class converter():
    def convert(unit_from, unit_to, value):
        if unit_from == "m" and unit_to == "km":
            return value / 1000
        elif unit_from == "km" and unit_to == "m":
            return value * 1000
        elif unit_from == "cm" and unit_to == "m":
            return value / 100
        elif unit_from == "m" and unit_to == "cm":
            return value * 100
        elif unit_from == "mm" and unit_to == "cm":
            return value / 10
        elif unit_from == "cm" and unit_to == "mm":
            return value * 10
        elif unit_from == "g" and unit_to == "kg":
            return value / 1000
        elif unit_from == "kg" and unit_to == "g":
            return value * 1000
        elif unit_from == "c" and unit_to == "f":
            return (value * 9/5) + 32
        elif unit_from == "f" and unit_to == "c":
            return (value - 32) * 5/9
        elif unit_from == "sec" and unit_to == "min":
            return value / 60
        elif unit_from == "min" and unit_to == "sec":
            return value * 60
        elif unit_from == "hour" and unit_to == "min":
            return value * 60
        elif unit_from == "min" and unit_to == "hour":
            return value / 60
        elif unit_from == "litre" and unit_to == "ml":
            return value * 1000
        elif unit_from == "ml" and unit_to == "litre":
            return value / 1000
        elif unit_from == "m^3" and unit_to == "litre":
            return value * 1000
        elif unit_from == "litre" and unit_to == "m^3":
            return value / 1000
        else:
            return "Invalid conversion"
        
