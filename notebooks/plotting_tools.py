def to_superscript(number):
    superscript_map = {
    "0": "\u2070",
    "1": "\u00B9",
    "2": "\u00B2",
    "3": "\u00B3",
    "4": "\u2074",
    "5": "\u2075",
    "6": "\u2076",
    "7": "\u2077",
    "8": "\u2078",
    "9": "\u2079"}
    return ''.join(superscript_map.get(char, char) for char in str(number))
    

def write_powers(power):
    base = 10
    if power < 0:
        # For negative powers, use the superscript minus sign (⁻) followed by the power
        power_str = "\u207B" + to_superscript(abs(power))
    else:
        power_str = to_superscript(power)
    return f"{base}{power_str}"