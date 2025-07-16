from axium.template import AxiumTemplate

class PositivePlusNegative(AxiumTemplate):
    id = "PosNeg"

    input = {
        "a": int,
        "b": int
    }

    output = {
        "result": int
    }

    @classmethod
    def validate_input(cls, a, b):
        return {
            "a": (a >= 0, "Value 'a' must be positive"),
            "b": (b < 0,  "Value 'b' must be negative")
        }

    @classmethod
    def run(cls, a, b):
        return a + b