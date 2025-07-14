WEB_DIRECTORY = "./web"


class Sum:
    __FUNCTION__ = "sum"
    __CATEGORY__ = "math"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "ports": [
                ("a", "AXIUM.FLOAT", {"default": 0}),
                ("b", "AXIUM.FLOAT", {"default": 0})
            ],
            "parameters": [
                ("b", "AXIUM.SELECTION", {
                    "default": "",
                    "options": [
                        "a",
                        "b",
                        "c",
                        "d"
                    ]
                })
            ]
        }

    @classmethod
    def OUTPUT_TYPES(s):
        return [
            ("result", "AXIUM.FLOAT")
        ]

    def sum(inputs):
        a = inputs[0]
        b = inputs[1]
        return [a+b]


class Abs:
    __FUNCTION__ = "abs"
    __CATEGORY__ = "math"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "ports": [
                ("a", "AXIUM.FLOAT", {"default": 0, "required": True})
            ],
            "parameters": []
        }

    @classmethod
    def OUTPUT_TYPES(s):
        return [
            ("result", "AXIUM.FLOAT")
        ]

    @classmethod
    def VALIDATION(s):
        return [

        ]

    def abs(inputs):
        return abs(inputs[0])


class N:
    pass


EXPORT_NODES = {
    "sum": Sum,
    "abs": Abs,
    "n": N
}
