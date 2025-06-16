WEB_DIRECTORY = "./web"


class Sum:
    __FUNCTION__ = "sum"
    __CATEGORY__ = "math"

    @classmethod
    def INPUT_TYPES(s):
        return [
            ("a", "AXIUM.FLOAT", {"default": 0, "required": True}),
            ("b", "AXIUM.FLOAT", {"default": 0, "required": True})
        ]

    @classmethod
    def OUTPUT_TYPES(s):
        return [
            ("result", "FLOAT")
        ]

    def sum(inputs):
        a = inputs[0]
        b = inputs[1]
        return [a+b]


EXPORT_NODES = {
    "sum": Sum
}
