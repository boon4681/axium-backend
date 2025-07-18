from axium.template import AxiumTemplate


class PositivePlusNegative(AxiumTemplate):
    id = "PosNeg"
    category = "math"

    input = {
        "a": "axium.int",
        "b": "axium.int"
    }

    output = {
        "result": "axium.int"
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


class LoadCSV(AxiumTemplate):
    name = "LoadCSV"
    id = "LoadCSV"
    category = "basic"

    input = {
    }

    output = {
        "result": "axium.dataframe"
    }

    @classmethod
    def run(cls, a, b):
        return a + b


class FillEmpty(AxiumTemplate):
    name = "FillEmpty"
    id = "FillEmpty"
    category = "basic"

    input = {
        "data": "axium.dataframe",
    }

    output = {
        "result": "axium.dataframe"
    }

    @classmethod
    def run(cls, data):
        return data


class GetColumn(AxiumTemplate):
    name = "GetColumn"
    id = "GetColumn"
    category = "basic"

    input = {
        "data": "axium.dataframe",
    }

    output = {
        "result": "axium.dataframe"
    }

    @classmethod
    def run(cls, data):
        return data


class SplitTrainingSet:
    name = "SplitTrainingSet"
    id = "SplitTrainingSet"
    category = "basic"

    input = {
        "train_data": "axium.dataframe",
        "train_target": "axium.dataframe",
    }

    output = {
        "train_input": "axium.dataframe",
        "train_target": "axium.dataframe",
        "test_input": "axium.dataframe",
        "test_target": "axium.dataframe",
    }

    @classmethod
    def run(cls, data):
        return data
