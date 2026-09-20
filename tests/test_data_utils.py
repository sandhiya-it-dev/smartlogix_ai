import math
from src.data_utils import parse_bool, parse_number, parse_weight_kg

def test_numbers_and_units():
    assert parse_number("₹79,382.91")==79382.91
    assert parse_weight_kg("87540 g")==87.54
    assert parse_weight_kg("16.89KG")==16.89

def test_boolean_variants():
    assert parse_bool("Yes")==1
    assert parse_bool("N")==0
    assert math.isnan(parse_bool("unknown"))

