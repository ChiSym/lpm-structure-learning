import polars as pl

from lpm_query import QueryInference
from lpm_structure_learning import learn_structure

def test_learn_structure_smoke():
    n = 100
    column = ["a"] * n + ["b"] * n
    df = pl.DataFrame({"foo":column, "bar":column, "qux": column})
    model_spec, schema = learn_structure(df, max_clusters=5)
    qi = QueryInference(model_spec, schema)
    assert qi.logpdf({"foo":"a", "bar":"a"}) > qi.logpdf({"foo":"a", "bar":"b"})
