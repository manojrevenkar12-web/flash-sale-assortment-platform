import pandas as pd
import pytest

from fsap.contracts import model as cm
from fsap.contracts.pandas_engine import enforce as enforce_pd
from tests.conftest import has_java

pytestmark = [pytest.mark.spark, pytest.mark.skipif(not has_java(), reason="needs Java for Spark")]


@pytest.fixture(scope="module")
def spark():
    from fsap.pipelines.spark_jobs import spark_session

    return spark_session("fsap-tests")


def test_spark_and_pandas_engines_agree(spark):
    from fsap.contracts.spark_engine import enforce as enforce_sp

    c = cm.load("campaign_offers")
    base = {
        "campaign_id": "C000001",
        "article_id": "B001-0001",
        "price": "20.0",
        "rrp": "50.0",
        "discount": "0.6",
        "unit_cost": "8.0",
        "units_bought": "40",
        "units_sold": "30",
        "units_returned": "5",
        "sold_out_day": "-1",
    }
    variants = [
        {},
        {"units_sold": "-3"},
        {"units_sold": "50"},
        {"price": None},
        {"article_id": "B999-9999"},
        {"price": "80"},
        {"units_bought": "12.5"},
        {"units_bought": "12.0"},
        {"campaign_id": "X1"},
        {"discount": "abc"},
        {},
    ]
    rows = [{**base, **v, "campaign_id": v.get("campaign_id", f"C{i:06d}")} for i, v in enumerate(variants)]
    rows[-1]["campaign_id"] = rows[0]["campaign_id"]  # duplicate key
    df = pd.DataFrame(rows)
    refs_pd = {"articles": {"B001-0001"}}
    v_pd, q_pd, r_pd = enforce_pd(df, c, refs_pd, raise_on_breach=False)
    sdf = spark.createDataFrame(df.astype(object).where(df.notna(), None))
    refs_sp = {"articles": spark.createDataFrame(pd.DataFrame({"article_id": ["B001-0001"]}))}
    v_sp, q_sp, r_sp = enforce_sp(sdf, c, refs_sp, raise_on_breach=False)
    assert r_pd["reasons"] == r_sp["reasons"]
    assert sorted(q_pd["campaign_id"]) == sorted(r.campaign_id for r in q_sp.collect())
    assert sorted(v_pd["campaign_id"]) == sorted(r.campaign_id for r in v_sp.collect())
