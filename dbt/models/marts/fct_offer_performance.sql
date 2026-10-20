-- One row per article per campaign with the commercial outcome. Contribution uses the same cost
-- assumptions as the planner (dbt vars), so analytics and optimisation agree on what "profit" is.
with o as (select * from {{ ref('stg_offers') }}),
c as (select campaign_id, kind, theme, start_date, duration_days from {{ ref('stg_campaigns') }}),
a as (select article_id, brand_id, category, season, gender from {{ ref('stg_articles') }})
select
    o.campaign_id,
    o.article_id,
    a.brand_id,
    a.category,
    a.gender,
    c.kind as campaign_kind,
    c.start_date,
    cast(date_trunc('week', c.start_date) as date) as start_week,
    o.price,
    o.discount,
    o.units_bought,
    o.units_sold,
    o.units_returned,
    o.sold_out,
    {{ safe_divide('o.units_sold', 'o.units_bought') }} as sell_through,
    (o.units_sold - o.units_returned) * o.price as net_revenue,
    (o.units_sold - o.units_returned) * o.price
        - o.units_bought * o.unit_cost
        - o.units_sold * {{ var('fulfilment_cost_per_unit') }}
        - o.units_returned * {{ var('return_handling_cost_per_unit') }}
        + (o.units_bought - o.units_sold + o.units_returned) * o.unit_cost * {{ var('salvage_fraction') }} as contribution
from o
join c using (campaign_id)
join a using (article_id)
