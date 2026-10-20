-- Daily sales must add up to the offer's units sold (allowing for offers whose daily rows were
-- quarantined); a mismatch means two systems disagree about what happened.
with d as (select campaign_id, article_id, sum(units_sold) as daily_units from {{ ref('stg_daily_sales') }} group by 1, 2)
select o.campaign_id, o.article_id, o.units_sold, d.daily_units
from {{ ref('stg_offers') }} o
join d using (campaign_id, article_id)
where d.daily_units <> o.units_sold
