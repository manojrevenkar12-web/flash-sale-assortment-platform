select
    f.campaign_id,
    min(f.campaign_kind) as campaign_kind,
    min(f.start_date) as start_date,
    count(*) as articles,
    sum(f.units_bought) as units_bought,
    sum(f.units_sold) as units_sold,
    {{ safe_divide('sum(f.units_sold)', 'sum(f.units_bought)') }} as sell_through,
    avg(case when f.sold_out then 1.0 else 0.0 end) as sold_out_share,
    sum(f.net_revenue) as net_revenue,
    sum(f.contribution) as contribution,
    {{ safe_divide('sum(f.contribution)', 'sum(f.net_revenue)') }} as contribution_margin,
    avg(case when f.contribution < 0 then 1.0 else 0.0 end) as loss_making_share
from {{ ref('fct_offer_performance') }} f
group by f.campaign_id
