-- What buyers look at when they negotiate the next lot: how a brand's category sold, week by week.
select
    brand_id,
    category,
    start_week,
    count(*) as offers,
    sum(units_bought) as units_bought,
    sum(units_sold) as units_sold,
    {{ safe_divide('sum(units_sold)', 'sum(units_bought)') }} as sell_through,
    avg(case when sold_out then 1.0 else 0.0 end) as sold_out_share,
    sum(contribution) as contribution
from {{ ref('fct_offer_performance') }}
group by brand_id, category, start_week
