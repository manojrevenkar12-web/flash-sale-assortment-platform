select
    campaign_id,
    article_id,
    price,
    rrp,
    discount,
    unit_cost,
    units_bought,
    units_sold,
    units_returned,
    sold_out_day,
    sold_out_day >= 0 as sold_out
from {{ silver('campaign_offers') }}
