select campaign_id, article_id, day, units_sold
from {{ silver('daily_sales') }}
