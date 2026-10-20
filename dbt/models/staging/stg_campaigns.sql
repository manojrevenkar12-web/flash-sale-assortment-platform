select
    campaign_id,
    kind,
    theme,
    brand_id,
    start_date,
    end_date,
    duration_days,
    slots,
    visits
from {{ silver('campaigns') }}
