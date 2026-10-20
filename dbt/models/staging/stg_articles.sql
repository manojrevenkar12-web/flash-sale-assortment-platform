select article_id, brand_id, category, season, gender, rrp, supplier_title
from {{ silver('articles') }}
