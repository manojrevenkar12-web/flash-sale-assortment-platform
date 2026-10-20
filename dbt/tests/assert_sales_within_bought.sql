-- Contract enforcement should make this impossible; the test proves it end to end.
select * from {{ ref('stg_offers') }} where units_sold > units_bought or units_returned > units_sold
