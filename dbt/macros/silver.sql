{# Read a silver table: Parquet files locally, a Unity Catalog table on Databricks. #}
{% macro silver(name) -%}
  {%- if target.type == 'duckdb' -%}
    read_parquet('{{ env_var("FSAP_DATA_ROOT", "../data") }}/silver/{{ name }}/*.parquet')
  {%- else -%}
    lounge_assortment.silver.{{ name }}
  {%- endif -%}
{%- endmacro %}

{% macro safe_divide(num, den) -%}
  case when ({{ den }}) = 0 then null else ({{ num }}) * 1.0 / ({{ den }}) end
{%- endmacro %}

{# Schema names without the target prefix, so marts are always in `marts`. #}
{% macro generate_schema_name(custom_schema_name, node) -%}
  {{ custom_schema_name if custom_schema_name else target.schema }}
{%- endmacro %}
