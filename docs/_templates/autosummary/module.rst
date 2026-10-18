{{ fullname | escape | underline }}

{% if modules %}
.. automodule:: {{ fullname }}
   :no-members:

.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules if not item.endswith('__main__') %}
   {{ item }}
{%- endfor %}
{% else %}
.. automodule:: {{ fullname }}
{% endif %}
