# custom_table.py
from astropy.table import QTable
import astropy.units as u

class CustomQTable(QTable):
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if isinstance(value, u.Quantity):
            self.update_metadata(key, value.unit)

    def add_column(self, column, index=None, name=None, copy=True, latex_name=None):
        super().add_column(column, index=index, name=name, copy=copy)
        if isinstance(column, u.Quantity):
            self.update_metadata(name, column.unit, latex_name)

    def update_metadata(self, column_name, unit = None, latex_name=None):
        if 'latex' not in self.meta:
            self.meta['latex'] = {}
        if latex_name is None:
            latex_name = f'${column_name}$'
        if unit is None:
            unit = self[column_name].unit
        self.meta['latex'][column_name] = {
            'latex_name': latex_name,
            'latex_unit': f'[{str(unit)}]'

        }

    def change_unit(self, column_name, new_unit):
        if column_name in self.colnames:
            self[column_name] = self[column_name].to(new_unit)
            self.update_metadata(column_name, new_unit)

    def change_latex_name(self, column_name, new_latex_name):
        if 'latex' in self.meta and column_name in self.meta['latex']:
            self.meta['latex'][column_name]['latex_name'] = new_latex_name

    def print_latex(self, column_name):
        return self.meta["latex"][column_name]["latex_name"] + " " + self.meta["latex"][column_name]["latex_unit"]

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        elif isinstance(key, int):
            return super().__getitem__(key)
        else:
            result = super().__getitem__(key)
            if 'latex' in self.meta:
                return self.meta['latex']
            else:
                return None
