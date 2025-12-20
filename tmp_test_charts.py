from app import charts_data
r = charts_data()
print(type(r))
print(list(r.get_json().keys()))
