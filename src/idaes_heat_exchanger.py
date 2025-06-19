from idaes.core import FlowsheetBlock
from idaes.models.unit_models.heat_exchanger import HeatExchanger
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.environ import ConcreteModel, SolverFactory, value
from property_packages.build_package import build_package


# Create the ConcreteModel and Flowsheet
m = ConcreteModel()
m.fs = FlowsheetBlock(dynamic=False)
m.fs.water_properties = build_package("helmholtz",["water"],["Liq","Vap"])

# Add a Heat Exchanger unit
m.fs.heat_exchanger = HeatExchanger(
    hot_side={
        "property_package": m.fs.water_properties
    },
    cold_side={
        "property_package": m.fs.water_properties
    },
)

# Set inlet conditions for hot and cold sides
m.fs.heat_exchanger.hot_side_inlet.flow_mol.fix(10)      # mol/s
m.fs.heat_exchanger.hot_side.properties_in[0].constrain_component(
    m.fs.heat_exchanger.hot_side.properties_in[0].temperature,
    400  # K
)
m.fs.heat_exchanger.hot_side_inlet.pressure.fix(101325)  # Pa

m.fs.heat_exchanger.cold_side_inlet.flow_mol.fix(8)      # mol/s
m.fs.heat_exchanger.cold_side.properties_in[0].constrain_component(
    m.fs.heat_exchanger.cold_side.properties_in[0].temperature,
    300  # K
)
m.fs.heat_exchanger.cold_side_inlet.pressure.fix(101325) # Pa

# Set heat exchanger parameters
m.fs.heat_exchanger.area.fix(50)                   # m^2
m.fs.heat_exchanger.overall_heat_transfer_coefficient.fix(150)  # W/m^2/K

m.fs.heat_exchanger.initialize()

# Check degrees of freedom
print("Degrees of freedom:", degrees_of_freedom(m))

# Solve the model
solver = SolverFactory("ipopt")
results = solver.solve(m, tee=True)

# Display results
print("\nHot outlet temperature: {:.2f} K".format(
    value(m.fs.heat_exchanger.hot_side.properties_out[0].temperature)))
print("Cold outlet temperature: {:.2f} K".format(
    value(m.fs.heat_exchanger.cold_side.properties_out[0].temperature)))
